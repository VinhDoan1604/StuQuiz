/**
 * frontend/src/services/api.ts
 * API client: gắn JWT vào mọi request cần đăng nhập; tự phát sự kiện
 * 'auth:expired' khi server trả 401 để App đăng xuất.
 *
 * Các hàm giữ NGUYÊN chữ ký cũ (tham số userId được bỏ qua — danh tính do
 * server lấy từ token), nên component hiện có không cần sửa.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TOKEN_KEY = 'study_token';

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);

async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(url, { ...init, headers });
  if (res.status === 401) window.dispatchEvent(new Event('auth:expired'));
  return res;
}

const jsonInit = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: body === undefined ? undefined : JSON.stringify(body),
});

async function errorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === 'string') return data.detail;
  } catch {
    /* ignore */
  }
  return fallback;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface Citation { 
  index?: number; 
  file?: string; 
  page?: number | null; 
  section?: string; 
  snippet?: string; 
}

export interface ReflectionTrace {
  context_grade: number | null;
  context_sufficient: boolean | null;
  rewritten_queries: string[];
  answer_grade: number | null;
  answer_faithful: boolean | null;
  num_loops: number;
  context_verified?: boolean | null; 
  context_forced?: boolean | null;
  answer_verified?: boolean | null;  
  answer_forced?: boolean | null;
}

export interface ChatResponse {
  intent: 'qa' | 'teaching' | 'quiz' | 'summary';
  answer: string;
  citations: Citation[];
  reflection: ReflectionTrace;
}

export interface AgentStepEvent {
  event: 'status' | 'step' | 'done' | 'error';
  step?: string;
  node?: string;
  title?: string;
  message?: string;
  intent?: string;
  review_topic?: string | null;
  chunks_found?: number;
  rewritten_queries?: string[];
  grade?: number;
  sufficient?: boolean;
  faithful?: boolean;
  feedback?: string;
  loop?: number;
  citations?: Citation[];
  response?: ChatResponse;
  graded?: boolean; 
  forced?: boolean;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_type?: string;
  subject?: string | null;
  num_chunks: number;
  uploaded_at: string | null;
  status?: 'processing' | 'ready' | 'failed';
  error_message?: string | null;
}

export interface QuizQuestion {
  attempt_id: string; 
  question: string; 
  choices: string[]; 
  topic: string;
  difficulty: 'easy' | 'medium' | 'hard';
}
export interface QuizAnswerResult { 
  attempt_id: string; 
  is_correct: boolean; 
  correct_choice: string; 
  explanation: string; 
}


export interface MasteryTopic {
  topic: string;
  mastery_score: number;
  streak: number;
  last_reviewed_at: string | null;
  next_review_due_at: string | null;
}

export interface UserProfile {
  id: string;
  email: string;
  full_name?: string | null;
  auth_provider?: string | null;
  is_verified?: boolean;
  created_at?: string | null;
}

export interface AuthResult {
  status: 'success' | 'needs_verification';
  message: string;
  email?: string;
  user?: UserProfile;
  token?: string;
  dev_code?: string;
}

export interface StudySpaceItem {
  id: string;
  title: string;
  description?: string | null;
  icon?: string;
  user_id?: string | null;
  document_ids: string[];
  num_documents: number;
  created_at?: string | null;
}

// ---------------------------------------------------------------------------
// 1. Chat & SSE
// ---------------------------------------------------------------------------
function chatBody(
  sessionId: string,
  message: string,
  documentIds?: string[],
  spaceId?: string
) {
  return {
    session_id: sessionId,
    message,
    document_ids: documentIds && documentIds.length > 0 ? documentIds : undefined,
    space_id: spaceId || undefined,
  };
}

export async function sendChatMessage(
  sessionId: string,
  message: string,
  documentIds?: string[],
  spaceId?: string,
  _userId?: string
): Promise<ChatResponse> {
  const res = await authFetch(
    `${API_BASE}/chat`,
    jsonInit('POST', chatBody(sessionId, message, documentIds, spaceId))
  );
  if (!res.ok) throw new Error(await errorDetail(res, `Chat error: ${res.statusText}`));
  return res.json();
}

export async function streamChatMessage(
  sessionId: string,
  message: string,
  onStep: (event: AgentStepEvent) => void,
  onDone: (response: ChatResponse) => void,
  onError: (err: string) => void,
  documentIds?: string[],
  spaceId?: string,
  _userId?: string
): Promise<void> {
  try {
    const res = await authFetch(
      `${API_BASE}/chat/stream`,
      jsonInit('POST', chatBody(sessionId, message, documentIds, spaceId))
    );
    if (!res.ok) throw new Error(await errorDetail(res, `Stream connection failed: ${res.statusText}`));

    const reader = res.body?.getReader();
    if (!reader) throw new Error('ReadableStream not supported by browser');

    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';

      for (const block of blocks) {
        const trimmed = block.trim();
        if (!trimmed.startsWith('data: ')) continue;
        try {
          const parsed: AgentStepEvent = JSON.parse(trimmed.slice(6));
          if (parsed.event === 'done' && parsed.response) onDone(parsed.response);
          else if (parsed.event === 'error') onError(parsed.message || 'Stream server error');
          else onStep(parsed);
        } catch (e) {
          console.warn('Failed to parse SSE payload:', trimmed, e);
        }
      }
    }
  } catch (err: any) {
    onError(err.message || 'Lỗi mạng khi kết nối stream');
  }
}

// ---------------------------------------------------------------------------
// 2. Documents
// ---------------------------------------------------------------------------
export async function fetchDocuments(_userId?: string): Promise<DocumentItem[]> {
  try {
    const res = await authFetch(`${API_BASE}/documents`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchDocuments error:', err);
    return [];
  }
}

export async function uploadDocument(
  file: File,
  _userId?: string
): Promise<{ document_id: string; num_chunks: number; status?: string; filename?: string; message?: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await authFetch(`${API_BASE}/documents/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await errorDetail(res, 'Tải tệp lên thất bại'));
  return res.json();
}

export async function fetchDocumentStatus(documentId: string): Promise<{
  id: string;
  filename: string;
  status: 'processing' | 'ready' | 'failed';
  num_chunks: number;
  error?: string | null;
}> {
  const res = await authFetch(`${API_BASE}/documents/${documentId}/status`);
  if (!res.ok) throw new Error('Không thể kiểm tra trạng thái tài liệu');
  return res.json();
}

export async function deleteDocument(documentId: string, _userId?: string): Promise<void> {
  const res = await authFetch(`${API_BASE}/documents/${documentId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await errorDetail(res, 'Xóa tài liệu thất bại'));
}

// ---------------------------------------------------------------------------
// 3. Quiz & Mastery
// ---------------------------------------------------------------------------
export async function generateQuiz(
  documentIds: string[],
  numQuestions: number = 5,
  prioritizeWeakTopics: boolean = true,
  _userId: string = '',
  topicPrompt?: string,
  spaceId?: string
): Promise<QuizQuestion[]> {
  const res = await authFetch(
    `${API_BASE}/quiz/generate`,
    jsonInit('POST', {
      document_ids: documentIds,
      num_questions: numQuestions,
      prioritize_weak_topics: prioritizeWeakTopics,
      topic_prompt: topicPrompt?.trim() || undefined,
      space_id: spaceId || undefined,
    })
  );
  if (!res.ok) throw new Error(await errorDetail(res, `Quiz generate error: ${res.statusText}`));
  return res.json();
}

export async function fetchWeakTopics(_userId?: string): Promise<MasteryTopic[]> {
  try {
    const res = await authFetch(`${API_BASE}/mastery/weak-topics`);
    return res.ok ? await res.json() : [];
  } catch (err) {
    console.warn('fetchWeakTopics error:', err);
    return [];
  }
}

export async function fetchAllMastery(_userId?: string): Promise<MasteryTopic[]> {
  try {
    const res = await authFetch(`${API_BASE}/mastery/all`);
    return res.ok ? await res.json() : [];
  } catch (err) {
    console.warn('fetchAllMastery error:', err);
    return [];
  }
}

export async function answerQuiz(attemptId: string, choice: string): Promise<QuizAnswerResult> {
  const res = await authFetch(`${API_BASE}/quiz/answer`, jsonInit('POST', { attempt_id: attemptId, choice }));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không ghi nhận được câu trả lời'));
  return res.json();
}

// ---------------------------------------------------------------------------
// 4. Authentication (endpoint đăng nhập dùng fetch thường: 401 ở đây KHÔNG được
//    kích hoạt đăng xuất toàn cục)
// ---------------------------------------------------------------------------
async function authPost(path: string, body?: unknown, fallback = 'Yêu cầu thất bại'): Promise<AuthResult> {
  const res = await fetch(`${API_BASE}${path}`, jsonInit('POST', body));
  if (!res.ok) throw new Error(await errorDetail(res, fallback));
  return res.json();
}

export const loginUser = (email: string, password: string) =>
  authPost('/auth/login', { email, password }, 'Đăng nhập thất bại');

export const registerUser = (email: string, password: string, fullName?: string) =>
  authPost('/auth/register', { email, password, full_name: fullName }, 'Đăng ký thất bại');

export const verifyOtp = (email: string, code: string) =>
  authPost('/auth/verify-otp', { email, code }, 'Mã xác thực không hợp lệ');

export const resendOtp = (email: string) =>
  authPost('/auth/resend-otp', { email }, 'Không thể gửi lại mã xác thực');

export const demoLogin = () => authPost('/auth/demo', undefined, 'Không thể vào chế độ dùng thử');

export const googleAuth = (data: {
  credential?: string;
  access_token?: string;
  code?: string;
  redirect_uri?: string;
}) => authPost('/auth/google', data, 'Đăng nhập bằng Google thất bại');

export async function logoutUser(): Promise<void> {
  try {
    await authFetch(`${API_BASE}/auth/logout`, { method: 'POST' });
  } catch {
    /* ignore */
  }
}

export async function fetchCurrentUser(_userId?: string): Promise<UserProfile> {
  const res = await authFetch(`${API_BASE}/auth/me`);
  if (!res.ok) throw new Error('Không thể lấy thông tin người dùng');
  return res.json();
}

// ---------------------------------------------------------------------------
// 5. Study Spaces
// ---------------------------------------------------------------------------
export async function fetchSpaces(_userId?: string): Promise<StudySpaceItem[]> {
  try {
    const res = await authFetch(`${API_BASE}/spaces`);
    return res.ok ? await res.json() : [];
  } catch (err) {
    console.warn('fetchSpaces error:', err);
    return [];
  }
}

export async function createSpace(title: string, description?: string, _userId?: string): Promise<StudySpaceItem> {
  const res = await authFetch(`${API_BASE}/spaces`, jsonInit('POST', { title, description }));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể tạo bộ học tập mới'));
  return res.json();
}

export async function updateSpace(
  spaceId: string,
  updates: { title?: string; description?: string; icon?: string }
): Promise<StudySpaceItem> {
  const res = await authFetch(`${API_BASE}/spaces/${spaceId}`, jsonInit('PATCH', updates));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể cập nhật bộ học tập'));
  return res.json();
}

export async function getSpaceDetail(spaceId: string): Promise<any> {
  const res = await authFetch(`${API_BASE}/spaces/${spaceId}`);
  if (!res.ok) throw new Error('Không tìm thấy bộ học tập');
  return res.json();
}

export async function attachDocumentToSpace(spaceId: string, documentId: string): Promise<void> {
  const res = await authFetch(
    `${API_BASE}/spaces/${spaceId}/documents`,
    jsonInit('POST', { document_id: documentId })
  );
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể gắn tài liệu'));
}

export async function detachDocumentFromSpace(spaceId: string, documentId: string): Promise<void> {
  await authFetch(`${API_BASE}/spaces/${spaceId}/documents/${documentId}`, { method: 'DELETE' });
}

export async function deleteSpace(spaceId: string): Promise<void> {
  const res = await authFetch(`${API_BASE}/spaces/${spaceId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Không thể xóa bộ học tập');
}

export async function fetchSpaceChatHistory(spaceId: string): Promise<any[]> {
  try {
    const res = await authFetch(`${API_BASE}/spaces/${spaceId}/chat-history`);
    return res.ok ? await res.json() : [];
  } catch (err) {
    console.warn('fetchSpaceChatHistory error:', err);
    return [];
  }
}

export async function fetchSpaceQuizHistory(spaceId: string): Promise<any[]> {
  try {
    const res = await authFetch(`${API_BASE}/spaces/${spaceId}/quiz-history`);
    return res.ok ? await res.json() : [];
  } catch (err) {
    console.warn('fetchSpaceQuizHistory error:', err);
    return [];
  }
}
