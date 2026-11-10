/**
 * frontend/src/services/api.ts
 * API client duy nhất của frontend:
 * - Gắn JWT vào mọi request cần đăng nhập; server trả 401 => phát sự kiện
 *   'auth:expired' để App đăng xuất.
 * - Danh tính user do server lấy từ token — client KHÔNG gửi user_id.
 * - Mặc định gọi `/api` (cùng origin: Vite proxy khi dev, nginx khi deploy).
 */

const API_BASE = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');
const TOKEN_KEY = 'study_token';
const USER_KEY = 'study_user';

// ---------------------------------------------------------------------------
// Session (localStorage)
// ---------------------------------------------------------------------------
export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);

export function getStoredUser(): UserProfile | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw && getToken() ? (JSON.parse(raw) as UserProfile) : null;
  } catch {
    return null;
  }
}

export function saveSession(user: UserProfile, token: string): void {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearSession(): void {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
}

/** UUID v4 cho phiên chat mới (crypto.randomUUID chỉ có trong secure context). */
export function newSessionId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

export const errorMessage = (err: unknown, fallback: string): string =>
  err instanceof Error && err.message ? err.message : fallback;

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------
async function authFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
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

async function getJson<T>(path: string, fallback: string): Promise<T> {
  const res = await authFetch(path);
  if (!res.ok) throw new Error(await errorDetail(res, fallback));
  return res.json();
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

export type Intent = 'qa' | 'teaching' | 'quiz' | 'summary';

export interface ChatResponse {
  session_id?: string | null;
  turn_id?: string | null;
  intent: Intent;
  answer: string;
  citations: Citation[];
  reflection: ReflectionTrace;
}

export interface AgentStepEvent {
  event: 'status' | 'step' | 'done' | 'error' | 'token';
  text?: string;
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

export type DocumentStatus = 'processing' | 'ready' | 'failed';

export interface DocumentItem {
  id: string;
  filename: string;
  file_type?: string;
  subject?: string | null;
  num_chunks: number;
  uploaded_at: string | null;
  status?: DocumentStatus;
  error_message?: string | null;
}

export interface UploadResult {
  document_id: string;
  num_chunks: number;
  status?: DocumentStatus;
  filename?: string;
  message?: string;
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

export interface QuizHistoryItem {
  id: string;
  topic: string | null;
  question: string;
  choices: string[];
  correct_choice: string;
  user_choice: string | null;
  is_correct: boolean | null;
  created_at: string | null;
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

export interface ChatTurnItem {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  intent: Intent | null;
  reflection_trace: (ReflectionTrace & { citations?: Citation[] }) | null;
  feedback?: number | null;
  created_at: string | null;
}

export interface ChatSessionItem {
  session_id: string;
  title: string;
  created_at: string | null;
  turns_count: number;
  turns: ChatTurnItem[];
}

// ---------------------------------------------------------------------------
// 1. Chat & SSE
// ---------------------------------------------------------------------------
/** Phạm vi tài liệu do server tự xác định từ space (hoặc toàn bộ tài liệu nếu không có space). */
function chatBody(sessionId: string, message: string, spaceId?: string, documentIds?: string[], model?: string) {
  return {
    session_id: sessionId,
    message,
    space_id: spaceId || undefined,
    document_ids: documentIds && documentIds.length > 0 ? documentIds : undefined,
    model: model || undefined,
  };
}

/** Model user được chọn khi chat (cấu hình LLM_CHAT_MODELS ở backend). */
export interface ChatModelItem {
  id: string;        // "<provider>:<model>"
  provider: string;
  model: string;
  label: string;
}

export const fetchChatModels = (): Promise<{ models: ChatModelItem[] }> =>
  getJson('/chat/models', 'Không tải được danh sách model');

export async function sendChatMessage(
  sessionId: string,
  message: string,
  spaceId?: string,
  documentIds?: string[],
  model?: string
): Promise<ChatResponse> {
  const res = await authFetch('/chat', jsonInit('POST', chatBody(sessionId, message, spaceId, documentIds, model)));
  if (!res.ok) throw new Error(await errorDetail(res, `Lỗi chat: ${res.statusText}`));
  return res.json();
}

export async function rateTurn(turnId: string, rating: -1 | 0 | 1): Promise<void> {
  const res = await authFetch(`/chat/turns/${turnId}/feedback`, jsonInit('POST', { rating }));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không gửi được đánh giá'));
}

export interface StreamHandlers {
  onStep: (event: AgentStepEvent) => void;
  onToken?: (text: string) => void;
  onDone: (response: ChatResponse) => void;
  onError: (err: string) => void;
}

export async function streamChatMessage(
  sessionId: string,
  message: string,
  { onStep, onToken, onDone, onError }: StreamHandlers,
  spaceId?: string,
  documentIds?: string[],
  model?: string
): Promise<void> {
  let finished = false;
  try {
    const res = await authFetch('/chat/stream', jsonInit('POST', chatBody(sessionId, message, spaceId, documentIds, model)));
    if (!res.ok) throw new Error(await errorDetail(res, `Không kết nối được luồng phản hồi: ${res.statusText}`));

    const reader = res.body?.getReader();
    if (!reader) throw new Error('Trình duyệt không hỗ trợ ReadableStream');

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
        let parsed: AgentStepEvent;
        try {
          parsed = JSON.parse(trimmed.slice(6));
        } catch (e) {
          console.warn('Không parse được SSE payload:', trimmed, e);
          continue;
        }
        if (parsed.event === 'token') {
          onToken?.(parsed.text || '');
        } else if (parsed.event === 'done' && parsed.response) {
          finished = true;
          onDone(parsed.response);
        } else if (parsed.event === 'error') {
          finished = true;
          onError(parsed.message || 'Máy chủ gặp lỗi khi xử lý.');
        } else {
          onStep(parsed);
        }
      }
    }
    if (!finished) onError('Kết nối bị ngắt trước khi nhận được câu trả lời.');
  } catch (err) {
    if (!finished) onError(errorMessage(err, 'Lỗi mạng khi kết nối luồng phản hồi'));
  }
}

// ---------------------------------------------------------------------------
// 2. Documents
// ---------------------------------------------------------------------------
export const fetchDocuments = (): Promise<DocumentItem[]> =>
  getJson('/documents', 'Không tải được danh sách tài liệu');

export async function uploadDocument(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await authFetch('/documents/upload', { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await errorDetail(res, 'Tải tệp lên thất bại'));
  return res.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await authFetch(`/documents/${documentId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await errorDetail(res, 'Xóa tài liệu thất bại'));
}

// ---------------------------------------------------------------------------
// 3. Quiz & Mastery
// ---------------------------------------------------------------------------
export interface GenerateQuizOptions {
  documentIds: string[];
  numQuestions?: number;
  prioritizeWeakTopics?: boolean;
  topicPrompt?: string;
  spaceId?: string;
}

export async function generateQuiz({
  documentIds,
  numQuestions = 5,
  prioritizeWeakTopics = true,
  topicPrompt,
  spaceId,
}: GenerateQuizOptions): Promise<QuizQuestion[]> {
  const res = await authFetch(
    '/quiz/generate',
    jsonInit('POST', {
      document_ids: documentIds,
      num_questions: numQuestions,
      prioritize_weak_topics: prioritizeWeakTopics,
      topic_prompt: topicPrompt?.trim() || undefined,
      space_id: spaceId || undefined,
    })
  );
  if (!res.ok) throw new Error(await errorDetail(res, `Không sinh được đề: ${res.statusText}`));
  return res.json();
}

export async function answerQuiz(attemptId: string, choice: string): Promise<QuizAnswerResult> {
  const res = await authFetch('/quiz/answer', jsonInit('POST', { attempt_id: attemptId, choice }));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không ghi nhận được câu trả lời'));
  return res.json();
}

export const fetchWeakTopics = (): Promise<MasteryTopic[]> =>
  getJson('/mastery/weak-topics', 'Không tải được chủ đề cần ôn');

export const fetchAllMastery = (): Promise<MasteryTopic[]> =>
  getJson('/mastery/all', 'Không tải được tiến độ học tập');

// ---------------------------------------------------------------------------
// 4. Authentication (fetch thường: 401 ở đây KHÔNG kích hoạt đăng xuất toàn cục)
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

export const googleAuth = (data: { credential?: string; access_token?: string; code?: string; redirect_uri?: string }) =>
  authPost('/auth/google', data, 'Đăng nhập bằng Google thất bại');

/**
 * Client ID Google OAuth do backend cấu hình: trả `null` nếu máy chủ CHƯA bật Google;
 * ném lỗi nếu không kết nối được máy chủ (để UI không báo nhầm là "chưa bật").
 */
export async function fetchGoogleClientId(): Promise<string | null> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/auth/google/config`);
  } catch {
    throw new Error('Không kết nối được máy chủ. Vui lòng kiểm tra backend đang chạy rồi thử lại.');
  }
  if (!res.ok) throw new Error(`Máy chủ không phản hồi (HTTP ${res.status}). Vui lòng thử lại sau.`);
  const data: { client_id?: string | null } = await res.json();
  return data.client_id || null;
}

export async function logoutUser(): Promise<void> {
  try {
    await authFetch('/auth/logout', { method: 'POST' });
  } catch {
    /* ignore */
  }
}

export const fetchCurrentUser = (): Promise<UserProfile> =>
  getJson('/auth/me', 'Không thể lấy thông tin người dùng');

// ---------------------------------------------------------------------------
// 5. Study Spaces
// ---------------------------------------------------------------------------
export const fetchSpaces = (): Promise<StudySpaceItem[]> =>
  getJson('/spaces', 'Không tải được danh sách không gian học tập');

export async function createSpace(title: string, description?: string): Promise<StudySpaceItem> {
  const res = await authFetch('/spaces', jsonInit('POST', { title, description }));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể tạo không gian học tập mới'));
  return res.json();
}

export async function updateSpace(
  spaceId: string,
  updates: { title?: string; description?: string; icon?: string }
): Promise<StudySpaceItem> {
  const res = await authFetch(`/spaces/${spaceId}`, jsonInit('PATCH', updates));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể cập nhật không gian học tập'));
  return res.json();
}

export async function attachDocumentToSpace(spaceId: string, documentId: string): Promise<void> {
  const res = await authFetch(`/spaces/${spaceId}/documents`, jsonInit('POST', { document_id: documentId }));
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể gắn tài liệu'));
}

export async function detachDocumentFromSpace(spaceId: string, documentId: string): Promise<void> {
  const res = await authFetch(`/spaces/${spaceId}/documents/${documentId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể gỡ tài liệu'));
}

export async function deleteSpace(spaceId: string): Promise<void> {
  const res = await authFetch(`/spaces/${spaceId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await errorDetail(res, 'Không thể xóa không gian học tập'));
}

export const fetchSpaceChatHistory = (spaceId: string): Promise<ChatSessionItem[]> =>
  getJson(`/spaces/${spaceId}/chat-history`, 'Không tải được lịch sử chat');

export const fetchSpaceQuizHistory = (spaceId: string): Promise<QuizHistoryItem[]> =>
  getJson(`/spaces/${spaceId}/quiz-history`, 'Không tải được lịch sử quiz');
