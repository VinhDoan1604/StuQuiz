/**
 * frontend/src/services/api.ts
 * API Client kết nối trực tiếp với backend FastAPI & SSE Streaming
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Citation {
  file?: string;
  page?: number | null;
  section?: string;
}

export interface ReflectionTrace {
  context_grade: number | null;
  context_sufficient: boolean | null;
  rewritten_queries: string[];
  answer_grade: number | null;
  answer_faithful: boolean | null;
  num_loops: number;
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
  question: string;
  choices: string[];
  correct_choice: string;
  explanation: string;
  topic: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

export interface MasteryTopic {
  topic: string;
  mastery_score: number;
  streak: number;
  last_reviewed_at: string | null;
  next_review_due_at: string | null;
}

// ---------------------------------------------------------------------------
// 1. Chat & SSE Streaming
// ---------------------------------------------------------------------------

export async function sendChatMessage(
  sessionId: string, 
  message: string,
  documentIds?: string[],
  spaceId?: string,
  userId?: string
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      session_id: sessionId, 
      message,
      document_ids: documentIds && documentIds.length > 0 ? documentIds : undefined,
      space_id: spaceId || undefined,
      user_id: userId || undefined,
    }),
  });
  if (!res.ok) {
    throw new Error(`Chat error: ${res.statusText}`);
  }
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
  userId?: string
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        session_id: sessionId, 
        message,
        document_ids: documentIds && documentIds.length > 0 ? documentIds : undefined,
        space_id: spaceId || undefined,
        user_id: userId || undefined,
      }),
    });

    if (!res.ok) {
      throw new Error(`Stream connection failed: ${res.statusText}`);
    }

    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error('ReadableStream not supported by browser');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const block of lines) {
        const trimmed = block.trim();
        if (trimmed.startsWith('data: ')) {
          try {
            const parsed: AgentStepEvent = JSON.parse(trimmed.slice(6));
            if (parsed.event === 'done' && parsed.response) {
              onDone(parsed.response);
            } else if (parsed.event === 'error') {
              onError(parsed.message || 'Stream server error');
            } else {
              onStep(parsed);
            }
          } catch (e) {
            console.warn('Failed to parse SSE payload:', trimmed, e);
          }
        }
      }
    }
  } catch (err: any) {
    onError(err.message || 'Lỗi mạng khi kết nối stream');
  }
}

// ---------------------------------------------------------------------------
// 2. Documents Management
// ---------------------------------------------------------------------------

export async function fetchDocuments(userId?: string): Promise<DocumentItem[]> {
  try {
    const url = userId ? `${API_BASE}/documents?user_id=${userId}` : `${API_BASE}/documents`;
    const res = await fetch(url, {
      headers: userId ? { 'X-User-Id': userId } : {}
    });
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchDocuments error:', err);
    return [];
  }
}

export async function uploadDocument(
  file: File,
  userId?: string
): Promise<{ document_id: string; num_chunks: number; status?: string; filename?: string; message?: string }> {
  const formData = new FormData();
  formData.append('file', file);
  if (userId) {
    formData.append('user_id', userId);
  }

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    headers: userId ? { 'X-User-Id': userId } : {},
    body: formData,
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Upload failed: ${msg}`);
  }
  return res.json();
}

export async function fetchDocumentStatus(documentId: string): Promise<{
  id: string;
  filename: string;
  status: 'processing' | 'ready' | 'failed';
  num_chunks: number;
  error?: string | null;
}> {
  const res = await fetch(`${API_BASE}/documents/${documentId}/status`);
  if (!res.ok) throw new Error('Không thể kiểm tra trạng thái tài liệu');
  return res.json();
}

export async function deleteDocument(documentId: string, userId?: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: 'DELETE',
    headers: userId ? { 'X-User-Id': userId } : {},
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Xóa tài liệu thất bại: ${msg}`);
  }
}

// ---------------------------------------------------------------------------
// 3. Quiz & Mastery
// ---------------------------------------------------------------------------

export async function generateQuiz(
  documentIds: string[],
  numQuestions: number = 5,
  prioritizeWeakTopics: boolean = true,
  userId: string = 'demo_user',
  topicPrompt?: string,
  spaceId?: string
): Promise<QuizQuestion[]> {
  const res = await fetch(`${API_BASE}/quiz/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      document_ids: documentIds,
      num_questions: numQuestions,
      prioritize_weak_topics: prioritizeWeakTopics,
      user_id: userId,
      topic_prompt: topicPrompt?.trim() || undefined,
      space_id: spaceId || undefined,
    }),
  });

  if (!res.ok) {
    throw new Error(`Quiz generate error: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchWeakTopics(userId: string = 'demo_user'): Promise<MasteryTopic[]> {
  try {
    const res = await fetch(`${API_BASE}/mastery/weak-topics/${userId}`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchWeakTopics error:', err);
    return [];
  }
}

export async function fetchAllMastery(userId: string = 'demo_user'): Promise<MasteryTopic[]> {
  try {
    const res = await fetch(`${API_BASE}/mastery/all/${userId}`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchAllMastery error:', err);
    return [];
  }
}

export async function submitQuizResult(
  topic: string,
  isCorrect: boolean,
  userId: string = 'demo_user'
): Promise<void> {
  await fetch(`${API_BASE}/mastery/record`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      topic,
      is_correct: isCorrect,
    }),
  });
}

// ---------------------------------------------------------------------------
// 4. Authentication (Góp ý #4: OTP Gmail + Google Auth)
// ---------------------------------------------------------------------------

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

export async function loginUser(email: string, password: string): Promise<AuthResult> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Đăng nhập thất bại' }));
    throw new Error(err.detail || 'Đăng nhập thất bại');
  }
  return res.json();
}

export async function registerUser(email: string, password: string, fullName?: string): Promise<AuthResult> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, full_name: fullName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Đăng ký thất bại' }));
    throw new Error(err.detail || 'Đăng ký thất bại');
  }
  return res.json();
}

export async function verifyOtp(email: string, code: string): Promise<AuthResult> {
  const res = await fetch(`${API_BASE}/auth/verify-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, code }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Mã xác thực không hợp lệ' }));
    throw new Error(err.detail || 'Mã xác thực không hợp lệ');
  }
  return res.json();
}

export async function resendOtp(email: string): Promise<AuthResult> {
  const res = await fetch(`${API_BASE}/auth/resend-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Không thể gửi lại mã xác thực' }));
    throw new Error(err.detail || 'Không thể gửi lại mã xác thực');
  }
  return res.json();
}

export async function googleAuth(data: {
  credential?: string;
  access_token?: string;
  code?: string;
  redirect_uri?: string;
  email?: string;
  full_name?: string;
  google_id?: string;
}): Promise<AuthResult> {
  const res = await fetch(`${API_BASE}/auth/google`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Đăng nhập bằng Google thất bại' }));
    throw new Error(err.detail || 'Đăng nhập bằng Google thất bại');
  }
  return res.json();
}

export async function logoutUser(): Promise<void> {
  try {
    await fetch(`${API_BASE}/auth/logout`, { method: 'POST' });
  } catch {
    // Ignore errors on logout
  }
}

export async function fetchCurrentUser(userId?: string): Promise<UserProfile> {
  const url = userId ? `${API_BASE}/auth/me?user_id=${userId}` : `${API_BASE}/auth/me`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Không thể lấy thông tin người dùng');
  return res.json();
}

// ---------------------------------------------------------------------------
// 5. Study Spaces / Bộ Học Tập (Góp ý #1, #6 & #7)
// ---------------------------------------------------------------------------

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

export async function fetchSpaces(userId?: string): Promise<StudySpaceItem[]> {
  try {
    const url = userId ? `${API_BASE}/spaces?user_id=${userId}` : `${API_BASE}/spaces`;
    const res = await fetch(url);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchSpaces error:', err);
    return [];
  }
}

export async function createSpace(title: string, description?: string, userId?: string): Promise<StudySpaceItem> {
  const res = await fetch(`${API_BASE}/spaces`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, description, user_id: userId }),
  });
  if (!res.ok) {
    throw new Error('Không thể tạo bộ học tập mới');
  }
  return res.json();
}

export async function updateSpace(spaceId: string, updates: { title?: string; description?: string; icon?: string }): Promise<StudySpaceItem> {
  const res = await fetch(`${API_BASE}/spaces/${spaceId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
  if (!res.ok) {
    throw new Error('Không thể cập nhật bộ học tập');
  }
  return res.json();
}

export async function getSpaceDetail(spaceId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/spaces/${spaceId}`);
  if (!res.ok) throw new Error('Không tìm thấy bộ học tập');
  return res.json();
}

export async function attachDocumentToSpace(spaceId: string, documentId: string): Promise<void> {
  await fetch(`${API_BASE}/spaces/${spaceId}/documents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId }),
  });
}

export async function detachDocumentFromSpace(spaceId: string, documentId: string): Promise<void> {
  await fetch(`${API_BASE}/spaces/${spaceId}/documents/${documentId}`, {
    method: 'DELETE',
  });
}

export async function deleteSpace(spaceId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/spaces/${spaceId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    throw new Error('Không thể xóa bộ học tập');
  }
}

export async function fetchSpaceChatHistory(spaceId: string): Promise<any[]> {
  try {
    const res = await fetch(`${API_BASE}/spaces/${spaceId}/chat-history`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchSpaceChatHistory error:', err);
    return [];
  }
}

export async function fetchSpaceQuizHistory(spaceId: string): Promise<any[]> {
  try {
    const res = await fetch(`${API_BASE}/spaces/${spaceId}/quiz-history`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('fetchSpaceQuizHistory error:', err);
    return [];
  }
}

