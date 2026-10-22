import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatView } from './components/ChatView';
import { QuizView } from './components/QuizView';
import { AuthScreen } from './components/AuthScreen';
import { CollectionsDashboard } from './components/CollectionsDashboard';
import { 
  fetchDocuments, 
  fetchSpaces, 
  attachDocumentToSpace, 
  detachDocumentFromSpace,
  type DocumentItem, 
  type StudySpaceItem, 
  type UserProfile 
} from './services/api';

export function App() {
  // Authentication State
  const [user, setUser] = useState<UserProfile | null>(() => {
    try {
      const saved = localStorage.getItem('study_user');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [, setToken] = useState<string | null>(() => {
    return localStorage.getItem('study_token') || null;
  });

  // Study Spaces & Documents State
  const [spaces, setSpaces] = useState<StudySpaceItem[]>([]);
  const [activeSpace, setActiveSpace] = useState<StudySpaceItem | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);

  // Tab inside an active notebook (only Chat or Quiz)
  const [activeTab, setActiveTab] = useState<'chat' | 'quiz'>('chat');

  const loadDocuments = async (userId?: string) => {
    setIsLoadingDocs(true);
    try {
      const uid = userId || user?.id;
      const docs = await fetchDocuments(uid);
      setDocuments(docs);
    } catch (e) {
      console.warn('Failed to load documents:', e);
    } finally {
      setIsLoadingDocs(false);
    }
  };

  const loadSpaces = async (userId?: string) => {
    try {
      const spList = await fetchSpaces(userId || user?.id);
      setSpaces(spList);
      // Giữ activeSpace đồng bộ nếu đã chọn
      if (activeSpace) {
        const updated = spList.find(s => s.id === activeSpace.id);
        if (updated) setActiveSpace(updated);
      }
    } catch (e) {
      console.warn('Failed to load study spaces:', e);
    }
  };

  useEffect(() => {
    if (user) {
      loadDocuments(user.id);
      loadSpaces(user.id);
    }
  }, [user]);

  const handleLoginSuccess = (userProfile: UserProfile, authToken: string) => {
    setUser(userProfile);
    setToken(authToken);
    localStorage.setItem('study_user', JSON.stringify(userProfile));
    localStorage.setItem('study_token', authToken);
    loadDocuments(userProfile.id);
    loadSpaces(userProfile.id);
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    setActiveSpace(null);
    setSpaces([]);
    setDocuments([]);
    localStorage.removeItem('study_user');
    localStorage.removeItem('study_token');
  };

  const handleAttachDoc = async (docId: string) => {
    if (!activeSpace) return;
    try {
      await attachDocumentToSpace(activeSpace.id, docId);
      await loadSpaces();
      await loadDocuments();
    } catch (e) {
      console.error('Failed to attach document:', e);
    }
  };

  const handleDetachDoc = async (docId: string) => {
    if (!activeSpace) return;
    try {
      await detachDocumentFromSpace(activeSpace.id, docId);
      await loadSpaces();
      await loadDocuments();
    } catch (e) {
      console.error('Failed to detach document:', e);
    }
  };

  useEffect(() => {
    const onExpired = () => handleLogout();
    window.addEventListener('auth:expired', onExpired);
    // Có user lưu sẵn nhưng mất token => đăng xuất sạch
    if (user && !localStorage.getItem('study_token')) handleLogout();
    return () => window.removeEventListener('auth:expired', onExpired);
  }, []);

  // 1. GIAO DIỆN BAN ĐẦU: ĐĂNG NHẬP (Nếu chưa đăng nhập)
  if (!user) {
    return <AuthScreen onSuccess={handleLoginSuccess} />;
  }

  // Lọc danh sách tài liệu chỉ thuộc về Study Space đang chọn
  const activeSpaceDocs = activeSpace && activeSpace.document_ids && activeSpace.document_ids.length > 0
    ? documents.filter(d => activeSpace.document_ids.includes(d.id))
    : documents;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-base)' }}>
      {/* Header */}
      <Header 
        activeSpace={activeSpace}
        onBackToDashboard={() => setActiveSpace(null)}
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        user={user}
        onLogout={handleLogout}
      />

      {/* 2. ĐĂNG NHẬP XONG VÀO TRANG CHÍNH: HIỆN CÁC SỔ (Ô đầu là Tạo sổ) */}
      {!activeSpace ? (
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CollectionsDashboard
            spaces={spaces}
            onSelectSpace={(space) => {
              setActiveSpace(space);
              setActiveTab('chat');
            }}
            onRefreshSpaces={() => loadSpaces(user?.id)}
            user={user}
          />
        </main>
      ) : (
        /* 3. BẤM VÀO SỔ MỚI RA KHUNG CHAT / QUIZ CỦA SỔ ĐÓ */
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {activeTab === 'chat' && (
            <ChatView 
              documents={activeSpaceDocs}
              allDocuments={documents}
              isLoadingDocs={isLoadingDocs}
              onRefreshDocs={async () => {
                await loadDocuments();
                await loadSpaces(user?.id);
              }}
              activeSpace={activeSpace}
              onAttachDoc={handleAttachDoc}
              onDetachDoc={handleDetachDoc}
              user={user}
            />
          )}

          {activeTab === 'quiz' && (
            <div style={{ padding: '24px 28px', maxWidth: '1240px', margin: '0 auto', width: '100%' }}>
              <QuizView 
                documents={activeSpaceDocs} 
                activeSpace={activeSpace}
                user={user}
              />
            </div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
