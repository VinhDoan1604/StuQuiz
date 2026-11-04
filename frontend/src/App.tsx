import { useCallback, useEffect, useState } from 'react';
import { Header } from './components/Header';
import { ChatView } from './components/ChatView';
import { QuizView } from './components/QuizView';
import { AuthScreen } from './components/AuthScreen';
import { CollectionsDashboard } from './components/CollectionsDashboard';
import {
  attachDocumentToSpace,
  clearSession,
  detachDocumentFromSpace,
  fetchDocuments,
  fetchSpaces,
  getStoredUser,
  saveSession,
  type DocumentItem,
  type StudySpaceItem,
  type UserProfile,
} from './services/api';

export function App() {
  const [user, setUser] = useState<UserProfile | null>(getStoredUser);

  // Study Spaces & Documents
  const [spaces, setSpaces] = useState<StudySpaceItem[]>([]);
  const [activeSpaceId, setActiveSpaceId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);

  // Tab bên trong một không gian
  const [activeTab, setActiveTab] = useState<'chat' | 'quiz'>('chat');

  // Luôn suy ra từ danh sách mới nhất => không bao giờ lệch dữ liệu sau khi refresh
  const activeSpace = spaces.find((s) => s.id === activeSpaceId) ?? null;

  const loadDocuments = useCallback(async () => {
    setIsLoadingDocs(true);
    try {
      setDocuments(await fetchDocuments());
    } catch (e) {
      console.warn('Không tải được tài liệu:', e);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  const loadSpaces = useCallback(async () => {
    try {
      setSpaces(await fetchSpaces());
    } catch (e) {
      console.warn('Không tải được không gian học tập:', e);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadDocuments(), loadSpaces()]);
  }, [loadDocuments, loadSpaces]);

  const handleLogout = useCallback(() => {
    clearSession();
    setUser(null);
    setActiveSpaceId(null);
    setSpaces([]);
    setDocuments([]);
  }, []);

  useEffect(() => {
    if (user) refreshAll();
  }, [user, refreshAll]);

  useEffect(() => {
    window.addEventListener('auth:expired', handleLogout);
    return () => window.removeEventListener('auth:expired', handleLogout);
  }, [handleLogout]);

  const handleLoginSuccess = (userProfile: UserProfile, authToken: string) => {
    saveSession(userProfile, authToken);
    setUser(userProfile);
  };

  const handleAttachDoc = async (docId: string) => {
    if (!activeSpace) return;
    try {
      await attachDocumentToSpace(activeSpace.id, docId);
      await refreshAll();
    } catch (e) {
      console.error('Không gắn được tài liệu:', e);
    }
  };

  const handleDetachDoc = async (docId: string) => {
    if (!activeSpace) return;
    try {
      await detachDocumentFromSpace(activeSpace.id, docId);
      await refreshAll();
    } catch (e) {
      console.error('Không gỡ được tài liệu:', e);
    }
  };

  if (!user) {
    return <AuthScreen onSuccess={handleLoginSuccess} />;
  }

  // Chỉ các tài liệu thuộc không gian đang mở (không gian rỗng => danh sách rỗng)
  const activeSpaceDocs = activeSpace
    ? documents.filter((d) => activeSpace.document_ids.includes(d.id))
    : [];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-base)' }}>
      <Header
        activeSpace={activeSpace}
        onBackToDashboard={() => setActiveSpaceId(null)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        user={user}
        onLogout={handleLogout}
      />

      {!activeSpace ? (
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CollectionsDashboard
            spaces={spaces}
            onSelectSpace={(space) => {
              setSpaces((prev) => (prev.some((s) => s.id === space.id) ? prev : [space, ...prev]));
              setActiveSpaceId(space.id);
              setActiveTab('chat');
            }}
            onRefreshSpaces={loadSpaces}
          />
        </main>
      ) : (
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {activeTab === 'chat' && (
            <ChatView
              key={activeSpace.id}
              documents={activeSpaceDocs}
              allDocuments={documents}
              isLoadingDocs={isLoadingDocs}
              onRefreshDocs={refreshAll}
              activeSpace={activeSpace}
              onAttachDoc={handleAttachDoc}
              onDetachDoc={handleDetachDoc}
            />
          )}

          {activeTab === 'quiz' && (
            <div style={{ padding: '24px 28px', maxWidth: '1240px', margin: '0 auto', width: '100%' }}>
              <QuizView key={activeSpace.id} documents={activeSpaceDocs} activeSpace={activeSpace} />
            </div>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
