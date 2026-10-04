import React, { useState, useRef, useEffect } from 'react';
import { 
  FileText, 
  Upload, 
  Search, 
  ChevronLeft, 
  MessageSquare, 
  Plus, 
  Trash2, 
  FolderPlus,
  Loader2,
  AlertTriangle
} from 'lucide-react';
import { 
  uploadDocument, 
  fetchSpaceChatHistory,
  type DocumentItem, 
  type StudySpaceItem 
} from '../services/api';
import { UploadProgressBar, type UploadProgressData } from './UploadProgressBar';

interface DocumentSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  documents: DocumentItem[];
  allDocuments?: DocumentItem[];
  isLoading?: boolean;
  onRefresh: () => void;
  activeSpace?: StudySpaceItem | null;
  onAttachDoc?: (docId: string) => Promise<void>;
  onDetachDoc?: (docId: string) => Promise<void>;
  onSelectChatSession?: (session: any) => void;
  activeSessionId?: string;
  onNewChat?: () => void;
}

export const DocumentSidebar: React.FC<DocumentSidebarProps> = ({
  isOpen,
  onToggle,
  documents,
  allDocuments = [],
  isLoading: _isLoading,
  onRefresh,
  activeSpace,
  onAttachDoc,
  onDetachDoc,
  onSelectChatSession,
  activeSessionId,
  onNewChat,
}) => {
  const [activeTab, setActiveTab] = useState<'docs' | 'sessions'>('docs');
  const [searchTerm, setSearchTerm] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressData, setUploadProgressData] = useState<UploadProgressData | null>(null);
  const [showAttachPicker, setShowAttachPicker] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [spaceSessions, setSpaceSessions] = useState<any[]>([]);

  const loadSpaceSessions = async () => {
    if (!activeSpace) return;
    try {
      const history = await fetchSpaceChatHistory(activeSpace.id);
      setSpaceSessions(history);
    } catch (e) {
      console.warn('Failed to load space sessions:', e);
    }
  };

  useEffect(() => {
    if (activeSpace) {
      loadSpaceSessions();
    }
  }, [activeSpace, activeSessionId]);

  // Tự động làm mới danh sách khi có tài liệu đang xử lý nền
  useEffect(() => {
    const hasProcessing = documents.some(d => d.status === 'processing');
    if (!hasProcessing) return;

    const interval = setInterval(() => {
      onRefresh();
    }, 3000);
    return () => clearInterval(interval);
  }, [documents, onRefresh]);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setUploadProgressData({
      isUploading: true,
      fileName: file.name,
      fileSize: `${(file.size / 1024).toFixed(1)} KB`,
      stageText: 'Đang tải lên & phân tích nội dung...',
      error: null,
      numChunks: null,
    });

    try {
      const res = await uploadDocument(file);
      if (activeSpace && onAttachDoc && res.document_id) {
        await onAttachDoc(res.document_id);
      }
      onRefresh();
      setUploadProgressData({
        isUploading: false,
        fileName: file.name,
        numChunks: res.num_chunks,
        stageText: res.status === 'processing' ? '⚡ Tải lên thành công! Đang lập chỉ mục BGE-M3 trong nền...' : undefined,
        error: null
      });
      setTimeout(() => {
        setUploadProgressData(curr => (curr && !curr.isUploading ? null : curr));
      }, 4500);
    } catch (err: any) {
      setUploadProgressData({
        isUploading: false,
        fileName: file.name,
        error: err.message || 'Lỗi khi tải tệp lên'
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const filteredDocs = documents.filter(d => 
    d.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const availableToAttach = allDocuments.filter(
    ad => !documents.some(d => d.id === ad.id)
  );

  if (!isOpen) {
    return null;
  }

  return (
    <aside style={{
      width: '300px',
      background: 'var(--bg-surface)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      flexShrink: 0
    }}>
      {/* Sidebar Header */}
      <div style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '8px'
      }}>
        {/* Tab switch */}
        <div style={{
          display: 'flex',
          background: 'var(--bg-base)',
          padding: '2px',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-subtle)',
          flex: 1
        }}>
          <button
            onClick={() => setActiveTab('docs')}
            style={{
              flex: 1,
              padding: '4px 8px',
              borderRadius: 'var(--radius-xs)',
              border: 'none',
              background: activeTab === 'docs' ? 'var(--bg-elevated)' : 'transparent',
              color: activeTab === 'docs' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            Tài liệu ({documents.length})
          </button>
          <button
            onClick={() => setActiveTab('sessions')}
            style={{
              flex: 1,
              padding: '4px 8px',
              borderRadius: 'var(--radius-xs)',
              border: 'none',
              background: activeTab === 'sessions' ? 'var(--bg-elevated)' : 'transparent',
              color: activeTab === 'sessions' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            Lịch sử ({spaceSessions.length})
          </button>
        </div>

        <button 
          onClick={onToggle} 
          className="btn-ghost" 
          style={{ padding: '6px' }}
          title="Thu gọn thanh bên"
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      {/* Tab 1: Documents */}
      {activeTab === 'docs' && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Action Row: Upload button + Attach button */}
          <div style={{ padding: '10px 14px', display: 'flex', gap: '8px' }}>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept=".pdf,.docx,.txt,.md"
              onChange={handleFileChange}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="btn-primary"
              style={{ flex: 1, fontSize: '12.5px', padding: '7px 10px' }}
            >
              <Upload size={14} />
              <span>{isUploading ? 'Đang nạp...' : 'Tải tài liệu'}</span>
            </button>

            {availableToAttach.length > 0 && onAttachDoc && (
              <button
                onClick={() => setShowAttachPicker(!showAttachPicker)}
                className="btn-secondary"
                style={{ fontSize: '12px', padding: '7px 10px' }}
                title="Đính kèm từ thư viện đã có"
              >
                <FolderPlus size={14} />
              </button>
            )}
          </div>

          {/* Upload Status Card */}
          {uploadProgressData && (
            <div style={{ padding: '0 14px' }}>
              <UploadProgressBar 
                data={uploadProgressData} 
                onDismiss={() => setUploadProgressData(null)}
              />
            </div>
          )}

          {/* Attach from library modal/popover */}
          {showAttachPicker && availableToAttach.length > 0 && (
            <div style={{
              margin: '0 14px 10px',
              padding: '10px',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-md)',
              maxHeight: '180px',
              overflowY: 'auto'
            }}>
              <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Chọn tài liệu từ kho lưu trữ:
              </div>
              {availableToAttach.map(doc => (
                <div 
                  key={doc.id}
                  onClick={async () => {
                    if (onAttachDoc) {
                      await onAttachDoc(doc.id);
                      setShowAttachPicker(false);
                      onRefresh();
                    }
                  }}
                  style={{
                    padding: '6px 8px',
                    fontSize: '12px',
                    borderRadius: 'var(--radius-xs)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    color: 'var(--text-primary)',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-hover)')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {doc.filename}
                  </span>
                  <Plus size={13} color="var(--primary)" />
                </div>
              ))}
            </div>
          )}

          {/* Search bar */}
          <div style={{ padding: '0 14px 8px' }}>
            <div style={{ position: 'relative' }}>
              <Search size={13} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Lọc tài liệu..."
                className="input-field"
                style={{ width: '100%', paddingLeft: '28px', paddingTop: '6px', paddingBottom: '6px', fontSize: '12px' }}
              />
            </div>
          </div>

          {/* Document List */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 14px 14px' }}>
            {filteredDocs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px 10px', color: 'var(--text-muted)', fontSize: '12.5px' }}>
                {searchTerm ? 'Không tìm thấy tài liệu phù hợp' : 'Chưa có tài liệu nào trong không gian này. Hãy tải lên giáo trình để bắt đầu.'}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {filteredDocs.map((doc) => (
                  <div
                    key={doc.id}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 'var(--radius-md)',
                      background: 'transparent',
                      border: '1px solid transparent',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '8px',
                      transition: 'all 0.15s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'var(--bg-hover)';
                      e.currentTarget.style.borderColor = 'var(--border-subtle)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.borderColor = 'transparent';
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                      <FileText size={15} color="var(--primary)" style={{ flexShrink: 0 }} />
                      <div style={{ minWidth: 0 }}>
                        <div style={{
                          fontSize: '12.5px',
                          fontWeight: 500,
                          color: 'var(--text-primary)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {doc.filename}
                        </div>
                        <div style={{ fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px', marginTop: '2px' }}>
                          {doc.status === 'processing' ? (
                            <span style={{ color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <Loader2 size={11} className="animate-spin" />
                              Đang lập chỉ mục...
                            </span>
                          ) : doc.status === 'failed' ? (
                            <span style={{ color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '4px' }} title={doc.error_message || 'Lỗi lập chỉ mục'}>
                              <AlertTriangle size={11} />
                              Lỗi xử lý
                            </span>
                          ) : (
                            <span style={{ color: 'var(--text-muted)' }}>
                              {doc.num_chunks} đoạn trích
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {onDetachDoc && (
                      <button
                        onClick={() => onDetachDoc(doc.id)}
                        className="btn-ghost"
                        style={{ padding: '4px', color: 'var(--text-muted)' }}
                        title="Gỡ tài liệu khỏi không gian"
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Sessions */}
      {activeTab === 'sessions' && (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {onNewChat && (
            <div style={{ padding: '10px 14px' }}>
              <button
                onClick={onNewChat}
                className="btn-secondary"
                style={{ width: '100%', fontSize: '12.5px', padding: '7px 10px' }}
              >
                <Plus size={14} />
                <span>Đoạn chat mới</span>
              </button>
            </div>
          )}

          <div style={{ flex: 1, overflowY: 'auto', padding: '0 14px 14px' }}>
            {spaceSessions.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px 10px', color: 'var(--text-muted)', fontSize: '12.5px' }}>
                Chưa có lịch sử chat nào trong không gian này.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {spaceSessions.map((session, sIdx) => {
                  const isActive = activeSessionId === session.session_id;
                  const firstMsg = session.turns?.[0]?.content || 'Đoạn hội thoại';
                  return (
                    <div
                      key={sIdx}
                      onClick={() => onSelectChatSession && onSelectChatSession(session)}
                      style={{
                        padding: '8px 10px',
                        borderRadius: 'var(--radius-md)',
                        background: isActive ? 'var(--primary-subtle)' : 'transparent',
                        border: `1px solid ${isActive ? 'var(--primary-border)' : 'transparent'}`,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                      onMouseEnter={(e) => {
                        if (!isActive) e.currentTarget.style.background = 'var(--bg-hover)';
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) e.currentTarget.style.background = 'transparent';
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                        <MessageSquare size={13} color={isActive ? 'var(--primary)' : 'var(--text-muted)'} />
                        <span style={{
                          fontSize: '12.5px',
                          fontWeight: isActive ? 600 : 500,
                          color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {firstMsg}
                        </span>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', paddingLeft: '19px' }}>
                        {session.turns?.length || 0} tin nhắn
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </aside>
  );
};
export default DocumentSidebar;
