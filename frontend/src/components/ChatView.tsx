import { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  BookOpen, 
  Plus, 
  PanelLeft, 
  Loader2,
  FileText
} from 'lucide-react';
import { 
  streamChatMessage, 
  type ChatResponse, 
  type AgentStepEvent, 
  type Citation,
  type DocumentItem, 
  type StudySpaceItem, 
  type UserProfile 
} from '../services/api';
import { ReflectionTimeline } from './ReflectionTimeline';
import { CitationViewer } from './CitationViewer';
import { DocumentSidebar } from './DocumentSidebar';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export interface ChatViewProps {
  documents?: DocumentItem[];
  allDocuments?: DocumentItem[];
  isLoadingDocs?: boolean;
  onRefreshDocs?: () => void;
  activeSpace?: StudySpaceItem | null;
  onAttachDoc?: (docId: string) => Promise<void>;
  onDetachDoc?: (docId: string) => Promise<void>;
  user?: UserProfile | null;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  reflection?: any;
  liveEvents?: AgentStepEvent[];
}

// [n] -> link nội bộ #cite-n (bỏ qua [n](...) đã là link markdown)
const linkifyCitations = (text: string) => text.replace(/\[(\d{1,2})\](?!\()/g, '[[$1]](#cite-$1)');

export const ChatView: React.FC<ChatViewProps> = ({
  documents = [],
  allDocuments = [],
  isLoadingDocs = false,
  onRefreshDocs = () => {},
  activeSpace,
  onAttachDoc,
  onDetachDoc,
  user,
}) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveEvents, setLiveEvents] = useState<AgentStepEvent[]>([]);
  const [sessionId, setSessionId] = useState<string>(() => 'session_' + Math.random().toString(36).substring(2, 9));
  
  // Citation Viewer Drawer
  const [selectedCitations, setSelectedCitations] = useState<Citation[]>([]);
  const [activeCitationIdx, setActiveCitationIdx] = useState<number | null>(null);
  const [showCitationDrawer, setShowCitationDrawer] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, liveEvents]);

  const handleSend = async (customMessage?: string) => {
    const textToSend = customMessage || input;
    if (!textToSend.trim() || isStreaming) return;

    const userMsg: Message = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: textToSend.trim(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);
    setLiveEvents([]);

    const assistantMsgId = 'msg_ast_' + Date.now();
    let currentLiveEvents: AgentStepEvent[] = [];

    const activeDocIds = activeSpace?.document_ids && activeSpace.document_ids.length > 0
      ? activeSpace.document_ids
      : documents.map(d => d.id);

    await streamChatMessage(
      sessionId,
      textToSend.trim(),
      (event: AgentStepEvent) => {
        currentLiveEvents = [...currentLiveEvents, event];
        setLiveEvents([...currentLiveEvents]);
      },
      (doneRes: ChatResponse) => {
        setIsStreaming(false);
        const assistantMsg: Message = {
          id: assistantMsgId,
          role: 'assistant',
          content: doneRes.answer,
          citations: doneRes.citations,
          reflection: doneRes.reflection,
          liveEvents: currentLiveEvents,
        };
        setMessages(prev => [...prev, assistantMsg]);
        setLiveEvents([]);

        if (doneRes.citations && doneRes.citations.length > 0) {
          setSelectedCitations(doneRes.citations);
          setActiveCitationIdx(0);
        }
      },
      (err: string) => {
        setIsStreaming(false);
        const errorMsg: Message = {
          id: assistantMsgId,
          role: 'assistant',
          content: `Có lỗi xảy ra trong quá trình phản hồi: ${err}`,
          liveEvents: currentLiveEvents,
        };
        setMessages(prev => [...prev, errorMsg]);
        setLiveEvents([]);
      },
      activeDocIds,
      activeSpace?.id,
      user?.id
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSelectChatSession = (session: any) => {
    setSessionId(session.session_id);
    if (session.turns && session.turns.length > 0) {
      const restored: Message[] = session.turns.map((t: any) => ({
        id: t.id || 'msg_' + Math.random(),
        role: t.role,
        content: t.content,
        citations: t.reflection_trace?.citations || [],
        reflection: t.reflection_trace,
      }));
      setMessages(restored);
    } else {
      setMessages([]);
    }
    setLiveEvents([]);
  };

  const handleNewSession = () => {
    setSessionId('session_' + Math.random().toString(36).substring(2, 9));
    setMessages([]);
    setLiveEvents([]);
    setSelectedCitations([]);
    setShowCitationDrawer(false);
  };

  const openCitationDetail = (citations: Citation[], index: number) => {
    setSelectedCitations(citations);
    setActiveCitationIdx(index);
    setShowCitationDrawer(true);
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 56px)', overflow: 'hidden' }}>
      {/* Document & Sessions Sidebar */}
      <DocumentSidebar 
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(false)}
        documents={documents}
        allDocuments={allDocuments}
        isLoading={isLoadingDocs}
        onRefresh={onRefreshDocs}
        activeSpace={activeSpace}
        onAttachDoc={onAttachDoc}
        onDetachDoc={onDetachDoc}
        onSelectChatSession={handleSelectChatSession}
        activeSessionId={sessionId}
        onNewChat={handleNewSession}
      />

      {/* Main Conversation Center */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0, position: 'relative' }}>
        {/* Top Control Bar */}
        <div style={{
          height: '46px',
          borderBottom: '1px solid var(--border-subtle)',
          padding: '0 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--bg-surface)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {!isSidebarOpen && (
              <button
                onClick={() => setIsSidebarOpen(true)}
                className="btn-ghost"
                style={{ padding: '6px' }}
                title="Mở thanh tài liệu"
              >
                <PanelLeft size={16} />
              </button>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--text-secondary)' }}>
              <span>Đang tra cứu:</span>
              <strong style={{ color: 'var(--text-primary)' }}>
                {documents.length} tài liệu
              </strong>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {selectedCitations.length > 0 && !showCitationDrawer && (
              <button 
                onClick={() => setShowCitationDrawer(true)}
                className="btn-secondary"
                style={{ padding: '4px 10px', fontSize: '12px' }}
              >
                <BookOpen size={13} />
                <span>Nguồn trích dẫn ({selectedCitations.length})</span>
              </button>
            )}

            <button 
              onClick={handleNewSession}
              className="btn-ghost"
              style={{ padding: '4px 10px', fontSize: '12.5px' }}
              title="Mở cuộc trò chuyện mới"
            >
              <Plus size={14} />
              <span>Đoạn chat mới</span>
            </button>
          </div>
        </div>

        {/* Message Thread */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '24px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px'
        }}>
          <div style={{ maxWidth: '780px', width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Empty State */}
            {messages.length === 0 && !isStreaming && (
              <div style={{ textAlign: 'center', padding: '60px 20px 40px' }}>
                <div style={{
                  width: '44px',
                  height: '44px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 16px',
                  color: 'var(--primary)'
                }}>
                  <BookOpen size={22} />
                </div>

                <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
                  {activeSpace ? activeSpace.title : 'Trợ lý học tập StudyRAG'}
                </h2>
                <p style={{ fontSize: '14px', color: 'var(--text-secondary)', maxWidth: '520px', margin: '0 auto 24px', lineHeight: '1.6' }}>
                  {documents.length > 0 
                    ? `Đặt câu hỏi về ${documents.length} tài liệu đã tải lên trong không gian này để được giải thích kèm trích dẫn đối soát.`
                    : 'Không gian này chưa có tài liệu nào. Hãy tải tệp lên ở thanh bên trái để bắt đầu hỏi đáp.'
                  }
                </p>

                {documents.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxWidth: '540px', margin: '0 auto', textAlign: 'left' }}>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Gợi ý câu hỏi:
                    </div>
                    {[
                      'Tóm tắt các nội dung cốt lõi và định nghĩa quan trọng nhất trong tài liệu?',
                      'Liệt kê các khái niệm dễ nhầm lẫn và cách phân biệt?',
                      'Đặt 3 câu hỏi thực hành dựa trên nội dung bài giảng?'
                    ].map((prompt, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSend(prompt)}
                        className="surface-card surface-card-hover"
                        style={{
                          padding: '10px 14px',
                          textAlign: 'left',
                          fontSize: '13px',
                          color: 'var(--text-secondary)',
                          cursor: 'pointer',
                          borderRadius: 'var(--radius-md)',
                        }}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Messages */}
            {messages.map((msg) => (
              <div 
                key={msg.id} 
                style={{ 
                  display: 'flex', 
                  flexDirection: 'column',
                  alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  width: '100%'
                }}
              >
                {/* Role Label */}
                <div style={{ 
                  fontSize: '12px', 
                  fontWeight: 600, 
                  color: 'var(--text-muted)', 
                  marginBottom: '4px',
                  paddingLeft: msg.role === 'assistant' ? '2px' : '0',
                  paddingRight: msg.role === 'user' ? '2px' : '0',
                }}>
                  {msg.role === 'user' ? 'Bạn' : 'StudyRAG'}
                </div>

                {/* Bubble */}
                <div style={{
                  maxWidth: msg.role === 'user' ? '82%' : '100%',
                  background: msg.role === 'user' ? 'var(--primary)' : 'var(--bg-card)',
                  color: msg.role === 'user' ? '#ffffff' : 'var(--text-primary)',
                  border: msg.role === 'user' ? 'none' : '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-lg)',
                  padding: msg.role === 'user' ? '10px 16px' : '16px 20px',
                  fontSize: '14.5px',
                  lineHeight: '1.65',
                  boxShadow: 'var(--shadow-sm)',
                }}>
                  {msg.role === 'assistant' ? (
                    <div className="markdown-body">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          a: ({ href, children }) => {
                            const m = href?.match(/^#cite-(\d+)$/);
                            if (m) {
                              const n = Number(m[1]);
                              const cites = msg.citations ?? [];
                              const idx = cites.some((c) => c.index != null) ? cites.findIndex((c) => c.index === n) : n - 1;
                              return idx >= 0 && cites[idx] ? (
                                <button className="cite-chip" onClick={() => openCitationDetail(cites, idx)} title="Xem đoạn trích gốc">{n}</button>
                              ) : (
                                <span>[{n}]</span>
                              );
                            }
                            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
                          },
                        }}
                      >
                        {linkifyCitations(msg.content)}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  )}
                </div>

                {/* Inline Citations */}
                {msg.citations && msg.citations.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap', marginTop: '6px' }}>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Tham chiếu:</span>
                    {msg.citations.map((c, cIdx) => (
                      <button
                        key={cIdx}
                        onClick={() => openCitationDetail(msg.citations!, cIdx)}
                        style={{
                          background: 'var(--bg-subtle)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: 'var(--radius-sm)',
                          color: 'var(--text-secondary)',
                          fontSize: '11.5px',
                          padding: '2px 8px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          transition: 'all 0.15s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = 'var(--primary)';
                          e.currentTarget.style.color = 'var(--primary)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = 'var(--border-subtle)';
                          e.currentTarget.style.color = 'var(--text-secondary)';
                        }}
                      >
                        <FileText size={11} />
                        <span>[{c.index ?? cIdx + 1}] {c.file ? c.file.split(/[\\/]/).pop() : 'Tài liệu'}</span>
                        {c.page && <span>(tr.{c.page})</span>}
                      </button>
                    ))}
                  </div>
                )}

                {/* Reflection Timeline */}
                {(msg.reflection || (msg.liveEvents && msg.liveEvents.length > 0)) && (
                  <div style={{ width: '100%' }}>
                    <ReflectionTimeline 
                      trace={msg.reflection} 
                      liveEvents={msg.liveEvents} 
                      isStreaming={false} 
                    />
                  </div>
                )}
              </div>
            ))}

            {/* Streaming Indicator */}
            {isStreaming && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '13px', padding: '8px 0' }}>
                <Loader2 size={15} className="animate-spin" color="var(--primary)" />
                <span>Đang tra cứu tài liệu và đối soát câu trả lời...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Bar */}
        <div style={{
          padding: '12px 20px 18px',
          background: 'var(--bg-surface)',
          borderTop: '1px solid var(--border-subtle)'
        }}>
          <div style={{ maxWidth: '780px', margin: '0 auto', position: 'relative' }}>
            <div style={{
              display: 'flex',
              alignItems: 'flex-end',
              gap: '8px',
              background: 'var(--bg-base)',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-lg)',
              padding: '8px 12px',
              transition: 'border-color 0.18s ease'
            }}>
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Hỏi bất kỳ điều gì về tài liệu... (Enter để gửi, Shift+Enter xuống dòng)"
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  color: 'var(--text-primary)',
                  fontSize: '14px',
                  lineHeight: '1.5',
                  resize: 'none',
                  maxHeight: '160px',
                  fontFamily: 'inherit',
                  padding: '4px 0'
                }}
              />

              <button
                onClick={() => handleSend()}
                disabled={isStreaming || !input.trim()}
                className="btn-primary"
                style={{
                  width: '34px',
                  height: '34px',
                  padding: 0,
                  borderRadius: 'var(--radius-md)',
                  flexShrink: 0
                }}
                title="Gửi câu hỏi"
              >
                <Send size={15} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right Drawer: Citations Detail */}
      {showCitationDrawer && (
        <aside style={{
          width: '320px',
          borderLeft: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)',
          height: '100%',
          flexShrink: 0
        }}>
          <CitationViewer
            citations={selectedCitations}
            activeCitationIndex={activeCitationIdx}
            onSelectCitation={(idx) => setActiveCitationIdx(idx)}
            onClose={() => setShowCitationDrawer(false)}
          />
        </aside>
      )}
    </div>
  );
};
export default ChatView;
