import { useState, useRef, useEffect } from 'react';
import { 
  UploadCloud, 
  FileText, 
  Layers, 
  RefreshCw, 
  FolderPlus, 
  Calendar, 
  FileCheck, 
  Loader2,
  Trash2,
  AlertTriangle 
} from 'lucide-react';
import { uploadDocument, deleteDocument, type DocumentItem } from '../services/api';
import { UploadProgressBar, type UploadProgressData } from './UploadProgressBar';

interface DocumentManagerProps {
  documents: DocumentItem[];
  isLoading: boolean;
  onRefresh: () => void;
  userId?: string;
}

export const DocumentManager = ({
  documents,
  isLoading,
  onRefresh,
  userId,
}: DocumentManagerProps) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressData, setUploadProgressData] = useState<UploadProgressData | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Tự động làm mới danh sách tài liệu nếu có tài liệu đang lập chỉ mục nền
  useEffect(() => {
    const hasProcessing = documents.some(d => d.status === 'processing');
    if (!hasProcessing) return;

    const interval = setInterval(() => {
      onRefresh();
    }, 3000);
    return () => clearInterval(interval);
  }, [documents, onRefresh]);

  const formatBytes = (bytes: number) => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    const fSize = formatBytes(file.size);
    const startTime = Date.now();

    setUploadProgressData({
      isUploading: true,
      fileName: file.name,
      fileSize: fSize,
      progress: 15,
      stageText: '📤 Đang tải tệp lên máy chủ & phân tích cấu trúc...',
      currentStep: 1,
      elapsedSec: 0,
      error: null,
      numChunks: null,
    });

    const timer = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      setUploadProgressData(prev => prev && prev.isUploading ? {
        ...prev,
        progress: Math.min(25 + elapsed * 10, 92),
        elapsedSec: elapsed
      } : prev);
    }, 150);

    try {
      const res = await uploadDocument(file, userId);
      clearInterval(timer);
      const totalElapsed = (Date.now() - startTime) / 1000;
      const isProcessing = res.status === 'processing';

      setUploadProgressData({
        isUploading: false,
        fileName: file.name,
        fileSize: fSize,
        progress: 100,
        stageText: isProcessing
          ? `⚡ Tải lên thành công (${totalElapsed.toFixed(1)}s)! Hệ thống đang lập chỉ mục vector (BGE-M3) trong nền...`
          : `✅ Nạp và xử lý tệp thành công! Đã tạo ${res.num_chunks} vector chunks vào Qdrant.`,
        currentStep: 4,
        elapsedSec: totalElapsed,
        numChunks: res.num_chunks,
        error: null,
      });

      onRefresh();
      setTimeout(() => {
        setUploadProgressData(curr => (curr && !curr.isUploading ? null : curr));
      }, 5000);
    } catch (err: any) {
      clearInterval(timer);
      const totalElapsed = (Date.now() - startTime) / 1000;
      setUploadProgressData({
        isUploading: false,
        fileName: file.name,
        fileSize: fSize,
        progress: 100,
        stageText: '❌ Thất bại khi nạp tệp.',
        currentStep: 1,
        elapsedSec: totalElapsed,
        error: err.message || 'Lỗi khi upload tệp tin.',
        numChunks: null,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', padding: '24px 20px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Info */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '20px', fontWeight: 800, color: '#f8fafc' }}>
            Kho Tài Liệu Học Tập (Knowledge Base)
          </h2>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Tự động trích xuất nội dung, phân đoạn ngữ nghĩa (Hybrid Chunking) và tạo embedding đa ngữ BGE-M3.
          </p>
        </div>
        <button 
          onClick={onRefresh} 
          disabled={isLoading}
          className="btn-secondary"
          style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <RefreshCw size={14} className={isLoading ? 'pulse-badge' : ''} />
          <span>Làm mới</span>
        </button>
      </div>

      {/* Upload Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={triggerFileSelect}
        style={{
          border: dragOver ? '2px dashed #38bdf8' : '2px dashed rgba(255, 255, 255, 0.15)',
          background: dragOver ? 'rgba(56, 189, 248, 0.08)' : 'rgba(19, 27, 46, 0.5)',
          borderRadius: 'var(--radius-lg)',
          padding: '40px 24px',
          textAlign: 'center',
          cursor: isUploading ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        <input 
          ref={fileInputRef}
          type="file" 
          accept=".pdf,.docx,.pptx,.txt" 
          style={{ display: 'none' }}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFileUpload(e.target.files[0]);
            }
          }}
        />

        <div style={{
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          background: 'rgba(56, 189, 248, 0.12)',
          color: '#38bdf8',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 16px',
        }}>
          <UploadCloud size={28} />
        </div>

        <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#f1f5f9' }}>
          {isUploading ? 'Đang phân tích & lập chỉ mục vector...' : 'Kéo thả tệp hoặc bấm để chọn'}
        </h3>
        <p style={{ fontSize: '13px', color: 'var(--text-dim)', marginTop: '6px' }}>
          Hỗ trợ định dạng PDF, DOCX, PPTX, TXT • Giới hạn tối đa 50MB
        </p>

        {isUploading && (
          <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: 'var(--accent-amber)', fontSize: '13px' }}>
            <Loader2 size={16} className="animate-spin" />
            <span style={{ fontWeight: 600 }}>{uploadProgressData?.stageText || 'Đang xử lý tệp...'}</span>
          </div>
        )}
      </div>

      {/* Upload Progress & Feedback Card */}
      {uploadProgressData && (
        <UploadProgressBar 
          data={uploadProgressData} 
          onDismiss={() => setUploadProgressData(null)}
          compact={false}
        />
      )}

      {/* Document Grid / List */}
      <div>
        <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#e2e8f0', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileCheck size={16} color="#38bdf8" />
          Tài Liệu Đã Nạp ({documents.length})
        </h3>

        {documents.length === 0 ? (
          <div className="glass-card" style={{ padding: '36px', textAlign: 'center', color: 'var(--text-dim)' }}>
            <FolderPlus size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
            <p style={{ fontSize: '14px', fontWeight: 500 }}>Chưa có tài liệu nào trong hệ thống</p>
            <p style={{ fontSize: '12.5px', marginTop: '4px' }}>
              Hãy tải lên giáo trình, slide bài giảng hoặc đề thi để bắt đầu trải nghiệm RAG.
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '14px' }}>
            {documents.map((doc) => (
              <div key={doc.id} className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                  <div style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '8px',
                    background: 'rgba(99, 102, 241, 0.15)',
                    color: '#818cf8',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    <FileText size={18} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#f1f5f9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {doc.filename}
                    </h4>
                    <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                      ID: {doc.id.substring(0, 8)}...
                    </span>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border-subtle)', paddingTop: '10px', fontSize: '12px' }}>
                  {doc.status === 'processing' ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#38bdf8' }}>
                      <Loader2 size={13} className="animate-spin" />
                      <span style={{ fontWeight: 500 }}>Đang lập chỉ mục...</span>
                    </div>
                  ) : doc.status === 'failed' ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#ef4444' }} title={doc.error_message || 'Lỗi lập chỉ mục'}>
                      <AlertTriangle size={13} />
                      <span style={{ fontWeight: 500 }}>Lỗi lập chỉ mục</span>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#10b981' }}>
                      <Layers size={13} />
                      <span>{doc.num_chunks} vector chunks</span>
                    </div>
                  )}

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {doc.uploaded_at && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-dim)', fontSize: '11px' }}>
                        <Calendar size={12} />
                        <span>{new Date(doc.uploaded_at).toLocaleDateString()}</span>
                      </div>
                    )}
                    <button
                      onClick={async () => {
                        if (confirm(`Bạn có chắc muốn xóa tài liệu "${doc.filename}"?`)) {
                          try {
                            await deleteDocument(doc.id, userId);
                            onRefresh();
                          } catch (e: any) {
                            alert(e.message || 'Lỗi khi xóa tài liệu');
                          }
                        }
                      }}
                      className="btn-ghost"
                      style={{ padding: '3px 4px', color: 'var(--text-dim)', borderRadius: '4px' }}
                      title="Xóa tài liệu"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
