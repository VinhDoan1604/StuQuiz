import React from 'react';
import { 
  Loader2, 
  CheckCircle2, 
  AlertCircle, 
  X
} from 'lucide-react';

export interface UploadProgressData {
  isUploading: boolean;
  fileName: string;
  fileSize?: string;
  progress?: number;
  stageText?: string;
  currentStep?: number;
  elapsedSec?: number;
  error?: string | null;
  numChunks?: number | null;
}

interface UploadProgressBarProps {
  data: UploadProgressData;
  onDismiss?: () => void;
  compact?: boolean;
}

export const UploadProgressBar: React.FC<UploadProgressBarProps> = ({
  data,
  onDismiss,
}) => {
  const {
    isUploading,
    fileName,
    fileSize,
    stageText,
    error,
    numChunks
  } = data;

  const isCompleted = !isUploading && !error;
  const isError = Boolean(error);

  return (
    <div
      style={{
        background: isError 
          ? 'var(--danger-subtle)' 
          : isCompleted 
            ? 'var(--success-subtle)' 
            : 'var(--bg-elevated)',
        border: `1px solid ${
          isError 
            ? 'var(--danger-border)' 
            : isCompleted 
              ? 'var(--success-border)' 
              : 'var(--border-subtle)'
        }`,
        borderRadius: 'var(--radius-md)',
        padding: '10px 12px',
        margin: '8px 0',
        transition: 'all 0.2s ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0, flex: 1 }}>
          {isUploading && (
            <Loader2 size={15} className="animate-spin" color="var(--primary)" style={{ flexShrink: 0 }} />
          )}
          {isCompleted && (
            <CheckCircle2 size={15} color="#10b981" style={{ flexShrink: 0 }} />
          )}
          {isError && (
            <AlertCircle size={15} color="#ef4444" style={{ flexShrink: 0 }} />
          )}

          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{
              fontSize: '12.5px',
              fontWeight: 600,
              color: 'var(--text-primary)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}>
              {fileName}
            </div>

            <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginTop: '1px' }}>
              {isError 
                ? (error || 'Tải lên thất bại')
                : isCompleted 
                  ? `Đã nạp xong • ${numChunks ? `${numChunks} đoạn trích` : 'Sẵn sàng tra cứu'}`
                  : (stageText || 'Đang xử lý tài liệu...')}
              {fileSize && <span> • {fileSize}</span>}
            </div>
          </div>
        </div>

        {onDismiss && !isUploading && (
          <button onClick={onDismiss} className="btn-ghost" style={{ padding: '2px' }}>
            <X size={13} />
          </button>
        )}
      </div>
    </div>
  );
};
export default UploadProgressBar;
