import { BookOpen, FileText, X } from 'lucide-react';
import type { Citation } from '../services/api';

interface CitationViewerProps {
  citations: Citation[];
  activeCitationIndex: number | null;
  onSelectCitation: (index: number) => void;
  onClose?: () => void;
}

export const CitationViewer = ({
  citations,
  activeCitationIndex,
  onSelectCitation,
  onClose,
}: CitationViewerProps) => {
  if (!citations || citations.length === 0) {
    return (
      <div className="surface-card" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
        <BookOpen size={30} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
        <p style={{ fontSize: '13px', fontWeight: 500 }}>Chưa có nguồn trích dẫn nào</p>
      </div>
    );
  }

  const activeIdx = activeCitationIndex !== null && citations[activeCitationIndex] 
    ? activeCitationIndex 
    : 0;
  const activeCitation = citations[activeIdx];
  const fileName = activeCitation.file ? activeCitation.file.split(/[\\/]/).pop() : 'Tài liệu nguồn';

  return (
    <div className="surface-card" style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100%', 
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden' 
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-surface)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={15} color="var(--primary)" />
          <h3 style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Đoạn Trích Dẫn ({citations.length})
          </h3>
        </div>
        {onClose && (
          <button 
            onClick={onClose}
            className="btn-ghost"
            style={{ padding: '4px' }}
            title="Đóng bảng trích dẫn"
          >
            <X size={15} />
          </button>
        )}
      </div>

      {/* Citations Tab List */}
      {citations.length > 1 && (
        <div style={{
          display: 'flex',
          gap: '6px',
          padding: '8px 12px',
          overflowX: 'auto',
          borderBottom: '1px solid var(--border-subtle)',
          background: 'var(--bg-base)'
        }}>
          {citations.map((c, idx) => {
            const isSelected = activeIdx === idx;
            const name = c.file ? c.file.split(/[\\/]/).pop() : `Trích dẫn ${idx + 1}`;
            return (
              <button
                key={idx}
                onClick={() => onSelectCitation(idx)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  padding: '4px 10px',
                  borderRadius: 'var(--radius-sm)',
                  border: isSelected ? '1px solid var(--primary)' : '1px solid var(--border-subtle)',
                  background: isSelected ? 'var(--primary-subtle)' : 'transparent',
                  color: isSelected ? 'var(--primary)' : 'var(--text-secondary)',
                  fontSize: '12px',
                  fontWeight: 500,
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.15s ease'
                }}
              >
                <span>[{idx + 1}] {name}</span>
                {c.page && <span style={{ opacity: 0.7 }}>p.{c.page}</span>}
              </button>
            );
          })}
        </div>
      )}

      {/* Citation Detail Box */}
      <div style={{ flex: 1, padding: '16px', overflowY: 'auto' }}>
        <div style={{
          padding: '12px 14px',
          background: 'var(--bg-base)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
          marginBottom: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              {fileName}
            </span>
            {activeCitation.page && (
              <span className="badge badge-subtle" style={{ fontSize: '11px' }}>
                Trang {activeCitation.page}
              </span>
            )}
          </div>

          {activeCitation.section && (
            <p style={{
              fontSize: '13px',
              color: 'var(--text-secondary)',
              lineHeight: '1.6',
              whiteSpace: 'pre-wrap',
              borderLeft: '2px solid var(--primary)',
              paddingLeft: '10px'
            }}>
              "{activeCitation.section}"
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
export default CitationViewer;
