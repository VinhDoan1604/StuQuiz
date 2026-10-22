import { BookOpen, FileText, X } from 'lucide-react';
import type { Citation } from '../services/api';

interface CitationViewerProps {
  citations: Citation[];
  activeCitationIndex: number | null;
  onSelectCitation: (index: number) => void;
  onClose?: () => void;
}

const baseName = (path?: string) => (path ? path.split(/[\\/]/).pop() : undefined);

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

  const activeIdx =
    activeCitationIndex !== null && citations[activeCitationIndex] ? activeCitationIndex : 0;
  const active = citations[activeIdx];
  const label = (c: Citation, i: number) => c.index ?? i + 1; // số [n] khớp với số trong câu trả lời

  return (
    <div
      className="surface-card"
      style={{ display: 'flex', flexDirection: 'column', height: '100%', borderRadius: 'var(--radius-lg)', overflow: 'hidden' }}
    >
      <div
        style={{
          padding: '12px 16px', borderBottom: '1px solid var(--border-subtle)', display: 'flex',
          alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-surface)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={15} color="var(--primary)" />
          <h3 style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
            Đoạn Trích Dẫn ({citations.length})
          </h3>
        </div>
        {onClose && (
          <button onClick={onClose} className="btn-ghost" style={{ padding: '4px' }} title="Đóng bảng trích dẫn">
            <X size={15} />
          </button>
        )}
      </div>

      {citations.length > 1 && (
        <div
          style={{
            display: 'flex', gap: '6px', padding: '8px 12px', overflowX: 'auto',
            borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-base)',
          }}
        >
          {citations.map((c, idx) => {
            const selected = activeIdx === idx;
            return (
              <button
                key={idx}
                onClick={() => onSelectCitation(idx)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '5px', padding: '4px 10px',
                  borderRadius: 'var(--radius-sm)',
                  border: selected ? '1px solid var(--primary)' : '1px solid var(--border-subtle)',
                  background: selected ? 'var(--primary-subtle)' : 'transparent',
                  color: selected ? 'var(--primary)' : 'var(--text-secondary)',
                  fontSize: '12px', fontWeight: 500, cursor: 'pointer', whiteSpace: 'nowrap',
                }}
              >
                <span>[{label(c, idx)}] {baseName(c.file) ?? 'Nguồn'}</span>
                {c.page != null && <span style={{ opacity: 0.7 }}>tr.{c.page}</span>}
              </button>
            );
          })}
        </div>
      )}

      <div style={{ flex: 1, padding: '16px', overflowY: 'auto' }}>
        <div
          style={{
            padding: '12px 14px', background: 'var(--bg-base)', borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              [{label(active, activeIdx)}] {baseName(active.file) ?? 'Tài liệu nguồn'}
            </span>
            {active.page != null && (
              <span className="badge badge-subtle" style={{ fontSize: '11px' }}>Trang {active.page}</span>
            )}
          </div>

          {active.section && (
            <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginBottom: '10px' }}>
              Mục: {active.section}
            </div>
          )}

          {active.snippet ? (
            <p
              style={{
                fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.65, whiteSpace: 'pre-wrap',
                borderLeft: '2px solid var(--primary)', paddingLeft: '10px', margin: 0,
              }}
            >
              {active.snippet}
            </p>
          ) : (
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', margin: 0 }}>
              Không có đoạn trích gốc cho nguồn này (hội thoại được lưu trước khi hỗ trợ hiển thị đoạn trích).
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
export default CitationViewer;
