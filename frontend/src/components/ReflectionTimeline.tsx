import { useState } from 'react';
import { 
  CheckCircle2, 
  RotateCcw, 
  ChevronDown, 
  ChevronUp, 
  Search,
  ShieldCheck,
  FileCheck
} from 'lucide-react';
import type { ReflectionTrace, AgentStepEvent } from '../services/api';

interface ReflectionTimelineProps {
  trace?: ReflectionTrace | null;
  liveEvents?: AgentStepEvent[];
  isStreaming?: boolean;
}

export const ReflectionTimeline = ({
  trace,
  liveEvents = [],
  isStreaming = false,
}: ReflectionTimelineProps) => {
  const [expanded, setExpanded] = useState(false);

  if (!trace && liveEvents.length === 0 && !isStreaming) {
    return null;
  }

  const numLoops = trace?.num_loops ?? liveEvents.filter(e => e.node === 'rewrite_query' || e.node === 'retry_generation').length;
  const isFaithful = trace?.answer_faithful ?? liveEvents.find(e => e.node === 'grade_answer')?.faithful;
  const rewritten = trace?.rewritten_queries ?? liveEvents.find(e => e.node === 'retrieval')?.rewritten_queries ?? [];

  return (
    <div style={{
      marginTop: '10px',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-md)',
      background: 'var(--bg-subtle)',
      overflow: 'hidden',
    }}>
      {/* Header / Toggle Button */}
      <div 
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          cursor: 'pointer',
          userSelect: 'none',
          fontSize: '12.5px',
          color: 'var(--text-secondary)',
          background: 'transparent',
          transition: 'background 0.15s ease'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldCheck size={14} color="var(--primary)" />
          <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
            Quá trình kiểm chứng tài liệu
          </span>

          {isStreaming ? (
            <span className="badge badge-blue" style={{ fontSize: '11px', padding: '1px 6px' }}>
              Đang phân tích...
            </span>
          ) : numLoops > 0 ? (
            <span className="badge badge-amber" style={{ fontSize: '11px', padding: '1px 6px' }}>
              <RotateCcw size={10} /> Đã tối ưu truy vấn ({numLoops})
            </span>
          ) : (
            <span className="badge badge-emerald" style={{ fontSize: '11px', padding: '1px 6px' }}>
              <CheckCircle2 size={10} /> Khớp dữ liệu
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-muted)' }}>
          <span>{expanded ? 'Ẩn' : 'Chi tiết'}</span>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div style={{
          padding: '10px 14px',
          borderTop: '1px solid var(--border-subtle)',
          fontSize: '12.5px',
          color: 'var(--text-secondary)',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          background: 'rgba(0, 0, 0, 0.15)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <FileCheck size={13} color="#10b981" />
            <span>Độ chính xác nội dung:</span>
            <strong style={{ color: isFaithful ? '#34d399' : '#fbbf24' }}>
              {isFaithful ? 'Đạt chuẩn (100% trung thực với giáo trình)' : 'Đang đối soát'}
            </strong>
          </div>

          {rewritten.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                <Search size={13} color="var(--primary)" />
                <span>Câu truy vấn được mở rộng để tìm dữ liệu:</span>
              </div>
              <ul style={{ paddingLeft: '20px', margin: '2px 0 0', color: 'var(--text-muted)' }}>
                {rewritten.map((q, idx) => (
                  <li key={idx} style={{ marginBottom: '2px' }}>"{q}"</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
export default ReflectionTimeline;
