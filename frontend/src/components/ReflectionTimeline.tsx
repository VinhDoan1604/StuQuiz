import { useState } from 'react';
import { CheckCircle2, RotateCcw, ChevronDown, ChevronUp, Search, ShieldCheck, ShieldAlert, FileCheck } from 'lucide-react';
import type { ReflectionTrace, AgentStepEvent } from '../services/api';

interface ReflectionTimelineProps {
  trace?: ReflectionTrace | null;
  liveEvents?: AgentStepEvent[];
  isStreaming?: boolean;
}

const pct = (v: number | null | undefined) => (typeof v === 'number' ? `${Math.round(v * 100)}%` : '—');

type Status = { kind: 'running' | 'verified' | 'forced' | 'unverified' | 'na'; text: string };

export const ReflectionTimeline = ({ trace, liveEvents = [], isStreaming = false }: ReflectionTimelineProps) => {
  const [expanded, setExpanded] = useState(false);

  if (!trace && liveEvents.length === 0 && !isStreaming) return null;

  const gradeAnswerEvt = [...liveEvents].reverse().find((e) => e.node === 'grade_answer');
  const gradeContextEvt = [...liveEvents].reverse().find((e) => e.node === 'grade_context');

  // Ưu tiên trace cuối cùng từ server; khi đang stream thì dùng sự kiện trực tiếp
  const answerGrade = trace ? trace.answer_grade : gradeAnswerEvt?.grade ?? null;
  const answerFaithful = trace ? trace.answer_faithful : gradeAnswerEvt?.faithful ?? null;
  const answerVerified = trace ? trace.answer_verified : gradeAnswerEvt ? gradeAnswerEvt.graded !== false : null;
  const answerForced = trace ? trace.answer_forced : gradeAnswerEvt?.forced ?? null;
  const contextGrade = trace ? trace.context_grade : gradeContextEvt?.grade ?? null;
  const rewritten =
    trace?.rewritten_queries ?? liveEvents.find((e) => e.node === 'retrieval')?.rewritten_queries ?? [];
  const numLoops =
    trace?.num_loops ?? liveEvents.filter((e) => e.node === 'rewrite_query' || e.node === 'retry_generation').length;

  let status: Status;
  if (isStreaming) status = { kind: 'running', text: 'Đang phân tích...' };
  else if (answerVerified === null || answerVerified === undefined)
    status = { kind: 'na', text: 'Không áp dụng kiểm chứng cho chế độ này' };
  else if (answerVerified === false)
    status = { kind: 'unverified', text: 'Chưa kiểm chứng được (bộ chấm gặp lỗi)' };
  else if (answerForced)
    status = { kind: 'forced', text: `Chấp nhận sau ${numLoops} lần thử — chưa đạt ngưỡng trung thực` };
  else if (answerFaithful)
    status = { kind: 'verified', text: `Đã kiểm chứng · ${pct(answerGrade)}` };
  else status = { kind: 'unverified', text: 'Chưa đạt ngưỡng trung thực' };

  const badgeClass =
    status.kind === 'verified' ? 'badge-emerald' : status.kind === 'running' ? 'badge-blue'
    : status.kind === 'na' ? 'badge-subtle' : 'badge-amber';
  const Icon = status.kind === 'verified' ? CheckCircle2 : status.kind === 'na' || status.kind === 'running' ? ShieldCheck : ShieldAlert;

  return (
    <div style={{ marginTop: '10px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', background: 'var(--bg-subtle)', overflow: 'hidden' }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', cursor: 'pointer', userSelect: 'none', fontSize: '12.5px', color: 'var(--text-secondary)' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <ShieldCheck size={14} color="var(--primary)" />
          <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>Quá trình kiểm chứng</span>
          <span className={`badge ${badgeClass}`} style={{ fontSize: '11px', padding: '1px 6px' }}>
            <Icon size={10} /> {status.text}
          </span>
          {!isStreaming && numLoops > 0 && (
            <span className="badge badge-subtle" style={{ fontSize: '11px', padding: '1px 6px' }}>
              <RotateCcw size={10} /> {numLoops} vòng thử lại
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-muted)' }}>
          <span>{expanded ? 'Ẩn' : 'Chi tiết'}</span>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      {expanded && (
        <div style={{ padding: '10px 14px', borderTop: '1px solid var(--border-subtle)', fontSize: '12.5px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(0,0,0,0.15)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <FileCheck size={13} color="#60a5fa" />
            <span>Độ liên quan &amp; đầy đủ của ngữ cảnh:</span>
            <strong style={{ color: 'var(--text-primary)' }}>{pct(contextGrade)}</strong>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <FileCheck size={13} color="#10b981" />
            <span>Độ trung thực của câu trả lời với tài liệu:</span>
            <strong style={{ color: 'var(--text-primary)' }}>{answerVerified === false ? 'không chấm được' : pct(answerGrade)}</strong>
          </div>

          {rewritten.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                <Search size={13} color="var(--primary)" />
                <span>Truy vấn đã được viết lại để tìm thêm dữ liệu:</span>
              </div>
              <ul style={{ paddingLeft: '20px', margin: '2px 0 0', color: 'var(--text-muted)' }}>
                {rewritten.map((q, i) => (
                  <li key={i} style={{ marginBottom: '2px' }}>"{q}"</li>
                ))}
              </ul>
            </div>
          )}

          <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
            Điểm do mô hình AI chấm tự động dựa trên đoạn tài liệu được truy xuất; đây là chỉ báo tham khảo, không phải bảo đảm tuyệt đối.
          </div>
        </div>
      )}
    </div>
  );
};
export default ReflectionTimeline;
