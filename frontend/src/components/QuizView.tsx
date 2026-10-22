import { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, ArrowRight, RotateCcw, Clock, Loader2 } from 'lucide-react';
import {
  generateQuiz,
  answerQuiz,
  fetchSpaceQuizHistory,
  type QuizQuestion,
  type QuizAnswerResult,
  type DocumentItem,
  type StudySpaceItem,
  type UserProfile,
} from '../services/api';

interface QuizViewProps {
  documents: DocumentItem[];
  activeSpace?: StudySpaceItem | null;
  user?: UserProfile | null; // giữ lại cho tương thích; danh tính do server lấy từ token
}

export const QuizView = ({ documents, activeSpace }: QuizViewProps) => {
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedChoice, setSelectedChoice] = useState<string | null>(null);
  const [result, setResult] = useState<QuizAnswerResult | null>(null); // kết quả do SERVER chấm
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [score, setScore] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const [numQuestions, setNumQuestions] = useState(5);
  const [topicPrompt, setTopicPrompt] = useState('');
  const [quizHistory, setQuizHistory] = useState<any[]>([]);

  const loadQuizHistory = async () => {
    if (!activeSpace) return;
    try {
      setQuizHistory(await fetchSpaceQuizHistory(activeSpace.id));
    } catch (e) {
      console.warn('Failed to load quiz history:', e);
    }
  };

  useEffect(() => {
    if (activeSpace) loadQuizHistory();
  }, [activeSpace]);

  const resetRun = () => {
    setCurrentIndex(0);
    setSelectedChoice(null);
    setResult(null);
    setScore(0);
    setIsCompleted(false);
  };

  const handleStartQuiz = async () => {
    if (documents.length === 0) {
      alert('Không gian này chưa có tài liệu nào. Hãy tải tài liệu lên trước khi tạo đề trắc nghiệm.');
      return;
    }
    setIsGenerating(true);
    try {
      const generated = await generateQuiz(
        documents.map((d) => d.id),
        numQuestions,
        true,
        '',
        topicPrompt,
        activeSpace?.id
      );
      if (generated && generated.length > 0) {
        setQuestions(generated);
        resetRun();
      } else {
        alert('Không tìm thấy nội dung phù hợp để sinh câu hỏi. Vui lòng kiểm tra lại tài liệu.');
      }
    } catch (err: any) {
      alert(`Lỗi sinh quiz: ${err.message || 'Hệ thống đang bận'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSelectChoice = async (choice: string) => {
    if (result || isSubmitting) return;
    const q = questions[currentIndex];
    setSelectedChoice(choice);
    setIsSubmitting(true);
    try {
      const res = await answerQuiz(q.attempt_id, choice);
      setResult(res);
      if (res.is_correct) setScore((s) => s + 1);
      if (activeSpace) loadQuizHistory();
    } catch (e: any) {
      setSelectedChoice(null);
      alert(`Không ghi nhận được câu trả lời: ${e.message || 'lỗi mạng'}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleNext = () => {
    if (currentIndex + 1 < questions.length) {
      setCurrentIndex((i) => i + 1);
      setSelectedChoice(null);
      setResult(null);
    } else {
      setIsCompleted(true);
      if (activeSpace) loadQuizHistory();
    }
  };

  const handleReset = () => {
    setQuestions([]);
    resetRun();
  };

  // Screen 1: kết quả
  if (isCompleted) {
    const percent = Math.round((score / questions.length) * 100);
    const good = percent >= 70;
    return (
      <div style={{ maxWidth: '640px', margin: '40px auto', padding: '0 20px' }}>
        <div className="surface-card" style={{ padding: '36px 28px', textAlign: 'center' }}>
          <div
            style={{
              width: '56px', height: '56px', borderRadius: 'var(--radius-full)',
              background: good ? 'var(--success-subtle)' : 'var(--accent-amber-subtle)',
              border: `1px solid ${good ? 'var(--success-border)' : 'var(--accent-amber-border)'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 16px', color: good ? '#10b981' : '#f59e0b',
            }}
          >
            <CheckCircle2 size={28} />
          </div>
          <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
            Hoàn Thành Bài Ôn Tập!
          </h2>
          <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '24px' }}>
            Bạn đã trả lời đúng <strong>{score}</strong> / <strong>{questions.length}</strong> câu ({percent}%).
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px' }}>
            <button onClick={handleReset} className="btn-secondary">
              <RotateCcw size={14} />
              <span>Đổi cấu hình</span>
            </button>
            <button onClick={handleStartQuiz} disabled={isGenerating} className="btn-primary">
              <span>{isGenerating ? 'Đang soạn câu hỏi...' : 'Luyện lại với cấu hình này'}</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Screen 2: làm bài
  if (questions.length > 0) {
    const q = questions[currentIndex];
    const progressPercent = ((currentIndex + 1) / questions.length) * 100;

    return (
      <div style={{ maxWidth: '720px', margin: '30px auto', padding: '0 20px' }}>
        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '13px' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              Câu hỏi {currentIndex + 1} / {questions.length}
            </span>
            <span style={{ color: 'var(--text-secondary)' }}>Đúng: {score} câu</span>
          </div>
          <div style={{ width: '100%', height: '4px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
            <div style={{ width: `${progressPercent}%`, height: '100%', background: 'var(--primary)', transition: 'width 0.3s ease' }} />
          </div>
        </div>

        <div className="surface-card" style={{ padding: '24px', marginBottom: '16px' }}>
          <div style={{ marginBottom: '10px', display: 'flex', gap: '6px' }}>
            {q.topic && <span className="badge badge-subtle">{q.topic}</span>}
            <span className="badge badge-subtle">{q.difficulty}</span>
          </div>

          <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.6, marginBottom: '20px' }}>
            {q.question}
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {q.choices.map((choice, cIdx) => {
              const isSelected = selectedChoice === choice;
              const isCorrect = result !== null && choice === result.correct_choice;

              let bg = 'var(--bg-base)';
              let border = 'var(--border-subtle)';
              let color = 'var(--text-primary)';
              if (result) {
                if (isCorrect) {
                  bg = 'var(--success-subtle)'; border = 'var(--success-border)'; color = '#34d399';
                } else if (isSelected) {
                  bg = 'var(--danger-subtle)'; border = 'var(--danger-border)'; color = '#f87171';
                }
              } else if (isSelected) {
                bg = 'var(--primary-subtle)'; border = 'var(--primary-border)';
              }

              return (
                <button
                  key={cIdx}
                  onClick={() => handleSelectChoice(choice)}
                  disabled={result !== null || isSubmitting}
                  style={{
                    padding: '12px 14px', borderRadius: 'var(--radius-md)', background: bg,
                    border: `1px solid ${border}`, color, textAlign: 'left', fontSize: '14px',
                    cursor: result || isSubmitting ? 'default' : 'pointer',
                    display: 'flex', alignItems: 'center', gap: '12px', transition: 'all 0.15s ease',
                  }}
                >
                  <span
                    style={{
                      width: '24px', height: '24px', borderRadius: 'var(--radius-xs)',
                      background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center',
                      justifyContent: 'center', fontSize: '12px', fontWeight: 600, flexShrink: 0,
                    }}
                  >
                    {String.fromCharCode(65 + cIdx)}
                  </span>
                  <span style={{ flex: 1 }}>{choice}</span>
                  {isSubmitting && isSelected && <Loader2 size={16} className="animate-spin" />}
                  {result && isCorrect && <CheckCircle2 size={16} color="#10b981" />}
                  {result && isSelected && !isCorrect && <XCircle size={16} color="#ef4444" />}
                </button>
              );
            })}
          </div>

          {result && (
            <>
              <div
                style={{
                  marginTop: '20px', padding: '14px 16px', borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
                  fontSize: '13.5px', lineHeight: 1.6, color: 'var(--text-secondary)',
                }}
              >
                <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '4px' }}>
                  {result.is_correct ? 'Chính xác!' : 'Chưa đúng.'} Giải thích:
                </strong>
                {result.explanation || 'Không có giải thích cho câu này.'}
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
                <button onClick={handleNext} className="btn-primary">
                  <span>{currentIndex + 1 < questions.length ? 'Câu tiếp theo' : 'Xem kết quả'}</span>
                  <ArrowRight size={15} />
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  // Screen 3: cấu hình
  return (
    <div style={{ maxWidth: '800px', margin: '20px auto', padding: '0 20px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '24px', alignItems: 'flex-start' }}>
        <div className="surface-card" style={{ padding: '24px' }}>
          <h2 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
            Luyện Tập Trắc Nghiệm
          </h2>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginBottom: '20px', lineHeight: 1.5 }}>
            Hệ thống dựa trên tài liệu trong không gian này để biên soạn câu hỏi kèm giải thích. Việc chấm điểm
            do máy chủ thực hiện và được dùng để lên lịch ôn tập.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Số lượng câu hỏi:
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {[3, 5, 10].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setNumQuestions(n)}
                    style={{
                      flex: 1, padding: '8px', borderRadius: 'var(--radius-sm)',
                      border: numQuestions === n ? '1px solid var(--primary)' : '1px solid var(--border-subtle)',
                      background: numQuestions === n ? 'var(--primary-subtle)' : 'var(--bg-base)',
                      color: numQuestions === n ? 'var(--primary)' : 'var(--text-secondary)',
                      fontWeight: 600, fontSize: '13px', cursor: 'pointer', transition: 'all 0.15s ease',
                    }}
                  >
                    {n} câu
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Chủ đề định hướng (tùy chọn):
              </label>
              <input
                type="text"
                value={topicPrompt}
                onChange={(e) => setTopicPrompt(e.target.value)}
                placeholder="Ví dụ: Chương 2, Thuật toán sắp xếp, v.v."
                className="input-field"
                style={{ width: '100%', fontSize: '13px' }}
              />
            </div>

            <button
              onClick={handleStartQuiz}
              disabled={isGenerating || documents.length === 0}
              className="btn-primary"
              style={{ marginTop: '8px', padding: '10px 18px', width: '100%' }}
            >
              {isGenerating ? 'Đang soạn câu hỏi từ tài liệu...' : 'Bắt đầu làm bài'}
            </button>
          </div>
        </div>

        <div className="surface-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '14px', fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)' }}>
            <Clock size={15} color="var(--primary)" />
            <span>Lịch sử luyện tập gần đây</span>
          </div>

          {quizHistory.length === 0 ? (
            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Chưa có câu nào được trả lời trong không gian này.
            </p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {quizHistory.slice(0, 5).map((h) => (
                <div
                  key={h.id}
                  style={{
                    padding: '8px 10px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-base)',
                    border: '1px solid var(--border-subtle)', fontSize: '12px',
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px',
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {h.topic || 'Tổng hợp'}
                  </span>
                  <span style={{ fontWeight: 600, color: h.is_correct ? '#34d399' : '#f87171', flexShrink: 0 }}>
                    {h.is_correct ? 'Đúng' : 'Sai'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
export default QuizView;
