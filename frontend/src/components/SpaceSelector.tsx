import React, { useState } from 'react';
import { FolderKanban, ChevronDown, Plus, Check } from 'lucide-react';
import { createSpace, type StudySpaceItem } from '../services/api';

interface SpaceSelectorProps {
  spaces: StudySpaceItem[];
  activeSpace: StudySpaceItem | null;
  onSelectSpace: (space: StudySpaceItem) => void;
  onRefreshSpaces: () => void;
  userId?: string;
}

export const SpaceSelector: React.FC<SpaceSelectorProps> = ({
  spaces,
  activeSpace,
  onSelectSpace,
  onRefreshSpaces,
  userId,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;
    setIsSubmitting(true);
    try {
      const created = await createSpace(newTitle.trim(), newDesc.trim(), userId);
      onRefreshSpaces();
      onSelectSpace(created);
      setShowCreateModal(false);
      setNewTitle('');
      setNewDesc('');
      setIsOpen(false);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid var(--border-amber)',
          padding: '6px 14px',
          borderRadius: 'var(--radius-md)',
          cursor: 'pointer',
          color: '#f8fafc',
          fontSize: '13.5px',
          fontWeight: 600,
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(245, 158, 11, 0.08)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(255, 255, 255, 0.04)')}
      >
        <FolderKanban size={17} color="var(--accent-amber)" />
        <span style={{ maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {activeSpace ? activeSpace.title : 'Chọn Bộ Học Tập'}
        </span>
        <span className="badge badge-amber" style={{ fontSize: '10.5px', padding: '1px 6px' }}>
          {activeSpace?.document_ids?.length || 0} docs
        </span>
        <ChevronDown size={14} color="var(--text-dim)" />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div 
          className="glass-panel"
          style={{
            position: 'absolute',
            top: 'calc(100% + 8px)',
            left: 0,
            width: '320px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-amber)',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-amber)',
            padding: '8px',
            zIndex: 60,
            display: 'flex',
            flexDirection: 'column',
            gap: '4px'
          }}
        >
          <div style={{ padding: '6px 10px', fontSize: '11.5px', color: 'var(--text-dim)', fontWeight: 700, letterSpacing: '0.04em' }}>
            DANH SÁCH BỘ HỌC TẬP (STUDY SPACES)
          </div>

          <div style={{ maxHeight: '220px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '3px' }}>
            {spaces.map((sp) => {
              const isSelected = activeSpace?.id === sp.id;
              return (
                <div
                  key={sp.id}
                  onClick={() => {
                    onSelectSpace(sp);
                    setIsOpen(false);
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 10px',
                    borderRadius: 'var(--radius-sm)',
                    background: isSelected ? 'rgba(245, 158, 11, 0.12)' : 'transparent',
                    border: isSelected ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid transparent',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: isSelected ? '#fbbf24' : '#f1f5f9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {sp.title}
                    </div>
                    {sp.description && (
                      <div style={{ fontSize: '11px', color: 'var(--text-dim)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {sp.description}
                      </div>
                    )}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexShrink: 0 }}>
                    <span style={{ fontSize: '11px', color: 'var(--accent-amber)', fontWeight: 600 }}>
                      {sp.document_ids?.length || 0} tệp
                    </span>
                    {isSelected && <Check size={14} color="#fbbf24" />}
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '6px', marginTop: '4px' }}>
            <button
              onClick={() => {
                setShowCreateModal(true);
                setIsOpen(false);
              }}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm)',
                background: 'rgba(245, 158, 11, 0.08)',
                border: '1px dashed var(--border-amber)',
                color: 'var(--accent-amber)',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              <Plus size={15} />
              <span>Tạo Bộ Học Tập Mới</span>
            </button>
          </div>
        </div>
      )}

      {/* Create New Space Modal */}
      {showCreateModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(5, 8, 14, 0.75)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 110,
          padding: '20px'
        }}>
          <div 
            className="glass-panel" 
            style={{
              width: '100%',
              maxWidth: '420px',
              background: 'var(--bg-card)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-amber)',
              boxShadow: 'var(--shadow-amber)',
              padding: '24px',
            }}
          >
            <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#f8fafc', marginBottom: '8px' }}>
              Tạo Bộ Học Tập Mới (Study Space)
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '18px' }}>
              Mỗi bộ sẽ quản lý tài liệu và ngữ cảnh hội thoại độc lập.
            </p>

            <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 600 }}>
                  Tên Bộ Học Tập (Ví dụ: Hệ Điều Hành, Giải Thuật...)
                </label>
                <input
                  type="text"
                  required
                  placeholder="Nhập tên môn / chủ đề..."
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    padding: '10px 14px',
                    color: '#f8fafc',
                    fontSize: '14px',
                    outline: 'none',
                    fontFamily: 'inherit'
                  }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 600 }}>
                  Mô tả ngắn
                </label>
                <textarea
                  rows={2}
                  placeholder="Ghi chú mục tiêu học tập..."
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    padding: '10px 14px',
                    color: '#f8fafc',
                    fontSize: '14px',
                    outline: 'none',
                    fontFamily: 'inherit',
                    resize: 'none'
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '10px', marginTop: '6px' }}>
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="btn-secondary"
                  style={{ flex: 1, justifyContent: 'center' }}
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting || !newTitle.trim()}
                  className="btn-primary"
                  style={{ flex: 1, justifyContent: 'center' }}
                >
                  {isSubmitting ? 'Đang tạo...' : 'Tạo Không Gian'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
