import React, { useState } from 'react';
import { 
  Plus, 
  Folder, 
  Trash2, 
  Pencil, 
  Search, 
  FileText, 
  ArrowRight,
  X,
  BookOpen
} from 'lucide-react';
import { createSpace, updateSpace, deleteSpace, type StudySpaceItem, type UserProfile } from '../services/api';

interface CollectionsDashboardProps {
  spaces: StudySpaceItem[];
  onSelectSpace: (space: StudySpaceItem) => void;
  onRefreshSpaces: () => void;
  user: UserProfile | null;
}

export const CollectionsDashboard: React.FC<CollectionsDashboardProps> = ({
  spaces,
  onSelectSpace,
  onRefreshSpaces,
  user,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  
  // Modal Tạo không gian mới
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Modal Chỉnh sửa / Đổi tên
  const [editingSpace, setEditingSpace] = useState<StudySpaceItem | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);

  const filteredSpaces = spaces.filter(sp => 
    sp.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (sp.description && sp.description.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const handleCreateSpace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;

    setIsSubmitting(true);
    try {
      const created = await createSpace(newTitle.trim(), newDesc.trim(), user?.id);
      onRefreshSpaces();
      setIsCreateModalOpen(false);
      setNewTitle('');
      setNewDesc('');
      onSelectSpace(created);
    } catch (err) {
      console.error('Lỗi khi tạo không gian:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleOpenEdit = (space: StudySpaceItem, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSpace(space);
    setEditTitle(space.title);
    setEditDesc(space.description || '');
  };

  const handleUpdateSpace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingSpace || !editTitle.trim()) return;

    setIsUpdating(true);
    try {
      await updateSpace(editingSpace.id, {
        title: editTitle.trim(),
        description: editDesc.trim(),
      });
      onRefreshSpaces();
      setEditingSpace(null);
    } catch (err) {
      console.error('Lỗi khi cập nhật:', err);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDeleteSpace = async (spaceId: string, title: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm(`Bạn có chắc muốn xóa không gian "${title}"?`)) {
      return;
    }
    try {
      await deleteSpace(spaceId);
      onRefreshSpaces();
    } catch (err) {
      console.error('Lỗi khi xóa:', err);
    }
  };

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '32px 24px 60px',
      width: '100%',
    }}>
      {/* Header Section */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        marginBottom: '28px',
        borderBottom: '1px solid var(--border-subtle)',
        paddingBottom: '20px'
      }}>
        <div>
          <h1 style={{
            fontSize: '22px',
            fontWeight: 700,
            color: 'var(--text-primary)',
            letterSpacing: '-0.02em',
            marginBottom: '4px'
          }}>
            Không Gian Học Tập
          </h1>
          <p style={{
            fontSize: '14px',
            color: 'var(--text-secondary)'
          }}>
            Tổ chức tài liệu theo môn học để hỏi đáp chuyên sâu và luyện tập trắc nghiệm.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Search Box */}
          <div style={{ position: 'relative', width: '240px' }}>
            <Search 
              size={15} 
              style={{ 
                position: 'absolute', 
                left: '12px', 
                top: '50%', 
                transform: 'translateY(-50%)', 
                color: 'var(--text-muted)' 
              }} 
            />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Tìm kiếm không gian..."
              className="input-field"
              style={{
                width: '100%',
                paddingLeft: '34px',
                paddingTop: '8px',
                paddingBottom: '8px',
                fontSize: '13px'
              }}
            />
          </div>

          <button
            onClick={() => setIsCreateModalOpen(true)}
            className="btn-primary"
            style={{ fontSize: '13.5px', padding: '8px 16px' }}
          >
            <Plus size={16} />
            <span>Tạo không gian</span>
          </button>
        </div>
      </div>

      {/* Grid of Study Spaces */}
      {filteredSpaces.length === 0 ? (
        <div 
          className="surface-card" 
          style={{ 
            padding: '48px 24px', 
            textAlign: 'center',
            maxWidth: '520px',
            margin: '40px auto'
          }}
        >
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: 'var(--radius-md)',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            color: 'var(--text-muted)'
          }}>
            <BookOpen size={22} />
          </div>
          <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
            {searchTerm ? 'Không tìm thấy không gian phù hợp' : 'Chưa có không gian học tập nào'}
          </h3>
          <p style={{ fontSize: '13.5px', color: 'var(--text-muted)', marginBottom: '20px', lineHeight: '1.5' }}>
            {searchTerm 
              ? 'Thử tìm với từ khóa khác hoặc xóa bộ lọc tìm kiếm.'
              : 'Hãy tạo không gian học tập đầu tiên để tải lên tài liệu giáo trình và bắt đầu nghiên cứu.'}
          </p>
          {!searchTerm && (
            <button
              onClick={() => setIsCreateModalOpen(true)}
              className="btn-primary"
              style={{ fontSize: '13.5px' }}
            >
              <Plus size={15} />
              <span>Tạo không gian ngay</span>
            </button>
          )}
        </div>
      ) : (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
          gap: '16px',
        }}>
          {filteredSpaces.map((space) => {
            const docCount = space.num_documents ?? (space.document_ids ? space.document_ids.length : 0);
            return (
              <div
                key={space.id}
                onClick={() => onSelectSpace(space)}
                className="surface-card surface-card-hover"
                style={{
                  padding: '20px',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  minHeight: '160px',
                }}
              >
                <div>
                  {/* Card Header: Icon + Title + Action buttons */}
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '10px', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
                      <div style={{
                        width: '34px',
                        height: '34px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'var(--primary)',
                        flexShrink: 0
                      }}>
                        <Folder size={17} />
                      </div>
                      <h3 style={{
                        fontSize: '15px',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {space.title}
                      </h3>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }} onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={(e) => handleOpenEdit(space, e)}
                        className="btn-ghost"
                        style={{ padding: '5px' }}
                        title="Đổi tên"
                      >
                        <Pencil size={13} />
                      </button>
                      <button
                        onClick={(e) => handleDeleteSpace(space.id, space.title, e)}
                        className="btn-ghost"
                        style={{ padding: '5px', color: 'var(--text-muted)' }}
                        title="Xóa không gian"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </div>

                  {/* Description */}
                  <p style={{
                    fontSize: '13px',
                    color: 'var(--text-secondary)',
                    lineHeight: '1.5',
                    marginBottom: '16px',
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    minHeight: '38px'
                  }}>
                    {space.description || 'Không có mô tả chi tiết.'}
                  </p>
                </div>

                {/* Card Footer */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  borderTop: '1px solid var(--border-subtle)',
                  paddingTop: '12px',
                  fontSize: '12px',
                  color: 'var(--text-muted)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <FileText size={13} />
                    <span>{docCount} tài liệu</span>
                  </div>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--primary)', fontWeight: 500 }}>
                    Mở không gian <ArrowRight size={13} />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Modal Tạo Không Gian Mới */}
      {isCreateModalOpen && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.65)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          padding: '20px'
        }}>
          <div className="surface-card" style={{
            width: '100%',
            maxWidth: '440px',
            padding: '24px',
            boxShadow: 'var(--shadow-lg)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '17px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Tạo Không Gian Học Tập Mới
              </h3>
              <button onClick={() => setIsCreateModalOpen(false)} className="btn-ghost" style={{ padding: '4px' }}>
                <X size={16} />
              </button>
            </div>

            <form onSubmit={handleCreateSpace} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Tên không gian / môn học *
                </label>
                <input
                  type="text"
                  required
                  autoFocus
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="Ví dụ: Cấu trúc dữ liệu & Giải thuật"
                  className="input-field"
                  style={{ width: '100%' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Mô tả ngắn
                </label>
                <textarea
                  rows={3}
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  placeholder="Tóm tắt nội dung môn học hoặc mục tiêu ôn thi..."
                  className="input-field"
                  style={{ width: '100%', resize: 'vertical' }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '8px' }}>
                <button
                  type="button"
                  onClick={() => setIsCreateModalOpen(false)}
                  className="btn-secondary"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting || !newTitle.trim()}
                  className="btn-primary"
                >
                  {isSubmitting ? 'Đang tạo...' : 'Tạo không gian'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal Sửa Tên Không Gian */}
      {editingSpace && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.65)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          padding: '20px'
        }}>
          <div className="surface-card" style={{
            width: '100%',
            maxWidth: '440px',
            padding: '24px',
            boxShadow: 'var(--shadow-lg)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '17px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Chỉnh Sửa Không Gian
              </h3>
              <button onClick={() => setEditingSpace(null)} className="btn-ghost" style={{ padding: '4px' }}>
                <X size={16} />
              </button>
            </div>

            <form onSubmit={handleUpdateSpace} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Tên không gian *
                </label>
                <input
                  type="text"
                  required
                  autoFocus
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  className="input-field"
                  style={{ width: '100%' }}
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  Mô tả ngắn
                </label>
                <textarea
                  rows={3}
                  value={editDesc}
                  onChange={(e) => setEditDesc(e.target.value)}
                  className="input-field"
                  style={{ width: '100%', resize: 'vertical' }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '8px' }}>
                <button
                  type="button"
                  onClick={() => setEditingSpace(null)}
                  className="btn-secondary"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  disabled={isUpdating || !editTitle.trim()}
                  className="btn-primary"
                >
                  {isUpdating ? 'Đang lưu...' : 'Lưu thay đổi'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
export default CollectionsDashboard;
