import React from 'react';
import { 
  BookOpen, 
  ArrowLeft, 
  MessageSquare, 
  HelpCircle, 
  LogOut, 
  Folder
} from 'lucide-react';
import type { StudySpaceItem, UserProfile } from '../services/api';

interface HeaderProps {
  activeSpace: StudySpaceItem | null;
  onBackToDashboard: () => void;
  activeTab: 'chat' | 'quiz';
  setActiveTab: (tab: 'chat' | 'quiz') => void;
  user: UserProfile | null;
  onLogout: () => void;
}

export const Header: React.FC<HeaderProps> = ({ 
  activeSpace, 
  onBackToDashboard,
  activeTab, 
  setActiveTab, 
  user,
  onLogout,
}) => {
  const getInitials = (name?: string | null, email?: string) => {
    if (name && name.trim()) {
      const parts = name.trim().split(' ');
      if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
      return name.substring(0, 2).toUpperCase();
    }
    if (email) return email.substring(0, 2).toUpperCase();
    return 'HV';
  };

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      left: 0,
      right: 0,
      width: '100%',
      height: '56px',
      zIndex: 50,
      background: 'var(--bg-surface)',
      borderBottom: '1px solid var(--border-subtle)',
      padding: '0 20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '16px',
    }}>
      {/* Left: Brand & Navigation */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px', minWidth: 0 }}>
        {/* Brand */}
        <div 
          onClick={onBackToDashboard}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            cursor: 'pointer',
            userSelect: 'none',
            flexShrink: 0
          }}
          title="Về trang chủ không gian học tập"
        >
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: 'var(--radius-sm)',
            background: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
          }}>
            <BookOpen size={18} strokeWidth={2.2} />
          </div>
          <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            StudyRAG
          </span>
        </div>

        {/* Breadcrumb if inside an active space */}
        {activeSpace && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>/</span>

            <button
              onClick={onBackToDashboard}
              className="btn-ghost"
              style={{
                padding: '4px 8px',
                fontSize: '13px',
                color: 'var(--text-secondary)'
              }}
              title="Danh sách không gian học tập"
            >
              <ArrowLeft size={14} />
              <span>Không gian</span>
            </button>

            <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>/</span>

            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              minWidth: 0,
              padding: '3px 8px',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-subtle)',
            }}>
              <Folder size={14} color="var(--primary)" />
              <span style={{ 
                fontSize: '13px', 
                fontWeight: 600, 
                color: 'var(--text-primary)',
                maxWidth: '200px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}>
                {activeSpace.title}
              </span>
              <span className="badge badge-subtle" style={{ fontSize: '11px', padding: '1px 6px' }}>
                {activeSpace.num_documents ?? (activeSpace.document_ids?.length || 0)} tệp
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Center: Clean Segmented Tab Switcher (Only when in an active space) */}
      {activeSpace && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          background: 'var(--bg-base)',
          padding: '3px',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
        }}>
          <button
            onClick={() => setActiveTab('chat')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '5px 14px',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              background: activeTab === 'chat' ? 'var(--bg-elevated)' : 'transparent',
              color: activeTab === 'chat' ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: activeTab === 'chat' ? 'var(--shadow-sm)' : 'none',
              fontWeight: activeTab === 'chat' ? 600 : 500,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <MessageSquare size={14} />
            <span>Hỏi Đáp Tài Liệu</span>
          </button>

          <button
            onClick={() => setActiveTab('quiz')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '5px 14px',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              background: activeTab === 'quiz' ? 'var(--bg-elevated)' : 'transparent',
              color: activeTab === 'quiz' ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: activeTab === 'quiz' ? 'var(--shadow-sm)' : 'none',
              fontWeight: activeTab === 'quiz' ? 600 : 500,
              fontSize: '13px',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <HelpCircle size={14} />
            <span>Ôn Tập & Quiz</span>
          </button>
        </div>
      )}

      {/* Right: User Profile & Logout */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '28px',
            height: '28px',
            borderRadius: 'var(--radius-full)',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '11px',
            fontWeight: 700,
            color: 'var(--text-secondary)'
          }}>
            {getInitials(user?.full_name, user?.email)}
          </div>
          <span style={{
            fontSize: '13px',
            fontWeight: 500,
            color: 'var(--text-secondary)',
            maxWidth: '140px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap'
          }}>
            {user?.full_name || user?.email || 'Học viên'}
          </span>
        </div>

        <button
          onClick={onLogout}
          className="btn-ghost"
          style={{ padding: '6px 10px', fontSize: '12.5px' }}
          title="Đăng xuất khỏi hệ thống"
        >
          <LogOut size={14} />
          <span>Đăng xuất</span>
        </button>
      </div>
    </header>
  );
};
export default Header;
