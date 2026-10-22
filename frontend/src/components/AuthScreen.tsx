import React, { useState } from 'react';
import { 
  BookOpen, 
  Loader2, 
  Zap,
  AlertCircle
} from 'lucide-react';
import { loginUser, registerUser, verifyOtp, googleAuth, demoLogin, type UserProfile } from '../services/api';

const GOOGLE_CLIENT_ID = (import.meta as any).env?.VITE_GOOGLE_CLIENT_ID || '';
const IS_DUMMY_CLIENT_ID = !GOOGLE_CLIENT_ID || GOOGLE_CLIENT_ID.includes('dcnqpf7d8mtr1rdt7n42j4ggph90fda6');

interface AuthScreenProps {
  onSuccess: (user: UserProfile, token: string) => void;
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ onSuccess }) => {
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [authStep, setAuthStep] = useState<'form' | 'otp'>('form');
  
  // Credentials
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  
  // OTP
  const [pendingEmail, setPendingEmail] = useState('');
  const [otpCode, setOtpCode] = useState('');
  const [devCode, setDevCode] = useState<string | null>(null);

  // States
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);

  // Quick Demo Access
  const handleQuickDemo = async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await demoLogin();
      if (res.user && res.token) onSuccess(res.user, res.token);
    } catch (err: any) {
    setError(err.message || 'Không thể vào chế độ dùng thử');
  } finally {
    setLoading(false);
  }
};

  // Google Login
  const handleGoogleLogin = () => {
    setError(null);
    if (IS_DUMMY_CLIENT_ID) {
      setError('Google OAuth chưa được cấu hình Client ID thật trên Google Cloud. Bạn có thể bấm "Vào trực tiếp (Dùng thử)" hoặc Đăng ký bằng Email bên dưới.');
      return;
    }

    setGoogleLoading(true);
    try {
      const redirectUri = window.location.origin;
      const oauthUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${encodeURIComponent(GOOGLE_CLIENT_ID)}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=token&scope=openid%20email%20profile&prompt=select_account`;
      const popup = window.open(oauthUrl, 'google_oauth2_popup', 'width=500,height=650');
      if (!popup) {
        setGoogleLoading(false);
        setError('Trình duyệt đã chặn cửa sổ đăng nhập. Vui lòng cho phép popup.');
        return;
      }

      const pollTimer = setInterval(async () => {
        try {
          if (!popup || popup.closed) {
            clearInterval(pollTimer);
            setGoogleLoading(false);
            return;
          }
          if (popup.location.href.includes(redirectUri)) {
            const hash = popup.location.hash;
            if (hash) {
              const params = new URLSearchParams(hash.replace('#', '?'));
              const accessToken = params.get('access_token');
              if (accessToken) {
                clearInterval(pollTimer);
                popup.close();
                const res = await googleAuth({ access_token: accessToken });
                if (res.user && res.token) {
                  onSuccess(res.user, res.token);
                }
              }
            }
          }
        } catch {
          // Cross-origin while on google
        }
      }, 500);
    } catch (e: any) {
      setGoogleLoading(false);
      setError('Lỗi mở phiên xác thực Google: ' + (e.message || String(e)));
    }
  };

  // Submit email / password
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setInfoMessage(null);
    setLoading(true);

    try {
      if (tab === 'login') {
        const res = await loginUser(email.trim(), password);
        if (res.status === 'needs_verification') {
          setPendingEmail(res.email || email.trim());
          setDevCode(res.dev_code || null);
          setInfoMessage(res.message || 'Mã xác thực đã được gửi tới email của bạn.');
          setAuthStep('otp');
        } else if (res.user && res.token) {
          onSuccess(res.user, res.token);
        }
      } else {
        const res = await registerUser(email.trim(), password, fullName.trim());
        if (res.status === 'needs_verification') {
          setPendingEmail(res.email || email.trim());
          setDevCode(res.dev_code || null);
          setInfoMessage('Đăng ký thành công! Nhập mã OTP để kích hoạt tài khoản.');
          setAuthStep('otp');
        } else if (res.user && res.token) {
          onSuccess(res.user, res.token);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Đã có lỗi xảy ra');
    } finally {
      setLoading(false);
    }
  };

  // Verify OTP
  const handleVerifyOtp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!otpCode.trim() || otpCode.trim().length < 6) {
      setError('Vui lòng nhập đủ 6 chữ số mã xác thực.');
      return;
    }

    setError(null);
    setLoading(true);
    try {
      const res = await verifyOtp(pendingEmail, otpCode.trim());
      if (res.user && res.token) {
        onSuccess(res.user, res.token);
      }
    } catch (err: any) {
      setError(err.message || 'Mã xác thực không hợp lệ');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--bg-base)',
      padding: '24px'
    }}>
      <div style={{ width: '100%', maxWidth: '400px' }}>
        {/* Brand */}
        <div style={{ textAlign: 'center', marginBottom: '28px' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: 'var(--radius-md)',
            background: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 14px',
            color: '#ffffff'
          }}>
            <BookOpen size={24} strokeWidth={2.2} />
          </div>
          <h1 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            StudyRAG
          </h1>
          <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Trợ lý nghiên cứu & học tập tài liệu chuyên sâu
          </p>
        </div>

        {/* Card */}
        <div className="surface-card" style={{ padding: '28px', boxShadow: 'var(--shadow-lg)' }}>
          {authStep === 'form' ? (
            <>
              {/* Segmented Switch */}
              <div style={{
                display: 'flex',
                background: 'var(--bg-base)',
                padding: '3px',
                borderRadius: 'var(--radius-md)',
                marginBottom: '20px',
                border: '1px solid var(--border-subtle)'
              }}>
                <button
                  type="button"
                  onClick={() => { setTab('login'); setError(null); }}
                  style={{
                    flex: 1,
                    padding: '7px',
                    borderRadius: 'var(--radius-sm)',
                    border: 'none',
                    background: tab === 'login' ? 'var(--bg-elevated)' : 'transparent',
                    color: tab === 'login' ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontWeight: 600,
                    fontSize: '13px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  Đăng Nhập
                </button>
                <button
                  type="button"
                  onClick={() => { setTab('register'); setError(null); }}
                  style={{
                    flex: 1,
                    padding: '7px',
                    borderRadius: 'var(--radius-sm)',
                    border: 'none',
                    background: tab === 'register' ? 'var(--bg-elevated)' : 'transparent',
                    color: tab === 'register' ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontWeight: 600,
                    fontSize: '13px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  Đăng Ký
                </button>
              </div>

              {/* Quick Demo Button */}
              <button
                type="button"
                onClick={handleQuickDemo}
                className="btn-secondary"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  fontSize: '13px',
                  fontWeight: 600,
                  marginBottom: '10px',
                  color: 'var(--primary)',
                  borderColor: 'var(--primary-border)',
                  background: 'var(--primary-subtle)'
                }}
              >
                <Zap size={14} />
                <span>Vào trực tiếp hệ thống (Dùng thử)</span>
              </button>

              {/* Google Button */}
              <button
                type="button"
                onClick={handleGoogleLogin}
                disabled={googleLoading}
                className="btn-secondary"
                style={{
                  width: '100%',
                  padding: '9px 14px',
                  fontSize: '13px',
                  marginBottom: '16px'
                }}
              >
                {googleLoading ? (
                  <Loader2 size={15} className="animate-spin" />
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
                  </svg>
                )}
                <span>Tiếp tục với Google</span>
              </button>

              {/* Divider */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                margin: '14px 0 16px',
                color: 'var(--text-muted)',
                fontSize: '11px',
                textTransform: 'uppercase',
                letterSpacing: '0.04em'
              }}>
                <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }} />
                <span>hoặc với email</span>
                <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }} />
              </div>

              {/* Error / Info messages */}
              {error && (
                <div style={{
                  padding: '9px 12px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--danger-subtle)',
                  border: '1px solid var(--danger-border)',
                  color: '#f87171',
                  fontSize: '12.5px',
                  marginBottom: '14px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '6px'
                }}>
                  <AlertCircle size={15} style={{ flexShrink: 0, marginTop: '2px' }} />
                  <span>{error}</span>
                </div>
              )}

              {infoMessage && (
                <div style={{
                  padding: '9px 12px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--success-subtle)',
                  border: '1px solid var(--success-border)',
                  color: '#34d399',
                  fontSize: '12.5px',
                  marginBottom: '14px'
                }}>
                  {infoMessage}
                </div>
              )}

              {/* Form */}
              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {tab === 'register' && (
                  <div>
                    <label style={{ display: 'block', fontSize: '12.5px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '5px' }}>
                      Họ và Tên
                    </label>
                    <input
                      type="text"
                      required
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="Nguyễn Văn A"
                      className="input-field"
                      style={{ width: '100%', fontSize: '13.5px' }}
                    />
                  </div>
                )}

                <div>
                  <label style={{ display: 'block', fontSize: '12.5px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '5px' }}>
                    Email
                  </label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="ban@example.com"
                    className="input-field"
                    style={{ width: '100%', fontSize: '13.5px' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12.5px', fontWeight: 500, color: 'var(--text-secondary)', marginBottom: '5px' }}>
                    Mật khẩu
                  </label>
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className="input-field"
                    style={{ width: '100%', fontSize: '13.5px' }}
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="btn-primary"
                  style={{ width: '100%', marginTop: '6px', padding: '10px' }}
                >
                  {loading ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : tab === 'login' ? (
                    <span>Đăng Nhập</span>
                  ) : (
                    <span>Đăng Ký Tài Khoản</span>
                  )}
                </button>
              </form>
            </>
          ) : (
            /* OTP Step */
            <div>
              <div style={{ textAlign: 'center', marginBottom: '18px' }}>
                <h2 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
                  Xác Thực Mã OTP
                </h2>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                  Mã 6 chữ số đã được gửi tới: <strong style={{ color: 'var(--text-primary)' }}>{pendingEmail}</strong>
                </p>
              </div>

              {devCode && (
                <div style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--accent-amber-subtle)',
                  border: '1px dashed var(--accent-amber-border)',
                  color: '#fbbf24',
                  fontSize: '12.5px',
                  marginBottom: '14px',
                  textAlign: 'center'
                }}>
                  Mã thử nghiệm: <strong>{devCode}</strong>
                  <button
                    type="button"
                    onClick={() => setOtpCode(devCode)}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--primary)',
                      textDecoration: 'underline',
                      cursor: 'pointer',
                      fontSize: '12px',
                      display: 'block',
                      margin: '4px auto 0'
                    }}
                  >
                    Bấm để điền tự động
                  </button>
                </div>
              )}

              {error && (
                <div style={{
                  padding: '9px 12px',
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--danger-subtle)',
                  border: '1px solid var(--danger-border)',
                  color: '#f87171',
                  fontSize: '12.5px',
                  marginBottom: '14px'
                }}>
                  {error}
                </div>
              )}

              <form onSubmit={handleVerifyOtp} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                <input
                  type="text"
                  required
                  maxLength={6}
                  value={otpCode}
                  onChange={(e) => setOtpCode(e.target.value.replace(/\D/g, ''))}
                  placeholder="000000"
                  className="input-field mono"
                  style={{ width: '100%', textAlign: 'center', fontSize: '20px', letterSpacing: '6px' }}
                />

                <button
                  type="submit"
                  disabled={loading}
                  className="btn-primary"
                  style={{ width: '100%', padding: '10px' }}
                >
                  {loading ? <Loader2 size={16} className="animate-spin" /> : 'Xác Nhận'}
                </button>
              </form>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
export default AuthScreen;
