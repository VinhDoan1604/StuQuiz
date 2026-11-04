/**
 * Đăng nhập Google bằng Google Identity Services (token client).
 * Google tự mở popup và trả access token qua callback — không cần redirect URI,
 * chỉ cần khai báo "Authorized JavaScript origins" trên Google Cloud Console.
 * Access token được gửi lên backend (/auth/google), backend xác minh `aud`
 * đúng client ID của app rồi mới cấp JWT.
 */

const GIS_SRC = 'https://accounts.google.com/gsi/client';

interface TokenResponse {
  access_token?: string;
  error?: string;
  error_description?: string;
}

interface TokenClient {
  requestAccessToken: (overrides?: { prompt?: string }) => void;
}

interface GoogleAccounts {
  oauth2: {
    initTokenClient: (config: {
      client_id: string;
      scope: string;
      prompt?: string;
      callback: (resp: TokenResponse) => void;
      error_callback?: (err: { type: string; message?: string }) => void;
    }) => TokenClient;
  };
}

declare global {
  interface Window {
    google?: { accounts: GoogleAccounts };
  }
}

let gisPromise: Promise<GoogleAccounts> | null = null;

function loadGis(): Promise<GoogleAccounts> {
  if (window.google?.accounts) return Promise.resolve(window.google.accounts);
  if (!gisPromise) {
    gisPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = GIS_SRC;
      script.async = true;
      script.onload = () =>
        window.google?.accounts ? resolve(window.google.accounts) : reject(new Error('Google Identity Services không khả dụng.'));
      script.onerror = () => {
        gisPromise = null;
        reject(new Error('Không tải được thư viện đăng nhập Google (kiểm tra mạng / trình chặn quảng cáo).'));
      };
      document.head.appendChild(script);
    });
  }
  return gisPromise;
}

const POPUP_ERRORS: Record<string, string> = {
  popup_closed: 'Bạn đã đóng cửa sổ đăng nhập Google.',
  popup_failed_to_open: 'Trình duyệt đã chặn cửa sổ đăng nhập. Vui lòng cho phép popup cho trang này.',
};

/** Mở popup Google và trả về access token (reject kèm thông báo tiếng Việt nếu thất bại). */
export async function requestGoogleAccessToken(clientId: string): Promise<string> {
  const accounts = await loadGis();
  return new Promise((resolve, reject) => {
    const client = accounts.oauth2.initTokenClient({
      client_id: clientId,
      scope: 'openid email profile',
      callback: (resp) => {
        if (resp.access_token) resolve(resp.access_token);
        else reject(new Error(resp.error_description || resp.error || 'Google không trả về token.'));
      },
      error_callback: (err) => reject(new Error(POPUP_ERRORS[err.type] || err.message || 'Đăng nhập Google thất bại.')),
    });
    client.requestAccessToken({ prompt: 'select_account' });
  });
}
