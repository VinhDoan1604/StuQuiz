# StudyRAG Frontend

React 19 + Vite + TypeScript. Xem README ở thư mục gốc để chạy toàn bộ hệ thống.

```bash
npm install
npm run dev      # http://localhost:5173, proxy /api -> http://127.0.0.1:8000
npm run lint
npm run build    # -> dist/
```

- Mọi lời gọi backend nằm trong `src/services/api.ts` (base URL mặc định `/api`).
- Đổi cổng backend khi dev: `API_PROXY_TARGET=http://127.0.0.1:9000 npm run dev`.
- Backend ở domain khác: build với `VITE_API_URL=https://api.example.com` và thêm domain frontend vào `CORS_ORIGINS` của backend.
- Deploy: `Dockerfile` build tĩnh rồi phục vụ bằng nginx (`nginx.conf` proxy `/api` tới service `api`, hỗ trợ SSE).
