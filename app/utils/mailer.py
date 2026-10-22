import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import settings

logger = logging.getLogger("app.mailer")


def send_verification_email(to_email: str, code: str) -> bool:
    """
    Gửi mã OTP qua SMTP nếu đã cấu hình (SMTP_HOST/SMTP_USER/SMTP_PASSWORD).

    Quy tắc bảo mật: mã OTP chỉ được ghi ra log khi APP_ENV=dev VÀ chưa cấu
    hình SMTP (để tiện thử nghiệm). Ở mọi môi trường khác tuyệt đối không log.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not (smtp_host and smtp_user and smtp_password):
        if settings.app_env == "dev":
            logger.info("[DEV] OTP cho %s = %s (SMTP chưa cấu hình)", to_email, code)
        else:
            logger.error("SMTP chưa được cấu hình — không thể gửi OTP tới %s.", to_email)
        return False

    ttl = settings.otp_ttl_minutes
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Mã xác thực StudyRAG của bạn"
        msg["From"] = f"StudyRAG <{smtp_user}>"
        msg["To"] = to_email

        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;padding:24px;
                    background:#0f1422;color:#f8fafc;border-radius:12px;border:1px solid #3b82f6;">
          <h2 style="margin-top:0;color:#60a5fa;">StudyRAG</h2>
          <p style="color:#cbd5e1;">Mã xác thực tài khoản của bạn:</p>
          <div style="text-align:center;margin:20px 0;padding:16px;border:2px dashed #3b82f6;border-radius:8px;">
            <span style="font-size:32px;font-weight:800;letter-spacing:6px;color:#93c5fd;">{code}</span>
          </div>
          <p style="font-size:13px;color:#94a3b8;">Mã có hiệu lực trong <strong>{ttl} phút</strong>.
          Không chia sẻ mã này cho bất kỳ ai.</p>
        </div>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        logger.info("Đã gửi email OTP tới %s.", to_email)
        return True
    except Exception as exc:
        logger.warning("Không thể gửi email OTP qua SMTP: %s", exc)
        return False
