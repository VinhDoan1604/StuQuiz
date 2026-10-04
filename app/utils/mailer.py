import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger("app.mailer")


def send_verification_email(to_email: str, code: str) -> bool:
    """
    Gửi mã OTP xác thực tới Gmail của người dùng.
    Nếu có cấu hình SMTP trong môi trường (.env), gửi email thật qua SMTP server.
    Nếu chưa cấu hình SMTP, in mã nổi bật ra Terminal console để tiện kiểm thử.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    # In ra log hệ thống với khung viền rõ ràng
    logger.info("=" * 60)
    logger.info("[GMAIL VERIFICATION OTP] Đang gửi mã xác thực:")
    logger.info(">> Người nhận : %s", to_email)
    logger.info(">> MÃ XÁC THỰC: %s", code)
    logger.info(">> Hạn sử dụng: 10 phút")
    logger.info("=" * 60)

    if not smtp_host or not smtp_user or not smtp_password:
        logger.info("[GMAIL SERVICE] Chưa cấu hình SMTP credentials (SMTP_USER / SMTP_PASSWORD) trong .env.")
        logger.info("[GMAIL SERVICE] Đã log mã '%s' ra console máy chủ để nhập trực tiếp.", code)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Mã xác thực StudyRAG Scholar của bạn: {code}"
        msg["From"] = f"StudyRAG Scholar <{smtp_user}>"
        msg["To"] = to_email

        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 520px; margin: 0 auto; padding: 24px; background: #090c13; color: #f8fafc; border-radius: 12px; border: 1px solid #f59e0b;">
            <h2 style="color: #f59e0b; margin-top: 0;">StudyRAG Scholar Studio</h2>
            <p style="font-size: 15px; color: #cbd5e1;">Xin chào bạn,</p>
            <p style="font-size: 15px; color: #cbd5e1;">Mã xác thực tài khoản của bạn trên StudyRAG là:</p>
            <div style="background: rgba(245, 158, 11, 0.15); border: 2px dashed #f59e0b; border-radius: 8px; padding: 16px; text-align: center; margin: 20px 0;">
                <span style="font-size: 32px; font-weight: 800; letter-spacing: 6px; color: #fbbf24;">{code}</span>
            </div>
            <p style="font-size: 13.5px; color: #94a3b8;">Mã này có hiệu lực trong vòng <strong>10 phút</strong>. Vui lòng không chia sẻ mã này cho bất kỳ ai.</p>
            <hr style="border: none; border-top: 1px solid #1e293b; margin: 20px 0;" />
            <p style="font-size: 12px; color: #64748b;">Hệ thống Trợ lý Học tập & RAG Chống Ảo Giác StudyRAG.</p>
        </div>
        """

        msg.attach(MIMEText(html_content, "html"))

        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.quit()
        logger.info("[GMAIL SERVICE] Đã gửi email thực tế thành công tới %s!", to_email)
        return True
    except Exception as exc:
        logger.warning("[GMAIL SERVICE] Không thể gửi email qua SMTP: %s", exc)
        return False
