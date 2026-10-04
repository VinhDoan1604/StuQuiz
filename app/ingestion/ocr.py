"""
app/ingestion/ocr.py

OCR fallback cho các trang PDF dạng scan / không có text layer, hoặc ảnh
upload trực tiếp.

Thiết kế:
- Dùng `pytesseract.image_to_data` (thay vì `image_to_string`) để lấy
  confidence theo từng token -> tính confidence trung bình cho cả trang.
  Caller (parsers.py) dùng giá trị này để quyết định có gắn cờ "cần review
  thủ công" hay không, thay vì âm thầm đẩy text rác vào index.
- Tiền xử lý ảnh bằng Pillow (grayscale + upscale nếu ảnh nhỏ). Nếu
  OpenCV có sẵn trong môi trường, dùng thêm adaptive threshold để tăng độ
  chính xác cho ảnh scan chất lượng thấp — nhưng đây là optional dependency,
  không bắt buộc phải cài để module chạy được.
- `lang="vie"` mặc định theo yêu cầu. Với tài liệu song ngữ (thuật ngữ
  tiếng Anh xen kẽ), có thể gọi `ocr_page_image(..., lang="vie+eng")` nếu
  đã cài traineddata tương ứng.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except ImportError:  # pragma: no cover
    pytesseract = None
    _HAS_PYTESSERACT = False

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    _HAS_CV2 = True
except ImportError:  # OpenCV là optional — fallback về tiền xử lý bằng Pillow
    _HAS_CV2 = False


# Dưới ngưỡng này, kết quả OCR nên được coi là không đáng tin cậy. Caller
# (parsers.py) sẽ gắn cờ "cần review thủ công" thay vì đẩy thẳng vào index
# retrieval, vì text rác trong index còn tệ hơn là thiếu 1 trang.
OCR_MIN_CONFIDENCE = 40.0

# Nếu cạnh ngắn hơn mức này (px), tesseract thường nhận diện kém với font
# nhỏ -> upscale trước khi OCR để cải thiện độ chính xác.
_MIN_DIMENSION_PX = 1500


class OCRError(RuntimeError):
    """OCR không chạy được (thiếu binary, thiếu traineddata, ảnh hỏng, ...)."""


@dataclass
class OCRResult:
    text: str
    confidence: float          # 0-100, trung bình confidence các token nhận diện được
    word_count: int
    low_confidence: bool       # True nếu confidence < OCR_MIN_CONFIDENCE


def _preprocess(image: Image.Image) -> Image.Image:
    """Grayscale + upscale nếu ảnh nhỏ + (nếu có cv2) adaptive threshold."""
    image = ImageOps.exif_transpose(image)  # tránh ảnh bị xoay sai do metadata EXIF
    image = image.convert("L")  # grayscale — tesseract hoạt động tốt hơn trên ảnh xám

    width, height = image.size
    scale = max(1.0, _MIN_DIMENSION_PX / max(1, min(width, height)))
    if scale > 1.0:
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    if not _HAS_CV2:
        return image

    arr = np.array(image)
    arr = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return Image.fromarray(arr)


def ocr_page_image(image_bytes: bytes, lang: str = "vie") -> OCRResult:
    """
    Chạy OCR trên ảnh của một trang.

    Args:
        image_bytes: nội dung ảnh (PNG/JPEG) dạng bytes — thường được render
            từ trang PDF bởi `parsers.py`, nhưng cũng nhận ảnh upload trực tiếp.
        lang: mã ngôn ngữ tesseract. Mặc định "vie". Dùng "vie+eng" cho tài
            liệu song ngữ nếu traineddata tương ứng đã được cài.

    Returns:
        OCRResult chứa text nhận diện được, confidence trung bình (0-100),
        và cờ `low_confidence` để caller quyết định có cần review thủ công.

    Raises:
        OCRError: nếu không đọc được ảnh, thiếu binary tesseract, hoặc
            thiếu traineddata cho `lang`.
    """
    if not _HAS_PYTESSERACT:
        raise OCRError(
            "pytesseract chưa được cài. Chạy `pip install pytesseract` và cài "
            "binary hệ thống `tesseract-ocr` + gói ngôn ngữ `tesseract-ocr-vie` "
            "(xem Dockerfile của project)."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise OCRError(f"Không đọc được ảnh đầu vào cho OCR: {exc}") from exc

    processed = _preprocess(image)

    try:
        data = pytesseract.image_to_data(
            processed,
            lang=lang,
            config="--oem 3 --psm 6",  # PSM 6: giả định 1 khối text đồng nhất (phù hợp trang tài liệu)
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError(
            "Không tìm thấy binary tesseract trong PATH. Kiểm tra Dockerfile "
            "đã cài `tesseract-ocr` và `tesseract-ocr-vie` chưa."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise OCRError(f"Tesseract lỗi khi OCR (lang={lang}): {exc}") from exc

    words: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_f < 0:  # tesseract trả -1 cho vùng được xác định không phải text
            continue
        words.append(text)
        confidences.append(conf_f)

    full_text = " ".join(words)
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    if not words:
        logger.warning("OCR (lang=%s) không nhận diện được token text nào trên trang.", lang)

    return OCRResult(
        text=full_text,
        confidence=avg_confidence,
        word_count=len(words),
        low_confidence=avg_confidence < OCR_MIN_CONFIDENCE,
    )