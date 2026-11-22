"""VLM: helper gửi ảnh cho LLM, chép chữ trang scan (OCR_ENGINE=vlm), fallback về Tesseract.
Không gọi LLM thật: get_llm / get_fallback_llm được thay bằng model giả."""
import asyncio
import io

import fitz
import pytest
from langchain_core.messages import AIMessage
from PIL import Image

import app.ingestion.parsers as parsers
from app.config import settings
from app.ingestion.ocr import OCRResult
from app.ingestion.vlm import NO_TEXT_MARKER, VLMError, clean_vlm_output, vlm_transcribe_page
from app.llm import router as llm_router
from app.llm.utils import (
    LLMCallError,
    acall_llm_vision,
    build_vision_messages,
    call_llm_vision,
    image_mime_type,
)


def _png(size=(40, 20)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    return buf.getvalue()


class FakeLLM:
    def __init__(self, reply="xin chào", error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)

    async def ainvoke(self, messages):
        return self.invoke(messages)


@pytest.fixture()
def fake_llms(monkeypatch):
    """primary / fallback giả; trả về dict để test chỉnh hành vi."""
    llms = {"primary": FakeLLM(), "fallback": None, "tasks": []}

    def get_llm(task, streaming=False):
        llms["tasks"].append(task)
        return llms["primary"]

    monkeypatch.setattr(llm_router, "get_llm", get_llm)
    monkeypatch.setattr(llm_router, "get_fallback_llm", lambda task, streaming=False: llms["fallback"])
    return llms


# --- helper ảnh ------------------------------------------------------------------------------

def test_mime_type_is_sniffed_from_bytes_not_extension():
    assert image_mime_type(_png()) == "image/png"
    assert image_mime_type(_jpeg()) == "image/jpeg"
    with pytest.raises(ValueError):
        image_mime_type(b"%PDF-1.7 not an image")


def test_vision_message_has_text_then_one_data_url_per_image():
    system, human = build_vision_messages("sys", "chép chữ", [_png(), _jpeg()])
    assert system.content == "sys"
    parts = human.content
    assert parts[0] == {"type": "text", "text": "chép chữ"}
    urls = [p["image_url"]["url"] for p in parts[1:]]
    assert urls[0].startswith("data:image/png;base64,") and urls[1].startswith("data:image/jpeg;base64,")


def test_vision_message_requires_an_image():
    with pytest.raises(ValueError):
        build_vision_messages("sys", "u", [])


def test_call_llm_vision_routes_through_vision_task(fake_llms):
    assert call_llm_vision("vision", "sys", "u", [_png()]) == "xin chào"
    assert fake_llms["tasks"] == ["vision"]
    sent = fake_llms["primary"].calls[0][1].content
    assert any(p.get("type") == "image_url" for p in sent)


def test_async_vision_helper_returns_text(fake_llms):
    assert asyncio.run(acall_llm_vision("vision", "sys", "u", [_png()])) == "xin chào"


def test_vision_falls_back_once_and_counts_the_fallback(fake_llms):
    fake_llms["primary"] = FakeLLM(error=RuntimeError("429"))
    fake_llms["fallback"] = FakeLLM(reply="từ model dự phòng")
    usage = llm_router.begin_request_usage()
    assert call_llm_vision("vision", "sys", "u", [_png()]) == "từ model dự phòng"
    assert usage["fallbacks"] == 1


def test_vision_without_fallback_raises_llm_call_error(fake_llms):
    fake_llms["primary"] = FakeLLM(error=RuntimeError("model không nhận ảnh"))
    with pytest.raises(LLMCallError):
        call_llm_vision("vision", "sys", "u", [_png()])


def test_vision_task_uses_cheap_model_at_zero_temperature(monkeypatch):
    monkeypatch.setattr(settings, "shopaikey_api_key", "sk-test")
    monkeypatch.setattr(settings, "cheap_model_provider", "shopaikey_google")
    monkeypatch.setattr(settings, "cheap_model_name", "gemini-2.5-flash")
    llm = llm_router.get_llm("vision")
    assert llm.model.endswith("gemini-2.5-flash")
    assert llm.temperature == 0.0
    # Có callback đếm token như mọi task khác
    assert any(isinstance(cb, llm_router.UsageTrackingCallbackHandler) for cb in llm.callbacks)


# --- chép chữ trang --------------------------------------------------------------------------

def test_clean_output_strips_fences_marker_and_normalizes_nfc():
    decomposed = "Tiếng Việt"  # dạng NFD
    assert clean_vlm_output(f"```text\n{decomposed}\n```") == "Tiếng Việt"
    assert clean_vlm_output(f"  {NO_TEXT_MARKER} ") == ""


def test_transcribe_page_wraps_llm_errors(fake_llms):
    fake_llms["primary"] = FakeLLM(error=RuntimeError("timeout"))
    with pytest.raises(VLMError):
        vlm_transcribe_page(_png())


# --- tích hợp parser ------------------------------------------------------------------------

def _scanned_pdf(pages: int = 1) -> bytes:
    """PDF chỉ có ảnh, không text layer -> đi vào nhánh OCR."""
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=300, height=300)
        page.insert_image(fitz.Rect(10, 10, 290, 290), stream=_png((200, 200)))
    data = doc.tobytes()
    doc.close()
    return data


def _fake_tesseract(calls):
    def ocr(image_bytes, lang="vie", min_word_confidence=0.0):
        calls.append(lang)
        return OCRResult(text="chữ từ tesseract", confidence=90.0, word_count=3, low_confidence=False)
    return ocr


def test_default_engine_is_tesseract_and_never_calls_the_vlm(monkeypatch, fake_llms):
    calls = []
    monkeypatch.setattr(parsers, "ocr_page_image", _fake_tesseract(calls))
    assert settings.ocr_engine == "tesseract"
    blocks = parsers.parse_document(_scanned_pdf(), "scan.pdf").blocks
    assert [b.extra["source"] for b in blocks] == ["ocr"]
    assert calls and fake_llms["primary"].calls == []


def test_vlm_engine_transcribes_scanned_page(monkeypatch, fake_llms):
    monkeypatch.setattr(settings, "ocr_engine", "vlm")
    monkeypatch.setattr(parsers, "ocr_page_image", _fake_tesseract([]))
    fake_llms["primary"] = FakeLLM(reply="Kho dữ liệu | Data warehouse")
    doc = parsers.parse_document(_scanned_pdf(), "scan.pdf")
    assert [(b.text, b.extra["source"], b.page) for b in doc.blocks] == [("Kho dữ liệu | Data warehouse", "vlm", 1)]
    assert not doc.warnings


def test_vlm_failure_falls_back_to_tesseract_with_a_warning(monkeypatch, fake_llms):
    monkeypatch.setattr(settings, "ocr_engine", "vlm")
    monkeypatch.setattr(parsers, "ocr_page_image", _fake_tesseract([]))
    fake_llms["primary"] = FakeLLM(error=RuntimeError("503"))
    doc = parsers.parse_document(_scanned_pdf(), "scan.pdf")
    assert [b.extra["source"] for b in doc.blocks] == ["ocr"]
    assert any("VLM lỗi" in w for w in doc.warnings)


def test_vlm_page_without_text_adds_no_block(monkeypatch, fake_llms):
    monkeypatch.setattr(settings, "ocr_engine", "vlm")
    fake_llms["primary"] = FakeLLM(reply=NO_TEXT_MARKER)
    assert parsers.parse_document(_scanned_pdf(), "scan.pdf").blocks == []


def test_ocr_page_limit_still_applies_to_vlm(monkeypatch, fake_llms):
    monkeypatch.setattr(settings, "ocr_engine", "vlm")
    monkeypatch.setattr(settings, "max_ocr_pages", 1)
    with pytest.raises(parsers.ParsingError):
        parsers.parse_document(_scanned_pdf(pages=2), "scan.pdf")
    assert len(fake_llms["primary"].calls) == 1  # dừng trước khi gửi trang thứ 2


def test_invalid_ocr_engine_is_rejected():
    from app.config import Settings

    with pytest.raises(ValueError):
        Settings(ocr_engine="easyocr")
