"""acall_structured: schema lồng nhau được inline `$ref` trước khi gửi cho provider.
Không gọi LLM thật: get_llm được thay bằng model giả."""
import asyncio
import json

from pydantic import BaseModel

from app.agents.quiz_summary_agent import QuizBatch
from app.llm import router as llm_router
from app.llm.utils import _structured_schema, acall_structured, inline_schema_refs


class _Flat(BaseModel):
    label: str


class _Node(BaseModel):
    name: str
    children: list["_Node"] = []


class _FakeStructuredLLM:
    def __init__(self, reply):
        self.reply, self.schemas = reply, []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        outer = self

        class _Runnable:
            async def ainvoke(self, _messages):
                return outer.reply

        return _Runnable()


def test_inline_removes_refs_and_defs():
    schema = inline_schema_refs(QuizBatch.model_json_schema())
    dumped = json.dumps(schema)
    assert "$ref" not in dumped and "$defs" not in dumped
    item = schema["properties"]["questions"]["items"]
    assert item["type"] == "object"
    assert {"question", "choices", "correct_choice"} <= set(item["properties"])


def test_flat_model_keeps_class():
    assert _structured_schema(_Flat) is _Flat


def test_recursive_model_falls_back_to_class():
    assert _structured_schema(_Node) is _Node


def test_acall_structured_sends_inlined_dict_and_validates(monkeypatch):
    reply = {"questions": [{
        "question": "Q?", "choices": ["a", "b", "c", "d"], "correct_choice": "a",
        "explanation": "vì a", "topic": "T", "difficulty": "easy",
    }]}
    fake = _FakeStructuredLLM(reply)
    monkeypatch.setattr(llm_router, "get_llm", lambda task, **kw: fake)

    batch = asyncio.run(acall_structured("quiz", "sys", "user", QuizBatch))

    assert isinstance(batch, QuizBatch) and batch.questions[0].correct_choice == "a"
    assert isinstance(fake.schemas[0], dict) and "$defs" not in fake.schemas[0]
