"""Tests for LCEL chains and agent tool schemas."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent import calculate, get_current_time, search_web
from chains import FactCheck, build_parallel_chain, build_qa_chain, build_structured_chain
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda


def test_qa_chain_returns_string():
    model = FakeListChatModel(responses=["Paris is the capital of France."])
    retriever = RunnableLambda(lambda _: "France is a European country. Paris is its capital.")
    chain = build_qa_chain(model, retriever)
    result = chain.invoke("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_structured_chain_returns_pydantic():
    fake_fact_check = FactCheck(claim="test", is_supported=True, confidence=0.9)
    fake_structured_model = MagicMock()
    fake_inner_chain = MagicMock()
    fake_inner_chain.invoke = MagicMock(return_value=fake_fact_check)
    fake_structured_model.with_structured_output = MagicMock(return_value=fake_inner_chain)

    with patch("chains.ChatPromptTemplate.from_template") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=fake_inner_chain)

        chain = build_structured_chain(fake_structured_model, FactCheck)
        result = chain.invoke({"claim": "test"})

    assert isinstance(result, FactCheck)
    assert result.claim == "test"
    assert result.is_supported is True
    assert result.confidence == 0.9


def test_parallel_chain_has_two_keys():
    model = FakeListChatModel(responses=["a short summary", "keyword1 keyword2 keyword3"])
    chain = build_parallel_chain(model)
    result = chain.invoke({"input": "LangChain is a framework for building LLM applications."})
    assert isinstance(result, dict)
    assert "summary" in result
    assert "keywords" in result


def test_tool_schemas_have_descriptions():
    tools = [search_web, calculate, get_current_time]
    for t in tools:
        assert t.description, f"Tool '{t.name}' has an empty description"
        assert len(t.description.strip()) > 0
