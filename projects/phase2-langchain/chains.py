"""LCEL chain patterns: QA, parallel, and structured output."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel


class FactCheck(BaseModel):
    claim: str
    is_supported: bool
    confidence: float


def build_qa_chain(model, retriever) -> RunnableParallel:
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context.\nContext: {context}\nQuestion: {question}"
    )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


def build_parallel_chain(model) -> RunnableParallel:
    prompt_summary = ChatPromptTemplate.from_template("Summarize: {input}")
    prompt_keywords = ChatPromptTemplate.from_template("List three keywords for: {input}")
    summary_branch = prompt_summary | model | StrOutputParser()
    keywords_branch = prompt_keywords | model | StrOutputParser()
    return RunnableParallel(summary=summary_branch, keywords=keywords_branch)


def build_structured_chain(model, schema: type[BaseModel]):
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the factual accuracy of this claim: {claim}"
    )
    return prompt | model.with_structured_output(schema)


if __name__ == "__main__":
    fake_responses = ["Paris is the capital of France.", "summary result", "keywords result"]
    fake_model = FakeListChatModel(responses=fake_responses)

    from langchain_core.runnables import RunnableLambda

    def _fake_retriever(_: str) -> str:
        return "France is a country in Europe. Paris is its capital."

    retriever_runnable = RunnableLambda(_fake_retriever)

    qa_chain = build_qa_chain(fake_model, retriever_runnable)
    qa_result = qa_chain.invoke("What is the capital of France?")
    print("QA chain result:", qa_result)

    parallel_model = FakeListChatModel(responses=["summary text", "keyword1 keyword2 keyword3"])
    parallel_chain = build_parallel_chain(parallel_model)
    parallel_result = parallel_chain.invoke({"input": "LangChain is an AI framework."})
    print("Parallel chain result:", parallel_result)

    fact_check_instance = FactCheck(claim="The sky is blue", is_supported=True, confidence=0.95)
    mock_structured_model = FakeListChatModel(responses=[str(fact_check_instance)])
    mock_structured_model.with_structured_output = lambda schema: (
        ChatPromptTemplate.from_template("{claim}")
        | FakeListChatModel(responses=["ignored"])
        | (lambda _: fact_check_instance)
    )
    structured_chain = build_structured_chain(mock_structured_model, FactCheck)
    print("Structured chain schema:", FactCheck.model_json_schema())
