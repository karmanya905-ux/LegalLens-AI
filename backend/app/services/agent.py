
import json
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from app.services.rag import RAGService


class LegalAgent:
    def __init__(self, rag: RAGService):
        self.rag = rag

    def run(self, document_id: str, request: str) -> dict:
        # These values are bound server-side for this request.
        # The LLM cannot choose a different document ID.
        sources_used = []
        action_used = []

        @tool
        def answer_document_question(question: str) -> str:
            """Answer a specific factual question using the selected document's relevant passages."""
            response = self.rag.answer(document_id, question)

            sources = response.get("sources", [])
            sources_used.extend(sources)
            action_used.append("question")

            return json.dumps({
                "answer": response["answer"],
                "sources": sources,
            })

        @tool
        def summarize_contract() -> str:
            """Summarize the selected legal document, including purpose, obligations, dates, notice periods, and termination terms."""
            excerpts = self.rag.retrieve(
                document_id,
                "Main purpose, parties, obligations, dates, notice periods, and termination terms",
                limit=8,
            )

            sources_used.extend(excerpts)
            action_used.append("summary")

            if not excerpts:
                return "No relevant indexed passages were found."

            prompt = (
                "Summarize this legal document for educational purposes. "
                "Use only the supplied excerpts. Organize the answer into "
                "Purpose, Key obligations, Dates and notice periods, and "
                "Missing or unclear terms. Cite factual claims using the "
                "provided [Page N] labels. Do not give legal advice.\n\n"
                + "\n\n".join(excerpts)
            )

            return self.rag.generate(prompt)

        @tool
        def review_contract_risks() -> str:
            """Review the selected contract for potentially risky, unclear, one-sided, or missing terms. This is educational, not legal advice."""
            excerpts = self.rag.retrieve(
                document_id,
                "Potentially risky, unclear, one-sided, or missing contractual terms",
                limit=10,
            )

            sources_used.extend(excerpts)
            action_used.append("risk_review")

            if not excerpts:
                return "No relevant indexed passages were found."

            prompt = (
                "Review these contract excerpts for issues worth examining. "
                "Do not assert illegality. For each issue explain why it may "
                "matter, a question to ask, and the supporting page citation. "
                "Distinguish missing information from actual risks. "
                "Educational information only, not legal advice.\n\n"
                + "\n\n".join(excerpts)
            )

            return self.rag.generate(prompt)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        model = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=api_key,
            temperature=0.1,
        )

        agent = create_agent(
            model=model,
            tools=[
                answer_document_question,
                summarize_contract,
                review_contract_risks,
            ],
            system_prompt=(
                "You are LegalLens AI, an assistant for understanding legal "
                "documents. Choose the most appropriate tool based on the "
                "user's intent. Use the question-answering tool for specific "
                "factual questions, the summary tool for requests to summarize "
                "or explain the document as a whole, and the risk-review tool "
                "for risks, red flags, unclear terms, or one-sided clauses. "
                "Normally call one tool, then explain its result clearly. "
                "Do not invent facts or legal citations. This is educational "
                "information, not legal advice."
            ),
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"recursion_limit": 8},
        )

        messages = result.get("messages", [])

        print("\n--- LANGGRAPH TOOL CALL TRACE ---")
        for message in messages:
            for call in getattr(message, "tool_calls", []):
                print("LLM selected tool:", call.get("name"))
        print("--- END TOOL CALL TRACE ---\n")

        final_answer = ""

        # The last assistant message is normally the final response.
        for message in reversed(messages):
            if getattr(message, "type", "") == "ai" and not getattr(
                message, "tool_calls", []
            ):
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    final_answer = content
                elif isinstance(content, list):
                    final_answer = "\n".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("text")
                    )
                break

        return {
            "action": action_used[-1] if action_used else "question",
            "result": final_answer or "The agent did not produce a final answer.",
            "sources": list(dict.fromkeys(sources_used)),
        }
