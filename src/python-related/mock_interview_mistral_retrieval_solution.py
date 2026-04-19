from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class FakeSearchAPI:
    def __init__(self) -> None:
        self.documents = [
            SearchResult(
                title="Python requests library",
                snippet="The requests library is commonly used to send HTTP requests in Python.",
                url="https://example.com/python-requests",
            ),
            SearchResult(
                title="Mistral chat basics",
                snippet="Chat APIs usually accept a list of messages with roles like system and user.",
                url="https://example.com/mistral-chat",
            ),
            SearchResult(
                title="Retrieval augmented generation",
                snippet="RAG adds external context to an LLM prompt before generation.",
                url="https://example.com/rag-basics",
            ),
            SearchResult(
                title="API error handling",
                snippet="Production API clients should use timeouts, retries, and clear fallback behavior.",
                url="https://example.com/api-errors",
            ),
        ]

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_terms = {term.lower() for term in query.split()}
        scored_results: list[tuple[int, SearchResult]] = []

        for document in self.documents:
            haystack = f"{document.title} {document.snippet}".lower()
            score = sum(term in haystack for term in query_terms)
            scored_results.append((score, document))

        scored_results.sort(key=lambda item: item[0], reverse=True)
        return [result for score, result in scored_results if score > 0][:top_k]


class FakeMistralClient:
    def chat(self, messages: list[dict[str, str]]) -> str:
        context_message = messages[-1]["content"]
        return (
            "Mock Mistral answer:\n"
            "Use retrieval first, then place the retrieved snippets inside the user prompt or a dedicated context block. "
            "Finally, ask the model to answer only from that context and return the sources.\n\n"
            f"Prompt received:\n{context_message}"
        )


class RetrievalAssistant:
    def __init__(self, search_client: FakeSearchAPI, llm_client: FakeMistralClient) -> None:
        self.search_client = search_client
        self.llm_client = llm_client

    def retrieve_context(self, question: str, top_k: int = 3) -> list[SearchResult]:
        return self.search_client.search(question, top_k=top_k)

    def build_context_block(self, results: list[SearchResult]) -> str:
        if not results:
            return "No external context was retrieved."

        blocks = []
        for index, result in enumerate(results, start=1):
            block = (
                f"Source {index}: {result.title}\n"
                f"Snippet: {result.snippet}\n"
                f"URL: {result.url}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def build_messages(self, question: str, context_block: str) -> list[dict[str, str]]:
        system_message = {
            "role": "system",
            "content": (
                "You are a concise assistant. Answer the user only with the provided context. "
                "If the context is insufficient, say so clearly."
            ),
        }
        user_message = {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion:\n{question}",
        }
        return [system_message, user_message]

    def answer_question(self, question: str) -> dict[str, object]:
        results = self.retrieve_context(question)
        context_block = self.build_context_block(results)
        messages = self.build_messages(question, context_block)
        answer = self.llm_client.chat(messages)
        return {"answer": answer, "sources": results}


def main() -> None:
    assistant = RetrievalAssistant(FakeSearchAPI(), FakeMistralClient())
    question = "How would I use retrieval with a Mistral chat API in Python?"

    response = assistant.answer_question(question)
    print("Final answer:\n")
    print(response["answer"])
    print("\nSources used:")
    for source in response["sources"]:
        print(f"- {source.title} | {source.url}")


if __name__ == "__main__":
    main()