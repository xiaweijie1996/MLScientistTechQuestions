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
        user_message = messages[-1]["content"]
        return (
            "Mock Mistral answer:\n"
            "I used the provided context to answer the question.\n\n"
            f"Prompt received:\n{user_message}"
        )


class RetrievalAssistant:
    def __init__(self, search_client: FakeSearchAPI, llm_client: FakeMistralClient) -> None:
        self.search_client = search_client
        self.llm_client = llm_client

    def retrieve_context(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return the most relevant search results for the question."""
        raise NotImplementedError("Implement retrieval first.")

    def build_context_block(self, results: list[SearchResult]) -> str:
        """Convert search results into a clean text block for the model."""
        raise NotImplementedError("Implement context formatting next.")

    def build_messages(self, question: str, context_block: str) -> list[dict[str, str]]:
        """Create the system and user messages for the model call."""
        raise NotImplementedError("Implement prompt construction next.")

    def answer_question(self, question: str) -> dict[str, object]:
        """Run retrieval plus generation and return both answer and sources."""
        raise NotImplementedError("Implement the full pipeline last.")


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