# Mock Interview: Retrieval + Mistral in Python

This mock exercise is based on the interview pattern you described:

1. Call a third-party information source.
2. Extract useful context.
3. Send that context into an LLM request.
4. Return an answer with source snippets.

This version is intentionally simplified for API practice. It does not require a real API key.

## Interview Prompt

Build a small Python assistant that answers a user question by combining:

- a retrieval client that returns external information snippets
- a Mistral-like client that uses those snippets as context

You are given two helper classes:

- `FakeSearchAPI`: simulates a third-party retrieval API
- `FakeMistralClient`: simulates a Mistral chat API

Your task is to implement the pipeline that:

1. accepts a user question
2. retrieves the top matching snippets
3. formats those snippets into a context block
4. builds a prompt for the model
5. returns the final answer and the sources used

## What The Interviewer Is Really Testing

This kind of question is usually less about advanced ML and more about practical engineering:

- can you read an API contract quickly
- can you move data from one system to another cleanly
- can you structure code into small functions
- can you handle missing data or empty API results
- can you explain what would change for a real production API

## Step-By-Step Plan

### Step 1: Understand the data shape

The retrieval client returns a list of `SearchResult` objects.

Each result has:

- `title`
- `snippet`
- `url`

Your code should not pass the raw objects directly to the model. First convert them into a clean text block.

### Step 2: Write retrieval logic

Create a method like `retrieve_context(question, top_k=3)`.

Its job:

- call the search client
- keep only the top `k` results
- return them

This is the equivalent of calling a real REST API and parsing JSON.

### Step 3: Format the context

Turn the retrieved results into a single string.

Example shape:

```text
Source 1: Python requests
Snippet: requests lets you send HTTP requests easily.
URL: https://example.com/requests

Source 2: Mistral API
Snippet: Chat completion APIs usually accept a list of messages.
URL: https://example.com/mistral
```

Why this matters:

- LLMs respond better to structured context
- it becomes easy to inspect what was retrieved
- debugging is easier during an interview

### Step 4: Build messages for the LLM

Create a small list of messages:

- one system message that defines behavior
- one user message that contains both context and question

This mirrors a real Mistral chat API call.

### Step 5: Call the model client

Pass the messages into the fake Mistral client.

The fake client returns a deterministic answer so you can test the whole flow without a real key.

### Step 6: Handle edge cases

During an interview, mention these cases explicitly:

- no search results found
- retrieval API timeout
- LLM API failure
- duplicated or low-quality sources
- token limits if context is too long

## Suggested Practice Flow

1. Open `mock_interview_mistral_retrieval_starter.py`.
2. Implement one TODO at a time.
3. Run the file after each step.
4. Compare your result with `mock_interview_mistral_retrieval_solution.py` only after trying yourself.

## What To Say Out Loud In A Real Interview

Use language like this:

- "I will separate retrieval, prompt building, and generation so each part is easy to test."
- "I want the retrieval output in a predictable text format before sending it to the model."
- "If no results come back, I will return a fallback answer instead of hallucinating."
- "In production I would add retries, timeouts, and structured logging around both API calls."

## Stretch Version With Real APIs

If you want to make this closer to a real interview later, replace the fake clients with:

- a real retrieval API, such as SerpAPI, Tavily, Wikipedia, or a company-specific endpoint
- a real Mistral SDK or HTTP request

Then keep the same orchestration code. Only the client layer should change.