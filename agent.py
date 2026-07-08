from crewai import Agent, Task, Crew, Process, LLM
from tools import run_pandas_code, get_schema_context
from config import CSV_PATH, SCHEMA_SAMPLE_ROWS, LLM_MODEL, LLM_API_KEY #, LLM_BASE_URL

# Built ONCE at module import time and reused across all requests.
# This client is a stateless config wrapper around the NIM API, so it's
# safe to share across concurrent requests.
llm = LLM(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    #base_url=LLM_BASE_URL,
)

analyst = Agent(
    role="Data Analyst",
    goal=(
        "Answer questions accurately about whatever dataset is loaded, by "
        "first understanding its schema and then querying it with pandas."
    ),
    backstory=(
        "You are an expert data analyst who adapts to any dataset you're "
        "given — sales records, survey responses, logs, customer data, or "
        "anything else. You never assume what the data represents ahead of "
        "time; you read the schema and sample rows you're given to infer "
        "the domain, the meaning of each column, and how they relate. You "
        "never load the raw data into the prompt — instead you write "
        "precise pandas code to query it and return clear, concise natural "
        "language answers grounded in what the data actually shows."
    ),
    tools=[run_pandas_code],
    llm=llm,
    verbose=True,
)


def _format_history_block(turns: list[dict]) -> str:
    """
    turns: list of {"role": "user"|"assistant", "content": str, ...}
    ordered oldest -> newest (already the last K full turns from the DB).
    """
    if not turns:
        return ""

    lines = ["\n\n--- Conversation so far ---"]
    for turn in turns:
        prefix = "Q" if turn["role"] == "user" else "A"
        lines.append(f"{prefix}: {turn['content']}")
    lines.append("--- End of history ---\n")
    return "\n".join(lines)


def build_crew(question: str, recent_turns: list[dict]) -> Crew:
    """
    Build a single-agent Crew for one question.
    `recent_turns` are the last K full turns for this session, pulled from
    the DB, injected so the agent can answer follow-up questions.
    """
    schema_ctx = get_schema_context(CSV_PATH, SCHEMA_SAMPLE_ROWS)
    history_block = _format_history_block(recent_turns)

    task = Task(
        description=(
            f"Dataset schema & sample:\n{schema_ctx}"
            f"{history_block}\n\n"
            f"User question: {question}\n\n"
            "Instructions:\n"
            "1. Write pandas code that queries `df` and assigns the answer to `result`.\n"
            "2. Call the `run_pandas_code` tool with that code.\n"
            "3. Interpret the tool output and answer the user in plain English.\n"
            "4. If a follow-up question references a previous answer, use that context.\n"
            "5. Keep your final answer concise and human-friendly."
        ),
        expected_output="A clear natural language answer to the user's question.",
        agent=analyst,
    )

    return Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )