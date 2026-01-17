from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_agentchat.messages import TextMessage
from .schemas.index import RetrivalSchema
from pydantic import ValidationError
import requests



model = OllamaChatCompletionClient(model="mistral", base_url="http://localhost:11434")

# Tools
def web_search(query: str) -> str:
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1}
    r = requests.get(url, params=params)
    data = r.json()
    print("Web Search Data:", data)
    return data.get("AbstractText") or f"No free answer found for {query}"


content_test = AssistantAgent(
    name="ContentTester",
    model_client=model,
    system_message="""
You are a Query–Response evaluator.

You MUST return ONLY valid JSON.
Do NOT add explanations, comments, or text outside JSON.
Do NOT use markdown.

The JSON MUST match this schema EXACTLY:

{
  "score": number (0–100),
  "description": string
}
"""
)

external_knowladge = AssistantAgent(
    name="ExternalKnowledgeAgent",
    model_client=model,
    tools=[web_search],
    system_message='''
    You are an agent that can fetch information
    from external knowledge sources when the internal knowledge
    base does not provide sufficient information.
    DO NOT mention tool usage, function calls, or search steps.
    Return ONLY the final answer to the user.
'''
)



async def test_content(query: str, response: str) -> RetrivalSchema:
    user_prompt = f"""
Query:
{query}

Response:
{response}
"""

    messages = [
        TextMessage(content=user_prompt, source="user")
    ]

    result = await content_test.on_messages(
        messages,
        cancellation_token=CancellationToken()
    )

    raw_output = result.chat_message.content
    print("Raw LLM Output:", raw_output)

    try:
        evaluation = RetrivalSchema.model_validate_json(raw_output)
        print("Parsed Evaluation:", evaluation.score)
        return evaluation
    except ValidationError as e:
        raise RuntimeError(f"Invalid JSON returned by LLM: {e}")

async def fetch_external_knowledge(query: str) -> str:
    messages = [
        TextMessage(content=query, source="user")
    ]

    result = await external_knowladge.on_messages(
        messages,
        cancellation_token=CancellationToken()
    )
    print("External Knowledge Response:", result.chat_message.content)
    return result.chat_message.content