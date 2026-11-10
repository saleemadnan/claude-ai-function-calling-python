import os
import time

import anthropic
import serpapi

MODEL = "claude-opus-5-5"

# measure time
start_time = time.time()


def get_search_result(query):
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
    results = client.search(
        {
            "q": query,
            "engine": "google",
        }
    )

    if "answer_box" not in results:
        return "No answer box found"

    return results["answer_box"]


TOOLS = [
    {
        "name": "get_search_result",
        "description": "Searches Google for the given query and returns the answer box, if there is one.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]


def run_tool(name, tool_input):
    if name == "get_search_result":
        return get_search_result(tool_input["query"])
    raise ValueError(f"Unknown tool: {name}")


client = anthropic.Anthropic()
messages = [{"role": "user", "content": "What's the weather in new York today"}]

# Keep going until Claude stops asking for tools
while True:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        tools=TOOLS,
        messages=messages,
        # If the model declines a request, let the API retry it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason != "tool_use":
        break

    # Send the assistant turn back unchanged, then answer every tool call in ONE user message
    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        try:
            result = run_tool(block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
        except Exception as e:
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True}
            )
    messages.append({"role": "user", "content": tool_results})

if response.stop_reason == "refusal":
    print("The model declined to answer this request.")
else:
    for block in response.content:
        if block.type == "text":
            print(block.text)

# measure time
print("--- %s seconds ---" % (time.time() - start_time))
