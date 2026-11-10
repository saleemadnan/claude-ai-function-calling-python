import time

import anthropic

MODEL = "claude-opus-5-5"

# measure time
start_time = time.time()


def score_checker(score):
    if score > 0.5:
        return "Pass"
    else:
        return "Fail"


TOOLS = [
    {
        "name": "score_checker",
        "description": "Takes the exam score (a number between 0 and 1) and returns whether it is a Pass or a Fail.",
        "input_schema": {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        },
    }
]


def run_tool(name, tool_input):
    if name == "score_checker":
        return score_checker(float(tool_input["score"]))
    raise ValueError(f"Unknown tool: {name}")


client = anthropic.Anthropic()
messages = [{"role": "user", "content": "I got the result. My score is 0.4 last night"}]

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
            print(f"{block.name}({block.input}) -> {result}")
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
