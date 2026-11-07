import anthropic
import time

# measure time
start_time = time.time()

def score_checker(score):
    if score > 0.5:
        return "Pass"
    else:
        return "Fail"

client = anthropic.Anthropic()
message = client.beta.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=16000,  # covers thinking (on by default) as well as the reply
    # Server-side refusal fallback (beta): if a safety classifier declines the
    # request, the API can re-run it on Anthropic's recommended fallback model.
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    tools=[
        {
            "name": "score_checker",
            "description": "A function that takes the score provided and return the results.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "string"
                    }
                },
                "required": ["score"]
            }
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "I got the result. My score is 0.4 last night"
        }
    ]
)

if message.stop_reason != "tool_use":
    print("End... No tool used (stop_reason: %s)" % message.stop_reason)
    exit()

# Only continue when tools needed
dataFromCustomFunction = None
if message.stop_reason == "tool_use":
    # The response can also contain thinking and text blocks, in any order,
    # so find the tool call by its block type rather than its position.
    tool_use = next(block for block in message.content if block.type == "tool_use")
    tool_params = tool_use.input
    tool_name = tool_use.name
    
    # call available tool function
    if tool_name == "score_checker":
        result = score_checker(float(tool_params["score"]))
        dataFromCustomFunction = result
        print(result)


# Request a chat response from the assistant
response = client.beta.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=16000,
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    messages=[
        {
            "role": "user",
            "content": "Answer to the customer. The customer exam status is " + dataFromCustomFunction
        }
    ]
)

if response.stop_reason == "refusal":
    print("End... The request was declined (stop_reason: refusal)")
    exit()

# Print only the text blocks; the response can start with a thinking block.
print("".join(block.text for block in response.content if block.type == "text"))

# measure time
print("--- %s seconds ---" % (time.time() - start_time))
