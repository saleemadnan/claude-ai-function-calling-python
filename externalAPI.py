import anthropic
import time
import os
import serpapi

# measure time
start_time = time.time()

def get_search_result(query):
    client = serpapi.Client(api_key=os.getenv("SERPAPI_API_KEY"))
    results = client.search({
        'q':query,
        'engine':"google",
    })

    if 'answer_box' not in results:
        return "No answer box found"
    
    return results['answer_box']

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
            "name": "get_search_result",
            "description": "A function that take a search query from the sentence and search the answer from Google.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "What's the weather in new York today"
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
    if tool_name == "get_search_result":
        result = get_search_result(tool_params["query"])
        # limit the response to 1000 characters only
        if len(str(result)) > 1000:
            result = str(result)[:1000] + "..."
        
        # the answer box can be a dict, so convert it to text for the prompt
        dataFromCustomFunction = str(result)


# Request a chat response from the assistant
response = client.beta.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=16000,
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
    messages=[
        {
            "role": "user",
            "content": "Answer to the customer in a nice way " + dataFromCustomFunction
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
