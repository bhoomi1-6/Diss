# import os
# import openai
# import backoff 

# completion_tokens = prompt_tokens = 0

# api_key = os.getenv("OPENAI_API_KEY", "")
# if api_key != "":
#     openai.api_key = api_key
# else:
#     print("Warning: OPENAI_API_KEY is not set")
    
# api_base = os.getenv("OPENAI_API_BASE", "")
# if api_base != "":
#     print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
#     openai.api_base = api_base

# @backoff.on_exception(backoff.expo, openai.error.OpenAIError)
# def completions_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)

# def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     messages = [{"role": "user", "content": prompt}]
#     return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
# def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
#     global completion_tokens, prompt_tokens
#     outputs = []
#     while n > 0:
#         cnt = min(n, 20)
#         n -= cnt
#         res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
#         outputs.extend([choice.message.content for choice in res.choices])
#         # log completion tokens
#         completion_tokens += res.usage.completion_tokens
#         prompt_tokens += res.usage.prompt_tokens
#     return outputs
    
# def gpt_usage(backend="gpt-4"):
#     global completion_tokens, prompt_tokens
#     if backend == "gpt-4":
#         cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
#     elif backend == "gpt-3.5-turbo":
#         cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
#     elif backend == "gpt-4o":
#         cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
#     return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


import json
import boto3
import threading
from botocore.config import Config

completion_tokens = 0
prompt_tokens = 0
_usage_lock = threading.Lock()

# Bedrock client with timeout and retry config
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(
        connect_timeout=30,
        read_timeout=120,
        retries={"max_attempts": 5, "mode": "adaptive"}
    )
)
print("calling claude model on bedrock...")

def claude_prompt(prompt, model="arn:aws:bedrock:us-east-1:340325702211:application-inference-profile/zp8vw056ctc0", temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{"role": "user", "content": prompt}]
    model = "arn:aws:bedrock:us-east-1:340325702211:application-inference-profile/zp8vw056ctc0"
    return claude_haiku(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def claude_haiku(messages, model="arn:aws:bedrock:us-east-1:340325702211:application-inference-profile/zp8vw056ctc0", temperature=0.7, max_tokens=1000, n=1, stop=None):
    global completion_tokens, prompt_tokens
    outputs = []

    prompt = messages[-1]["content"]

    for _ in range(n):

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        response = client.invoke_model(
            modelId=model,
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())

        outputs.append(result["content"][0]["text"])

        usage = result.get("usage", {})
        with _usage_lock:
            prompt_tokens += usage.get("input_tokens", 0)
            completion_tokens += usage.get("output_tokens", 0)

    return outputs


def claude_usage(backend="bedrock"):
    global completion_tokens, prompt_tokens
    # Claude Haiku 4.5 pricing: input $1.00/1M, output $5.00/1M
    input_cost_per_1k  = 0.001
    output_cost_per_1k = 0.005
    cost = (prompt_tokens / 1000 * input_cost_per_1k) + (completion_tokens / 1000 * output_cost_per_1k)
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost
    }
