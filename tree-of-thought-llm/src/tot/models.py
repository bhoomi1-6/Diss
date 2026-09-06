import json
import os
import boto3
import threading
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

completion_tokens = 0
prompt_tokens = 0
_usage_lock = threading.Lock()

BEDROCK_MODEL_ARN = os.environ.get("BEDROCK_MODEL_ARN")

# Bedrock client with timeout and retry config
client = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
    config=Config(
        connect_timeout=30,
        read_timeout=120,
        retries={"max_attempts": 5, "mode": "adaptive"}
    )
)
print("calling claude model on bedrock...")

def _require_model_arn(model):
    if model is None:
        raise RuntimeError(
            "No Bedrock model ARN configured. Set the BEDROCK_MODEL_ARN "
            "environment variable (see .env.example)."
        )
    return model

def claude_prompt(prompt, model=None, temperature=0.7, max_tokens=1000, n=1, stop=None):
    messages = [{"role": "user", "content": prompt}]
    model = _require_model_arn(model or BEDROCK_MODEL_ARN)
    return claude_haiku(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def claude_haiku(messages, model=None, temperature=0.7, max_tokens=1000, n=1, stop=None):
    global completion_tokens, prompt_tokens
    model = _require_model_arn(model or BEDROCK_MODEL_ARN)
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


