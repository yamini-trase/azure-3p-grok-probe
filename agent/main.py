"""One chat completion against Grok on Azure AI Foundry, through the governance gateway.

The agent never sees the Foundry host or any key. It addresses the ``azure-foundry``
connection on the gateway (``<TRASE_EGRESS_GATEWAY_URL>/azure-foundry/openai/v1``) with its
run credential as the bearer; the gateway checks the grant, signs in to Foundry with its own
identity and forwards. The step output is the model's reply plus what the gateway reported.
"""

import logging

from openai import OpenAI
from trase_os_sdk import sandbox


log = logging.getLogger(__name__)

CONNECTION = "azure-foundry"
MODEL = "grok-4.7"
PROMPT = "Reply with exactly: pong from grok on azure"


def run():
    """Ask Grok one question through the gateway and return the answer."""
    base_url = sandbox.connection_url(CONNECTION) + "/openai/v1"
    log.info("calling %s (%s) through the gateway: %s", MODEL, CONNECTION, base_url)
    client = OpenAI(base_url=base_url, api_key=sandbox.run_credential(), max_retries=0, timeout=60)
    completion = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0,
    )
    parsed = completion.parse()
    reply = parsed.choices[0].message.content or ""
    usage = parsed.usage
    log.info("grok replied: %s", reply)
    if not reply.strip():
        raise RuntimeError(f"no completion text: {parsed!r}")
    return {
        "connection": CONNECTION,
        "model": MODEL,
        "prompt": PROMPT,
        "reply": reply,
        "usage": {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        },
        "gateway_status": completion.status_code,
        "request_id": completion.headers.get("x-request-id", ""),
    }
