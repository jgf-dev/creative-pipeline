import os
from typing import Any, Iterable, cast

from dotenv import load_dotenv
from portkey_ai import Portkey

load_dotenv()

portkey = Portkey(api_key=os.getenv("PORTKEY_API_KEY"))

response = portkey.chat.completions.create(
    model="@xai/grok-4.20-experimental-beta-reasoning-latest",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is Portkey"}],
    max_tokens=512,
    stream=True,
)

# The response is an iterable of chunks when stream=True
for chunk in cast(Iterable[Any], response):
    if hasattr(chunk, "choices") and chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
print("\n")

