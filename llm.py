import openai

client = openai.OpenAI(base_url="http://172.28.34.1:8000/v1", api_key="token-not-needed")
MODEL = "google/gemma-4-26B-A4B-it"

def ask(system, user, temperature=0.3):
    """Send instructions (system) and a question with evidence (user) to Gemma."""
    r = client.chat.completions.create(model=MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content

    # ---------- Ask Gemma about an image ----------
import base64
import openai

_img_client = openai.OpenAI(base_url="http://172.28.34.1:8000/v1", api_key="token-not-needed")

def ask_with_image(system, question, image_bytes, mime="image/png"):
    b64 = base64.b64encode(image_bytes).decode()
    r = _img_client.chat.completions.create(
        model="google/gemma-4-26B-A4B-it",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return r.choices[0].message.content

    # ---------- Streaming versions (answer arrives word by word) ----------
MODEL = "google/gemma-4-26B-A4B-it"

def _stream(messages):
    stream = _img_client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.3, max_tokens=900, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def ask_stream(system, question):
    return _stream([
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ])


def ask_with_image_stream(system, question, image_bytes, mime="image/png"):
    b64 = base64.b64encode(image_bytes).decode()
    return _stream([
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]},
    ])