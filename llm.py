import openai
client = openai.OpenAI(base_url="http://172.28.34.1:8000/v1", api_key="token-not-needed")
MODEL = "google/gemma-4-26B-A4B-it"

def ask(system, user, temperature=0.3):
    """Send instructions (system) and a question with evidence (user) to Gemma."""
    r = client.chat.completions.create(model=MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content