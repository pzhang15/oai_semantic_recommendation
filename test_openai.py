import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads .env if present

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set. Create a .env or set the env var and try again.")

# Optional custom base URL
base_url = os.getenv("OPENAI_BASE_URL") or None

client = OpenAI(api_key=api_key, base_url=base_url)

# 1) Simple embeddings sanity check
model_embed = os.getenv("MODEL_EMBED", "text-embedding-3-small")
resp = client.embeddings.create(model=model_embed, input="hello world")
vec = resp.data[0].embedding
print(f"Embeddings OK → model={model_embed}, dim={len(vec)}")

# 2) Simple parsing sanity check (quick chat completion)
model_parser = os.getenv("MODEL_PARSER", "gpt-4o-mini")
prompt = "Extract main occasion and budget from: 'I need an outfit for a beach trip under $120'."
chat = client.chat.completions.create(
    model=model_parser,
    messages=[{"role":"user","content":prompt}],
)
print("Chat OK →", chat.choices[0].message.content[:120].replace("\n"," "))
print("All good!")

model_ids = [m.id for m in client.models.list().data]

print("Other Models: ")
for model_id in model_ids:
    print("  ", model_id)

