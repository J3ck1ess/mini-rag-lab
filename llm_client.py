import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:4b-instruct"


def generate(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]