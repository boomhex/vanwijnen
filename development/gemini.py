import json
from pathlib import Path

import os
from google import genai
from google.genai import types

MODEL_ID = "google/gemma-4-E4B-it"

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def ask_llm(prompt: str) -> str:
    while True:
        try:
            print("Generating Response")
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e.message)
    return response.text

def main():
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="How many r's in strawberry?",
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )

    print(response.text)


if __name__ == "__main__":
    main()