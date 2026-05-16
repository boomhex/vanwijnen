import json
from pathlib import Path

import os
from google import genai
from google.genai import types
import pdfplumber


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

def read_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        pages = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    txt = '\n'.join(pages)
    return txt

def read_txt(file):
    txt = ""
    with open(file, 'r') as f:
        txt = f.read()
    return txt

def extract_offer(file: Path):
    prompt = read_txt(Path("./prompts/extract_prompt.txt"))
    offer = read_pdf(file)

    answer = ask_llm('\n'.join([prompt, offer]))

    Path('./tmp/results').mkdir(exist_ok=True)
    with open(f'./tmp/results/{file.stem}.txt', 'a') as f:
        f.write(answer)

def compare_files(files):
    for file in files:
        extract_offer(file)

