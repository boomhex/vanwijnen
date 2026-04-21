import os
import json
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types

text_path = "text.txt"


def extract_first_json_object(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("Geen JSON-object gevonden.")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    return json.loads(candidate)

    raise ValueError("Geen volledig JSON-object kunnen extraheren.")


def main():
    with open(text_path, 'r') as file:
        text = file.read()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Zet eerst GEMINI_API_KEY in je environment.")

    client = genai.Client(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "Aannemer": {"type": "string"},
            "Totale prijs": {"type": "string"},
            "Offerteposten": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "omschrijving": {"type": "string"},
                        "totaalprijs": {"type": "string"},
                        "eenheidsprijs": {"type" : "string"},
                        "eenheid": {"type" : "string"},
                    },
                    "required": ["omschrijving", "totaalprijs", "eenheidsprijs", "eenheid"],
                },
            },
        },
        "required": ["Aannemer", "Totale prijs", "Offerteposten"],
    }

    system_instruction = """
    
Haal de naam van de aannemer, en de totaalprijs uit de tekst.
Haal alle offerteposten uit deze tekst.
Voor elke offertepost geef:
- omschrijving
- prijs
- eenheidsprijs
- eenheid

Regels:
- Geef alleen JSON terug
- Geen extra tekst
- Geen waarden gokken
- Als prijs niet gevonden wordt, gebruik "Niet gevonden"
"""

    print(text)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )

    print("RAW RESPONSE:")
    print(response.text)

    data = extract_first_json_object(response.text)

    print("\nPARSED JSON:")
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()