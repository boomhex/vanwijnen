import os
import json
from pathlib import Path

import pdfplumber
from google import genai
from google.genai import types


PDF_PATH = Path(
    "../../data/03.02_Offertes_ontvangen/22.31_baksteen_met_mortel/Zuidema/25-01506.pdf"
)


def extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
    return "\n\n".join(parts)


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
    text = extract_pdf_text(PDF_PATH)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Zet eerst GEMINI_API_KEY in je environment.")

    client = genai.Client(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "Aannemer": {"type": "string"},
            "Totale prijs": {"type": "string"},
            "Samenvatting": {"type": "string"},
            "Offerteposten": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "omschrijving": {"type": "string"},
                        "prijs": {"type": "string"},
                    },
                    "required": ["omschrijving", "prijs"],
                },
            },
        },
        "required": ["Aannemer", "Totale prijs", "Samenvatting", "Offerteposten"],
    }

    system_instruction = """
Je extraheert offertegegevens uit Nederlandse bouwdocumenten.

Haal de relevante informatie uit de tekst:
- Naam aannemer
- Totale prijs
- Korte samenvatting van bijzondere bepalingen
- Alle offerteposten, met "omschrijving" en "prijs"

Bepaal voor elk van deze posten de prijs uit de ggeven offerte:
- vermetselen gevelsteen wildverband		
- toeslag wilverband		
- toeslag strooisteen		
- inzet verreiker JCB 13.5m1 star t.b.v. vooropperen steen/mortel op de steiger, incl. aan-afvoer.		
- toeslag gebogen metselwerk		
- vermetselen accentsteen 20mm verdiept		
- steen zagen, wildverband,  0,5 mu/dzd		
- voegwerk doorstrijken		

Regels:
- Geen waarden gokken
- Als een waarde niet gevonden kan worden, gebruik exact: "Niet gevonden"
- Geef uitsluitend één geldig JSON-object terug
- Voeg geen toelichting, analyse of extra tekst toe
"""

    print(text)
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
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