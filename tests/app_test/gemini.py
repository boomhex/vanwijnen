import json
from pathlib import Path

import pdfplumber
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


PDF_PATH = Path(
    "../../data/03.02_Offertes_ontvangen/05.50_maatvoering/BDM/20250925-BDM25965-Offerte3465.pdf"
)

MODEL_ID = "google/gemma-4-E4B-it"


def extract_pdf_text(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
    return "\n\n".join(parts)


def main():
    text = extract_pdf_text(PDF_PATH)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        padding_side="left",
        use_fast=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )

    messages = [
        {
            "role": "system",
            "content": "Je extraheert offertegegevens uit Nederlandse bouwdocumenten. Geef alleen geldige JSON terug."
        },
        {
            "role": "user",
            "content": f"""
Haal de relevante informatie uit deze tekst:
- Naam aannemer
- Totale prijs
- Korte samenvatting van bijzondere bepalingen
- Alle offerteposten, met "omschrijving" en "prijs"

Regels:
- Geen waarden gokken
- Als een waarde niet gevonden kan worden, gebruik exact: "Niet gevonden"

Gebruik exact deze JSON-structuur:
{{
  "Aannemer": "",
  "Totale prijs": "",
  "Samenvatting": "",
  "Offerteposten": [
    {{
      "omschrijving": "",
      "prijs": ""
    }}
  ]
}}

TEKST:
{text}
"""
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
        )

    generated = outputs[0][input_len:]
    response_text = processor.decode(generated, skip_special_tokens=True)

    print(response_text)

    try:
        data = json.loads(response_text)
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print("Output was geen geldige JSON.")


if __name__ == "__main__":
    main()