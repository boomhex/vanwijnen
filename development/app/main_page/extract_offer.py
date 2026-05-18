import json
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

import os
from google import genai
from google.genai import types
import pdfplumber


MODEL_ID = "google/gemma-4-E4B-it"


client = None


def get_client():
    global client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

    return client


def ask_llm(prompt: str) -> str:
    while True:
        try:
            print("Generating Response")
            response = get_client().models.generate_content(
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


def parse_json_response(answer: str) -> dict:
    cleaned_answer = answer.strip()
    markdown_json = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', cleaned_answer, re.DOTALL)
    if markdown_json:
        cleaned_answer = markdown_json.group(1).strip()

    return json.loads(cleaned_answer)


def parse_money_value(value: str) -> Decimal | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() == 'ONBEKEND':
        return None

    cleaned = re.sub(r'[^\d,.\-]', '', text)
    if cleaned.count(',') > 1:
        raise ValueError(f'Invalid money amount: {value}')

    if ',' in cleaned and '.' in cleaned:
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    elif '.' in cleaned:
        parts = cleaned.split('.')
        if len(parts[-1]) != 2:
            cleaned = cleaned.replace('.', '')

    try:
        return Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f'Invalid money amount: {value}') from error


def validate_offer_json(offer_json: dict) -> list[str]:
    warnings = []
    required_keys = ['Naam opdrachtgever', 'Totaalprijs inc. BTW', 'Totaalprijs exc. BTW', 'Posten']

    for key in required_keys:
        if key not in offer_json:
            warnings.append(f'Missing field: {key}')

    posten = offer_json.get('Posten', [])
    if not isinstance(posten, list):
        warnings.append('Posten must be a list')
        return warnings

    money_fields = [
        ('Totaalprijs inc. BTW', offer_json.get('Totaalprijs inc. BTW')),
        ('Totaalprijs exc. BTW', offer_json.get('Totaalprijs exc. BTW')),
    ]
    for index, post in enumerate(posten, start=1):
        money_fields.append((f'Post {index} Totaalbedrag', post.get('Totaalbedrag')))
        money_fields.append((f'Post {index} Eenheidsprijs', post.get('Eenheidsprijs')))

    for label, value in money_fields:
        try:
            parse_money_value(value)
        except ValueError:
            warnings.append(f'{label} has an invalid money format: {value}')

    known_post_totals = []
    for post in posten:
        try:
            amount = parse_money_value(post.get('Totaalbedrag'))
        except ValueError:
            amount = None
        if amount is not None:
            known_post_totals.append(amount)

    try:
        total_exc = parse_money_value(offer_json.get('Totaalprijs exc. BTW'))
    except ValueError:
        total_exc = None

    try:
        total_inc = parse_money_value(offer_json.get('Totaalprijs inc. BTW'))
    except ValueError:
        total_inc = None

    if total_exc is not None and known_post_totals:
        post_sum = sum(known_post_totals, Decimal('0'))
        if abs(post_sum - total_exc) > Decimal('0.02'):
            warnings.append(f'Sum of post totals ({post_sum}) does not match total excl. BTW ({total_exc})')

    if total_exc is not None and total_inc is not None:
        expected_inc = (total_exc * Decimal('1.21')).quantize(Decimal('0.01'))
        if abs(expected_inc - total_inc) > Decimal('0.02'):
            warnings.append(f'Total incl. BTW ({total_inc}) does not match 21% BTW over excl. total ({expected_inc})')

    return warnings


def extract_offer(file: Path, results_path: Path):
    if isinstance(results_path, str):
        results_path = Path(results_path)
    elif not isinstance(results_path, Path):
        raise TypeError(f'Path: {results_path} Invalid')

    prompt = read_txt(Path("./prompts/extract_prompt.txt"))
    offer = read_pdf(file)

    answer = ask_llm('\n'.join([prompt, offer]))
    offer_json = parse_json_response(answer)
    validate_offer_json(offer_json)

    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f'{file.stem}.txt', 'w') as f:
        json.dump(offer_json, f, ensure_ascii=False, indent=4)

    return offer_json


def compare_files(files, results_path: Path = Path('./tmp/results')):
    return [extract_offer(file, results_path) for file in files]
