import json
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation
import time

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
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        try:
            print("Generating Response")
            response = get_client().models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            if response.text is None:
                raise RuntimeError('LLM response did not contain text')

            return response.text
        except KeyboardInterrupt:
            raise
        except Exception as error:
            error_text = str(error).strip()
            print(f'LLM request failed: {error_text}')

            lowered = error_text.lower()
            if any(token in lowered for token in ('quota', 'rate limit', 'too many requests', 'resource exhausted', '429')):
                raise RuntimeError(
                    'API limit reached. Try again later or increase your Gemini quota.'
                ) from error

            if attempt >= max_attempts:
                raise RuntimeError(f'LLM request failed after {max_attempts} attempts: {error_text}') from error

            time.sleep(2 ** (attempt - 1))

    raise RuntimeError('LLM request failed unexpectedly')


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
    """Parse a money value string into a Decimal.

    Supports many formats such as:
      - "1.234,56" (European)
      - "1,234.56" (US)
      - "12.000,-" (Dutch for 12000.00)
      - "€ 1.234,56", "EUR 1,234.56"
      - "(1.234,56)", "-1.234,56"
      - "1234", "1234.5", "1,5"

    Heuristics:
      - If both '.' and ',' are present the rightmost of the two is the
        decimal separator.
      - If only one separator is present and the fractional part length is 3
        then treat it as a thousands separator, otherwise as decimal.
      - Handles trailing ',-' by converting to ',00' first.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() == 'ONBEKEND':
        return None

    s = text

    # Normalize unicode minus
    s = s.replace('\u2212', '-')

    # Parentheses mean negative: (1.234,56)
    negative = False
    if s.startswith('(') and s.endswith(')'):
        negative = True
        s = s[1:-1].strip()

    # Remove currency symbols and letters, keep digits, separators and sign
    s = re.sub(r'[A-Za-z€£$¥¢\s]', '', s)

    # Dutch-style trailing ',-' means zero cents
    if s.endswith(',-'):
        s = s[:-2] + ',00'

    has_dot = '.' in s
    has_comma = ',' in s

    decimal_sep = None
    if has_dot and has_comma:
        # the rightmost separator is the decimal separator
        decimal_sep = '.' if s.rfind('.') > s.rfind(',') else ','
    elif has_dot:
        after = s.split('.')[-1]
        decimal_sep = '.' if len(after) != 3 else None
    elif has_comma:
        after = s.split(',')[-1]
        decimal_sep = ',' if len(after) != 3 else None

    # Remove thousands separators and normalize decimal separator to dot
    if decimal_sep is None:
        normalized = s.replace('.', '').replace(',', '')
    else:
        thousands = ',' if decimal_sep == '.' else '.'
        normalized = s.replace(thousands, '')
        normalized = normalized.replace(decimal_sep, '.')

    if normalized in ('', '-', '+'):
        raise ValueError(f'Invalid money amount: {value}')

    if negative and not normalized.startswith('-'):
        normalized = '-' + normalized

    # Only allow digits, optional leading -, and optional decimal point
    if not re.fullmatch(r'-?\d+(?:\.\d+)?', normalized):
        raise ValueError(f'Invalid money amount: {value} (normalized: {normalized})')

    try:
        return Decimal(normalized)
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
            warnings.append(f'{label} Heeft een ongeldig formaat: {value}')

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
            warnings.append(f'Som van totaalposten ({post_sum}) komt niet overeen met totaal excl. BTW ({total_exc})')

    if total_exc is not None and total_inc is not None:
        expected_inc = (total_exc * Decimal('1.21')).quantize(Decimal('0.01'))
        if abs(expected_inc - total_inc) > Decimal('0.02'):
            warnings.append(f'Totaal incl. BTW ({total_inc}) komt niet overeen met 21% BTW excl. over totaal ({expected_inc})')

    return warnings


def normalize_amounts(offer_json: dict) -> dict:
    """Normalize all monetary amounts to standard decimal format (e.g., "1234.56").
    
    Converts all amount strings to Decimal, then stores as string in standard format.
    
    Args:
        offer_json: The extracted offer dictionary
    
    Returns:
        dict: Offer with normalized amounts
    """
    # Top-level amount fields
    for key in ('Totaalprijs inc. BTW', 'Totaalprijs exc. BTW', 'Totaalbedrag'):
        if key in offer_json and offer_json[key] is not None:
            try:
                decimal_value = parse_money_value(offer_json[key])
                if decimal_value is not None:
                    offer_json[key] = str(decimal_value)
            except ValueError:
                pass  # Keep original if parsing fails

    # Normalize amounts in Posten array
    posten = offer_json.get('Posten', [])
    if isinstance(posten, list):
        for post in posten:
            for key in ('Eenheidsprijs', 'Totaalbedrag', 'Aantal'):
                if key in post and post[key] is not None:
                    try:
                        decimal_value = parse_money_value(post[key])
                        if decimal_value is not None:
                            post[key] = str(decimal_value)
                    except ValueError:
                        pass  # Keep original if parsing fails

    return offer_json


def extract_offer(file: Path, folder_handler):
    """Extract offer from PDF and save result via FolderHandler.
    
    Args:
        file: Path to the PDF file to extract
        folder_handler: FolderHandler instance for saving results
    
    Returns:
        dict: The extracted offer JSON data
    """
    from services.folder_handler import FolderHandler
    
    if not isinstance(folder_handler, FolderHandler):
        raise TypeError(f'folder_handler must be FolderHandler, got {type(folder_handler)}')

    prompt = read_txt(Path("./prompts/extract_prompt.txt"))
    offer = read_pdf(file)

    # Save raw PDF text for debugging
    folder_handler.save_raw_pdf_text(file, offer)

    answer = ask_llm('\n'.join([prompt, offer]))
    offer_json = parse_json_response(answer)
    validate_offer_json(offer_json)

    # Normalize all amounts to standard decimal format
    offer_json = normalize_amounts(offer_json)

    # Save result via FolderHandler
    folder_handler.save_result(file, offer_json)

    return offer_json


def compare_files(files, folder_handler):
    """Extract offers from multiple files.
    
    Args:
        files: List of PDF file paths to extract
        folder_handler: FolderHandler instance for saving results
    
    Returns:
        list: List of extracted offer dicts
    """
    return [extract_offer(file, folder_handler) for file in files]

