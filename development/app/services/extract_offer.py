import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from difflib import SequenceMatcher
from json import JSONDecodeError
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation
import time

import os
from google import genai
from google.genai import types
import pdfplumber

from domain.money import UNKNOWN

logger = logging.getLogger(__name__)


MODEL_ID = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-lite')
EXTRACTION_MODEL_ID = os.environ.get('EXTRACT_OFFER_MODEL', 'gemini-3.5-flash')
EXTRACTION_MODE = os.environ.get('EXTRACT_MODE', 'auto').strip().casefold()
EXTRACTION_MAX_OUTPUT_TOKENS = int(os.environ.get('EXTRACT_MAX_OUTPUT_TOKENS', '65536'))
POST_CHUNK_MAX_OUTPUT_TOKENS = int(os.environ.get('EXTRACT_POST_CHUNK_MAX_OUTPUT_TOKENS', '16384'))
SUMMARY_MAX_OUTPUT_TOKENS = int(os.environ.get('EXTRACT_SUMMARY_MAX_OUTPUT_TOKENS', '4096'))


OFFER_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'Naam aannemer': {'type': 'string'},
        'Totaalprijs inc. BTW': {'type': 'string'},
        'Totaalprijs exc. BTW': {'type': 'string'},
        'Posten': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'Omschrijving': {'type': 'string'},
                    'Beschrijving': {'type': 'string'},
                    'Categorie': {'type': 'string'},
                    'Totaalbedrag': {'type': 'string'},
                    'Eenheid': {'type': 'string'},
                    'Eenheidsprijs': {'type': 'string'},
                    'Aantal': {'type': 'string'},
                },
                'required': [
                    'Omschrijving',
                    'Beschrijving',
                    'Categorie',
                    'Totaalbedrag',
                    'Eenheid',
                    'Eenheidsprijs',
                    'Aantal',
                ],
            },
        },
    },
    'required': [
        'Naam aannemer',
        'Totaalprijs inc. BTW',
        'Totaalprijs exc. BTW',
        'Posten',
    ],
}


OFFER_SUMMARY_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'Naam aannemer': {'type': 'string'},
        'Totaalprijs inc. BTW': {'type': 'string'},
        'Totaalprijs exc. BTW': {'type': 'string'},
    },
    'required': [
        'Naam aannemer',
        'Totaalprijs inc. BTW',
        'Totaalprijs exc. BTW',
    ],
}


OFFER_POSTS_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'Posten': OFFER_RESPONSE_SCHEMA['properties']['Posten'],
    },
    'required': ['Posten'],
}


CHUNKED_EXTRACTION_THRESHOLD = int(os.environ.get('EXTRACT_CHUNKED_THRESHOLD_CHARS', '7000'))
POST_CHUNK_SIZE = int(os.environ.get('EXTRACT_POST_CHUNK_CHARS', '4500'))
POST_CHUNK_OVERLAP_LINES = int(os.environ.get('EXTRACT_CHUNK_OVERLAP_LINES', '40'))
VALID_EXTRACTION_MODES = {'auto', 'chunked', 'one_shot'}


client = None
StatusCallback = Callable[[str, str | None], None]
ResponseCallback = Callable[[str, str], None]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_extraction_status(
    folder_handler,
    file: Path,
    *,
    status: str,
    step: str,
    message: str | None = None,
    error: str | None = None,
    started_at: str | None = None,
) -> None:
    payload = {
        'status': status,
        'step': step,
        'updated_at': utc_now_iso(),
    }
    if started_at is not None:
        payload['started_at'] = started_at
    if message:
        payload['message'] = message
    if error:
        payload['error'] = error

    folder_handler.save_extraction_status(file, payload)
    logger.info(
        'Extraction status for %s: %s/%s%s',
        file,
        status,
        step,
        f' - {message}' if message else '',
    )


def get_client():
    global client
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY is not set')
        client = genai.Client(api_key=api_key)

    return client


def ask_llm(
    prompt: str,
    *,
    response_schema: dict | None = None,
    max_output_tokens: int = 65536,
    model: str = MODEL_ID,
    label: str = 'llm_call',
) -> str:
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                'LLM call started: label=%s model=%s attempt=%s/%s prompt_chars=%s max_output_tokens=%s schema=%s',
                label,
                model,
                attempt,
                max_attempts,
                len(prompt),
                max_output_tokens,
                response_schema is not None,
            )
            config_kwargs = {
                'temperature': 0.0,
                'max_output_tokens': max_output_tokens,
                'response_mime_type': 'application/json',
            }
            if response_schema is not None:
                config_kwargs['response_schema'] = response_schema

            response = get_client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            if response.text is None:
                raise RuntimeError('LLM response did not contain text')

            logger.info(
                'LLM response received: label=%s attempt=%s response_chars=%s',
                label,
                attempt,
                len(response.text),
            )
            return response.text
        except KeyboardInterrupt:
            raise
        except Exception as error:
            error_text = str(error).strip()
            logger.warning(
                'LLM request failed: label=%s attempt=%s/%s error=%s',
                label,
                attempt,
                max_attempts,
                error_text,
            )

            lowered = error_text.lower()
            if any(token in lowered for token in ('quota', 'resource exhausted')):
                raise RuntimeError(
                    'API limit reached. Try again later or increase your Gemini quota.'
                ) from error

            if attempt >= max_attempts:
                raise RuntimeError(f'LLM request failed after {max_attempts} attempts: {error_text}') from error

            sleep_seconds = min(2 ** (attempt - 1), 30)
            logger.info('LLM retry scheduled: label=%s sleep_seconds=%s', label, sleep_seconds)
            time.sleep(sleep_seconds)

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


def split_text_chunks(
    text: str,
    max_chars: int = POST_CHUNK_SIZE,
    overlap_lines: int = POST_CHUNK_OVERLAP_LINES,
) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    while start < len(lines):
        chunk_lines = []
        chunk_size = 0
        index = start

        while index < len(lines):
            line = lines[index]
            line_size = len(line) + 1
            if chunk_lines and chunk_size + line_size > max_chars:
                break

            if not chunk_lines and line_size > max_chars:
                chunk_lines.append(line[:max_chars])
                index += 1
                break

            chunk_lines.append(line)
            chunk_size += line_size
            index += 1

        if not chunk_lines:
            break

        chunks.append('\n'.join(chunk_lines))
        if index >= len(lines):
            break

        next_start = max(index - overlap_lines, start + 1)
        start = next_start

    return [chunk for chunk in chunks if chunk.strip()]


def post_identity(post: dict) -> tuple[str, str, str, str, str, str]:
    return (
        str(post.get('Omschrijving', '')).strip().casefold(),
        str(post.get('Categorie', '')).strip().casefold(),
        str(post.get('Aantal', '')).strip(),
        str(post.get('Eenheid', '')).strip().casefold(),
        str(post.get('Eenheidsprijs', '')).strip(),
        str(post.get('Totaalbedrag', '')).strip(),
    )


def merge_post_chunks(post_chunks: list[list[dict]]) -> list[dict]:
    merged = []

    for posts in post_chunks:
        for post in posts:
            if not isinstance(post, dict):
                continue

            duplicate_index = find_duplicate_post_index(merged, post)
            if duplicate_index is not None:
                merged[duplicate_index] = merge_duplicate_posts(merged[duplicate_index], post)
                continue

            merged.append(post)

    return merged


def find_duplicate_post_index(posts: list[dict], candidate: dict) -> int | None:
    for index, post in enumerate(posts):
        if are_duplicate_posts(post, candidate):
            return index

    return None


def are_duplicate_posts(first: dict, second: dict) -> bool:
    if post_identity(first) == post_identity(second):
        return True

    first_code = post_code(first)
    second_code = post_code(second)
    if first_code and first_code == second_code:
        return True

    if not descriptions_similar(first.get('Omschrijving'), second.get('Omschrijving')):
        return False

    if not compatible_text_values(first.get('Categorie'), second.get('Categorie')):
        return False

    descriptions_match = normalized_post_text(first.get('Omschrijving')) == normalized_post_text(second.get('Omschrijving'))
    totals_match = exact_known_match(first, second, 'Totaalbedrag')
    amounts_match = exact_known_match(first, second, 'Aantal') and compatible_text_values(first.get('Eenheid'), second.get('Eenheid'))
    unit_prices_match = exact_known_match(first, second, 'Eenheidsprijs') and compatible_text_values(first.get('Eenheid'), second.get('Eenheid'))

    if descriptions_match:
        return totals_match or amounts_match or unit_prices_match

    return (totals_match and amounts_match) or (totals_match and unit_prices_match) or (amounts_match and unit_prices_match)


def merge_duplicate_posts(first: dict, second: dict) -> dict:
    primary, fallback = (second, first) if post_completeness_score(second) > post_completeness_score(first) else (first, second)
    merged = dict(primary)

    for field in ('Omschrijving', 'Beschrijving', 'Categorie', 'Totaalbedrag', 'Eenheid', 'Eenheidsprijs', 'Aantal'):
        if not has_known_value(merged.get(field)) and has_known_value(fallback.get(field)):
            merged[field] = fallback[field]

    first_description = str(first.get('Omschrijving', '') or '')
    second_description = str(second.get('Omschrijving', '') or '')
    if normalized_post_text(first_description) in normalized_post_text(second_description):
        merged['Omschrijving'] = second_description
    elif normalized_post_text(second_description) in normalized_post_text(first_description):
        merged['Omschrijving'] = first_description

    return merged


def post_completeness_score(post: dict) -> int:
    score = 0
    for field in ('Omschrijving', 'Beschrijving', 'Categorie', 'Totaalbedrag', 'Eenheid', 'Eenheidsprijs', 'Aantal'):
        if has_known_value(post.get(field)):
            score += 100

    score += min(len(str(post.get('Omschrijving', '') or '')), 200)
    return score


def post_code(post: dict) -> str | None:
    match = re.match(r'\s*(\d{4,6})\b', str(post.get('Omschrijving', '') or ''))
    return match.group(1) if match else None


def descriptions_similar(first, second) -> bool:
    first_text = normalized_post_text(first)
    second_text = normalized_post_text(second)
    if not first_text or not second_text:
        return False
    if first_text == second_text:
        return True
    if min(len(first_text), len(second_text)) >= 24 and (first_text in second_text or second_text in first_text):
        return True

    ratio = SequenceMatcher(None, first_text, second_text).ratio()
    if ratio >= 0.86:
        return True

    first_tokens = set(first_text.split())
    second_tokens = set(second_text.split())
    shared_tokens = first_tokens & second_tokens
    if len(shared_tokens) < 4:
        return False

    return len(shared_tokens) / min(len(first_tokens), len(second_tokens)) >= 0.75


def normalized_post_text(value) -> str:
    text = str(value or '').casefold()
    text = re.sub(r'^\s*\d{4,6}\s+', '', text)
    text = re.sub(r'[^0-9a-z]+', ' ', text)
    return ' '.join(text.split())


def compatible_text_values(first, second) -> bool:
    if not has_known_value(first) or not has_known_value(second):
        return True

    return normalized_post_text(first) == normalized_post_text(second)


def exact_known_match(first: dict, second: dict, field: str) -> bool:
    first_value = first.get(field)
    second_value = second.get(field)
    if not has_known_value(first_value) or not has_known_value(second_value):
        return False

    return normalize_field_value(first_value) == normalize_field_value(second_value)


def normalize_field_value(value) -> str:
    parsed_value = parse_money_value_safely(value)
    if parsed_value is not None:
        return str(parsed_value)

    return normalized_post_text(value)


def has_known_value(value) -> bool:
    text = str(value or '').strip()
    return bool(text) and text.upper() != UNKNOWN


EXCLUDED_CONTEXT_TERMS = (
    'algemene voorwaarden',
    'voorwaarden',
    'bepaling',
    'bepalingen',
    'uitgangspunt',
    'uitgangspunten',
    'garantie',
    'betaling',
    'betalingstermijn',
    'offerte geldig',
    'arbo',
    'krediet',
    'coface',
)

EXCLUDED_DESCRIPTION_TERMS = (
    'op aanvraag',
    'op regiebasis',
    'zal worden berekend',
    'zullen wij hier een toeslag voor berekenen',
    'wordt verrekend',
    'worden verrekend',
    'wordt berekend',
    'worden berekend',
    'geen garantie',
)


def filter_non_price_posts(posts: list[dict]) -> list[dict]:
    filtered_posts = []
    removed_count = 0
    for post in posts:
        if should_exclude_post(post):
            removed_count += 1
            logger.info('Filtered non-price/bepalingen post: %s', post.get('Omschrijving'))
            continue

        filtered_posts.append(post)

    if removed_count:
        logger.info('Filtered %s non-price/bepalingen posts', removed_count)

    return filtered_posts


def should_exclude_post(post: dict) -> bool:
    if has_concrete_price_data(post):
        return False

    description = normalize_filter_text(post.get('Omschrijving'))
    detailed_description = normalize_filter_text(post.get('Beschrijving'))
    category = normalize_filter_text(post.get('Categorie'))
    combined = f'{category} {description} {detailed_description}'

    return any(term in combined for term in EXCLUDED_CONTEXT_TERMS) or any(
        term in f'{description} {detailed_description}' for term in EXCLUDED_DESCRIPTION_TERMS
    )


def has_concrete_price_data(post: dict) -> bool:
    return any(
        parse_money_value_safely(post.get(field)) is not None
        for field in ('Totaalbedrag', 'Eenheidsprijs', 'Aantal')
    )


def parse_money_value_safely(value) -> Decimal | None:
    try:
        return parse_money_value(value)
    except ValueError:
        return None


def normalize_filter_text(value) -> str:
    return ' '.join(str(value or '').casefold().split())


def format_chunked_llm_response(summary_answer: str, post_answers: list[str]) -> str:
    sections = ['=== summary ===', summary_answer]
    for index, answer in enumerate(post_answers, start=1):
        sections.extend([f'=== posts chunk {index} ===', answer])
    return '\n\n'.join(sections)


def parse_json_response(answer: str) -> dict:
    cleaned_answer = answer.strip()
    markdown_json = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', cleaned_answer, re.DOTALL)
    if markdown_json:
        cleaned_answer = markdown_json.group(1).strip()

    try:
        return json.loads(cleaned_answer)
    except JSONDecodeError as error:
        raise ValueError(
            f'LLM response was not valid JSON at line {error.lineno}, column {error.colno}. '
            'The raw response was saved as llm_response.txt.'
        ) from error


def parse_posts_response(answer: str) -> tuple[list[dict], bool]:
    """Parse a posts chunk response.

    Returns (posts, recovered). If the JSON is truncated, recover complete
    objects from the Posten array and ignore the incomplete tail. This avoids an
    extra LLM retry and keeps already extracted rows.
    """
    try:
        parsed = parse_json_response(answer)
    except ValueError:
        recovered_posts = recover_complete_posts(answer)
        if recovered_posts:
            return recovered_posts, True
        raise

    posts = parsed.get('Posten', [])
    return (posts if isinstance(posts, list) else []), False


def parse_offer_response(answer: str) -> tuple[dict, bool]:
    try:
        return parse_json_response(answer), False
    except ValueError:
        recovered_posts = recover_complete_posts(answer)
        if not recovered_posts:
            raise

        return {
            'Naam aannemer': recover_json_string_field(answer, 'Naam aannemer') or UNKNOWN,
            'Totaalprijs inc. BTW': recover_json_string_field(answer, 'Totaalprijs inc. BTW') or UNKNOWN,
            'Totaalprijs exc. BTW': recover_json_string_field(answer, 'Totaalprijs exc. BTW') or UNKNOWN,
            'Posten': recovered_posts,
            'Extractie waarschuwingen': [
                'De one-shot LLM response had ongeldige of afgekorte JSON; complete posten zijn behouden en de incomplete staart is overgeslagen.',
            ],
        }, True


def recover_json_string_field(answer: str, field: str) -> str | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, answer)
    if not match:
        return None

    try:
        return json.loads(f'"{match.group(1)}"')
    except JSONDecodeError:
        return match.group(1)


def recover_complete_posts(answer: str) -> list[dict]:
    array_start = find_posts_array_start(answer)
    if array_start is None:
        return []

    decoder = json.JSONDecoder()
    posts = []
    index = array_start
    while index < len(answer):
        while index < len(answer) and answer[index] in ' \t\r\n,':
            index += 1

        if index >= len(answer) or answer[index] == ']':
            break
        if answer[index] != '{':
            index += 1
            continue

        try:
            value, next_index = decoder.raw_decode(answer, index)
        except JSONDecodeError:
            break

        if isinstance(value, dict):
            posts.append(value)
        index = next_index

    return posts


def find_posts_array_start(answer: str) -> int | None:
    match = re.search(r'"Posten"\s*:\s*\[', answer)
    if not match:
        return None
    return match.end()


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
    if not text or text.upper() == UNKNOWN:
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
    required_keys = ['Naam aannemer', 'Totaalprijs inc. BTW', 'Totaalprijs exc. BTW', 'Posten']

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


def should_use_chunked_extraction(offer_text: str) -> bool:
    if EXTRACTION_MODE not in VALID_EXTRACTION_MODES:
        raise ValueError(f'Invalid EXTRACT_MODE "{EXTRACTION_MODE}". Use one of: {", ".join(sorted(VALID_EXTRACTION_MODES))}')
    if EXTRACTION_MODE == 'chunked':
        return True
    if EXTRACTION_MODE == 'one_shot':
        return False

    return len(offer_text) > CHUNKED_EXTRACTION_THRESHOLD


def build_posts_chunk_prompt(
    posts_prompt: str,
    chunk: str,
    *,
    index: int,
    total_chunks: int,
    previous_post: dict | None = None,
) -> str:
    previous_post_text = (
        json.dumps(previous_post, ensure_ascii=False, indent=2)
        if isinstance(previous_post, dict)
        else 'GEEN'
    )
    return '\n'.join([
        posts_prompt,
        'VORIGE GEACCEPTEERDE POST:',
        previous_post_text,
        'INSTRUCTIE VOOR OVERLAP:',
        '- De chunking is technisch en is GEEN onderdeel van de offerte.',
        '- Gebruik labels zoals "Deel", "VORIGE GEACCEPTEERDE POST" of "INSTRUCTIE VOOR OVERLAP" nooit als omschrijving, categorie of postinformatie.',
        '- Gebruik alleen categorieën/kopjes die letterlijk in de offerte zelf staan.',
        '- Begin met extraheren NA de vorige geaccepteerde post.',
        '- Neem de vorige geaccepteerde post niet opnieuw op.',
        '- Als het begin van dit tekstdeel alleen een vervolg/toelichting is op de vorige post, gebruik dit als context maar output geen post daarvoor.',
        '- Output alleen nieuwe volledige posten uit dit tekstdeel.',
        f'Deel {index} van {total_chunks}:',
        chunk,
    ])


def extract_offer_one_shot(
    prompt: str,
    offer_text: str,
    response_callback: ResponseCallback | None = None,
) -> tuple[dict, str]:
    answer = ask_llm(
        '\n'.join([prompt, offer_text]),
        response_schema=OFFER_RESPONSE_SCHEMA,
        max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS,
        model=EXTRACTION_MODEL_ID,
        label='offer_one_shot',
    )
    if response_callback is not None:
        response_callback('llm_response.txt', answer)
    logger.info('Parsing one-shot extraction response')
    offer_json, recovered = parse_offer_response(answer)
    if recovered:
        logger.warning('Recovered complete posts from truncated one-shot extraction response')

    return offer_json, answer


def extract_offer_chunked(
    offer_text: str,
    status_callback: StatusCallback | None = None,
    response_callback: ResponseCallback | None = None,
) -> tuple[dict, str]:
    summary_prompt = read_txt(Path('./prompts/extract_summary_prompt.txt'))
    posts_prompt = read_txt(Path('./prompts/extract_posts_chunk_prompt.txt'))

    if status_callback is not None:
        status_callback('extracting_summary', 'Extracting offer summary')

    summary_answer = ask_llm(
        '\n'.join([summary_prompt, offer_text]),
        response_schema=OFFER_SUMMARY_RESPONSE_SCHEMA,
        max_output_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        model=EXTRACTION_MODEL_ID,
        label='offer_summary',
    )
    if response_callback is not None:
        response_callback('llm_summary_response.txt', summary_answer)
        response_callback('llm_response.txt', format_chunked_llm_response(summary_answer, []))
    logger.info('Parsing extraction summary response')
    summary_json = parse_json_response(summary_answer)

    post_answers = []
    post_chunks = []
    recovered_chunks = []
    previous_post = None
    chunks = split_text_chunks(offer_text)
    logger.info(
        'Chunked post extraction started: chunks=%s overlap_lines=%s chunk_chars=%s',
        len(chunks),
        POST_CHUNK_OVERLAP_LINES,
        [len(chunk) for chunk in chunks],
    )
    for index, chunk in enumerate(chunks, start=1):
        if status_callback is not None:
            status_callback(
                f'extracting_posts_chunk_{index}_of_{len(chunks)}',
                f'Extracting posts chunk {index} of {len(chunks)}',
            )

        logger.info(
            'Post chunk LLM call preparing: chunk=%s/%s chunk_chars=%s chunk_lines=%s',
            index,
            len(chunks),
            len(chunk),
            len(chunk.splitlines()),
        )
        chunk_answer = ask_llm(
            build_posts_chunk_prompt(
                posts_prompt,
                chunk,
                index=index,
                total_chunks=len(chunks),
                previous_post=previous_post,
            ),
            response_schema=OFFER_POSTS_RESPONSE_SCHEMA,
            max_output_tokens=POST_CHUNK_MAX_OUTPUT_TOKENS,
            model=EXTRACTION_MODEL_ID,
            label=f'offer_posts_chunk_{index}_of_{len(chunks)}',
        )
        post_answers.append(chunk_answer)
        if response_callback is not None:
            response_callback(f'llm_posts_chunk_{index}_response.txt', chunk_answer)
            response_callback('llm_response.txt', format_chunked_llm_response(summary_answer, post_answers))
        logger.info('Parsing posts chunk response: chunk=%s/%s', index, len(chunks))
        posts, recovered = parse_posts_response(chunk_answer)
        if recovered:
            recovered_chunks.append(index)
            logger.warning(
                'Recovered %s complete posts from truncated chunk %s',
                len(posts),
                index,
            )
        logger.info(
            'Posts chunk parsed: chunk=%s/%s posts=%s recovered=%s',
            index,
            len(chunks),
            len(posts),
            recovered,
        )
        post_chunks.append(posts)
        if posts:
            previous_post = posts[-1]

    offer_json = {
        'Naam aannemer': summary_json.get('Naam aannemer', UNKNOWN),
        'Totaalprijs inc. BTW': summary_json.get('Totaalprijs inc. BTW', UNKNOWN),
        'Totaalprijs exc. BTW': summary_json.get('Totaalprijs exc. BTW', UNKNOWN),
        'Posten': merge_post_chunks(post_chunks),
    }
    if recovered_chunks:
        offer_json['Extractie waarschuwingen'] = [
            'Een of meer chunks hadden ongeldige JSON; complete posten zijn behouden en de incomplete staart is overgeslagen.',
            f'Herstelde chunks: {", ".join(str(index) for index in recovered_chunks)}.',
        ]

    return offer_json, format_chunked_llm_response(summary_answer, post_answers)


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

    started_at = utc_now_iso()

    def set_status(step: str, message: str | None = None, *, status: str = 'running') -> None:
        update_extraction_status(
            folder_handler,
            file,
            status=status,
            step=step,
            message=message,
            started_at=started_at,
        )

    def save_response(name: str, answer: str) -> None:
        if name == 'llm_response.txt':
            folder_handler.save_llm_response(file, answer)
            return

        folder_handler.save_named_llm_response(file, name, answer)

    try:
        set_status('reading_pdf', 'Reading PDF text')
        prompt = read_txt(Path("./prompts/extract_prompt.txt"))
        offer = read_pdf(file)
        logger.info('PDF text read: file=%s chars=%s lines=%s', file, len(offer), len(offer.splitlines()))

        set_status('saving_raw_text', 'Saving raw PDF text')
        folder_handler.save_raw_pdf_text(file, offer)
        logger.info('Raw PDF text saved: file=%s', file)

        use_chunked = should_use_chunked_extraction(offer)
        if use_chunked:
            set_status('extracting_chunked', 'Extracting long PDF in chunks')
            logger.info(
                'Extraction mode selected: file=%s mode=chunked configured_mode=%s model=%s threshold_chars=%s',
                file,
                EXTRACTION_MODE,
                EXTRACTION_MODEL_ID,
                CHUNKED_EXTRACTION_THRESHOLD,
            )
            offer_json, answer = extract_offer_chunked(offer, set_status, save_response)
        else:
            set_status('calling_llm', 'Calling LLM')
            logger.info(
                'Extraction mode selected: file=%s mode=one_shot configured_mode=%s model=%s threshold_chars=%s',
                file,
                EXTRACTION_MODE,
                EXTRACTION_MODEL_ID,
                CHUNKED_EXTRACTION_THRESHOLD,
            )
            offer_json, answer = extract_offer_one_shot(prompt, offer, save_response)

        set_status('saving_llm_response', 'Saving raw LLM response')
        save_response('llm_response.txt', answer)
        logger.info('Combined LLM response saved: file=%s response_chars=%s', file, len(answer))

        set_status('validating_json', 'Validating extracted JSON')
        if isinstance(offer_json.get('Posten'), list):
            offer_json['Posten'] = filter_non_price_posts(offer_json['Posten'])

        validation_warnings = validate_offer_json(offer_json)
        logger.info('Offer JSON validated: file=%s warnings=%s', file, len(validation_warnings))
        for warning in validation_warnings:
            logger.warning('Offer validation warning: file=%s warning=%s', file, warning)

        set_status('normalizing_amounts', 'Normalizing amounts')
        offer_json = normalize_amounts(offer_json)

        set_status('saving_extract', 'Saving extract.json')
        folder_handler.save_result(file, offer_json)
        logger.info('Extract saved: file=%s posts=%s', file, len(offer_json.get('Posten', [])))

        update_extraction_status(
            folder_handler,
            file,
            status='done',
            step='done',
            message='Extraction completed',
            started_at=started_at,
        )

        return offer_json
    except Exception as error:
        logger.exception('Extraction failed: file=%s', file)
        update_extraction_status(
            folder_handler,
            file,
            status='failed',
            step='failed',
            message='Extraction failed',
            error=str(error),
            started_at=started_at,
        )
        raise


def compare_files(files, folder_handler):
    """Extract offers from multiple files.
    
    Args:
        files: List of PDF file paths to extract
        folder_handler: FolderHandler instance for saving results
    
    Returns:
        list: List of extracted offer dicts
    """
    return [extract_offer(file, folder_handler) for file in files]
