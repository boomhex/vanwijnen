import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
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


MODEL_ID = os.environ.get('GEMINI_MODEL', 'gemini-3-flash-preview')


OFFER_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'Naam opdrachtgever': {'type': 'string'},
        'Totaalprijs inc. BTW': {'type': 'string'},
        'Totaalprijs exc. BTW': {'type': 'string'},
        'Posten': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'Omschrijving': {'type': 'string'},
                    'Categorie': {'type': 'string'},
                    'Totaalbedrag': {'type': 'string'},
                    'Eenheid': {'type': 'string'},
                    'Eenheidsprijs': {'type': 'string'},
                    'Aantal': {'type': 'string'},
                },
                'required': [
                    'Omschrijving',
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
        'Naam opdrachtgever',
        'Totaalprijs inc. BTW',
        'Totaalprijs exc. BTW',
        'Posten',
    ],
}


OFFER_SUMMARY_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'Naam opdrachtgever': {'type': 'string'},
        'Totaalprijs inc. BTW': {'type': 'string'},
        'Totaalprijs exc. BTW': {'type': 'string'},
    },
    'required': [
        'Naam opdrachtgever',
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
) -> str:
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug("Generating LLM response")
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

            return response.text
        except KeyboardInterrupt:
            raise
        except Exception as error:
            error_text = str(error).strip()
            logger.warning("LLM request failed: %s", error_text)

            lowered = error_text.lower()
            if any(token in lowered for token in ('quota', 'resource exhausted')):
                raise RuntimeError(
                    'API limit reached. Try again later or increase your Gemini quota.'
                ) from error

            if attempt >= max_attempts:
                raise RuntimeError(f'LLM request failed after {max_attempts} attempts: {error_text}') from error

            time.sleep(min(2 ** (attempt - 1), 30))

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


def split_text_chunks(text: str, max_chars: int = POST_CHUNK_SIZE) -> list[str]:
    chunks = []
    current_lines = []
    current_size = 0

    for line in text.splitlines():
        line_size = len(line) + 1
        if current_lines and current_size + line_size > max_chars:
            chunks.append('\n'.join(current_lines))
            current_lines = []
            current_size = 0

        if line_size > max_chars:
            for start in range(0, len(line), max_chars):
                part = line[start:start + max_chars]
                if current_lines:
                    chunks.append('\n'.join(current_lines))
                    current_lines = []
                    current_size = 0
                chunks.append(part)
            continue

        current_lines.append(line)
        current_size += line_size

    if current_lines:
        chunks.append('\n'.join(current_lines))

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
    seen = set()

    for posts in post_chunks:
        for post in posts:
            if not isinstance(post, dict):
                continue

            identity = post_identity(post)
            if identity in seen:
                continue

            seen.add(identity)
            merged.append(post)

    return merged


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


def extract_offer_one_shot(
    prompt: str,
    offer_text: str,
    response_callback: ResponseCallback | None = None,
) -> tuple[dict, str]:
    answer = ask_llm('\n'.join([prompt, offer_text]), response_schema=OFFER_RESPONSE_SCHEMA)
    if response_callback is not None:
        response_callback('llm_response.txt', answer)
    return parse_json_response(answer), answer


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
        max_output_tokens=4096,
    )
    if response_callback is not None:
        response_callback('llm_summary_response.txt', summary_answer)
        response_callback('llm_response.txt', format_chunked_llm_response(summary_answer, []))
    summary_json = parse_json_response(summary_answer)

    post_answers = []
    post_chunks = []
    chunks = split_text_chunks(offer_text)
    for index, chunk in enumerate(chunks, start=1):
        if status_callback is not None:
            status_callback(
                f'extracting_posts_chunk_{index}_of_{len(chunks)}',
                f'Extracting posts chunk {index} of {len(chunks)}',
            )

        chunk_answer = ask_llm(
            '\n'.join([posts_prompt, f'Deel {index} van {len(chunks)}:', chunk]),
            response_schema=OFFER_POSTS_RESPONSE_SCHEMA,
            max_output_tokens=16384,
        )
        post_answers.append(chunk_answer)
        if response_callback is not None:
            response_callback(f'llm_posts_chunk_{index}_response.txt', chunk_answer)
            response_callback('llm_response.txt', format_chunked_llm_response(summary_answer, post_answers))
        chunk_json = parse_json_response(chunk_answer)
        posts = chunk_json.get('Posten', [])
        post_chunks.append(posts if isinstance(posts, list) else [])

    offer_json = {
        'Naam opdrachtgever': summary_json.get('Naam opdrachtgever', UNKNOWN),
        'Totaalprijs inc. BTW': summary_json.get('Totaalprijs inc. BTW', UNKNOWN),
        'Totaalprijs exc. BTW': summary_json.get('Totaalprijs exc. BTW', UNKNOWN),
        'Posten': merge_post_chunks(post_chunks),
    }
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

        set_status('saving_raw_text', 'Saving raw PDF text')
        folder_handler.save_raw_pdf_text(file, offer)

        if len(offer) > CHUNKED_EXTRACTION_THRESHOLD:
            set_status('extracting_chunked', 'Extracting long PDF in chunks')
            offer_json, answer = extract_offer_chunked(offer, set_status, save_response)
        else:
            set_status('calling_llm', 'Calling LLM')
            offer_json, answer = extract_offer_one_shot(prompt, offer, save_response)

        set_status('saving_llm_response', 'Saving raw LLM response')
        save_response('llm_response.txt', answer)

        set_status('validating_json', 'Validating extracted JSON')
        validate_offer_json(offer_json)

        set_status('normalizing_amounts', 'Normalizing amounts')
        offer_json = normalize_amounts(offer_json)

        set_status('saving_extract', 'Saving extract.json')
        folder_handler.save_result(file, offer_json)

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
