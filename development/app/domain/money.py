import re
from decimal import Decimal, InvalidOperation

UNKNOWN = 'ONBEKEND'


def parse_money_value(value: str | int | float | Decimal | None) -> Decimal | None:
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

    Returns None for empty/unknown input. Raises ValueError for a non-empty
    value that cannot be parsed as a number, so callers that need to
    distinguish "unknown" from "malformed" (e.g. validation warnings) can do
    so; use parse_decimal() when you just want a best-effort Decimal or None.
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() == UNKNOWN:
        return None

    s = text

    # Normalize unicode minus
    s = s.replace('−', '-')

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


def parse_decimal(value: str | int | float | Decimal | None) -> Decimal | None:
    """Best-effort money parsing: returns None instead of raising on malformed input."""
    try:
        return parse_money_value(value)
    except ValueError:
        return None


def calculate_total(amount: str, unit_price: str, fallback_total: str | None = None) -> str:
    amount_value = parse_decimal(amount)
    unit_price_value = parse_decimal(unit_price)
    if amount_value is None or unit_price_value is None:
        return fallback_total or UNKNOWN

    return f'{amount_value * unit_price_value:.2f}'


def calculate_unit_price(total: str | None, amount: str | None, fallback_unit_price: str | None = None) -> str:
    unit_price_value = parse_decimal(fallback_unit_price)
    if unit_price_value is not None:
        return str(fallback_unit_price)

    total_value = parse_decimal(total)
    amount_value = parse_decimal(amount)
    if total_value is None or amount_value is None or amount_value == 0:
        return fallback_unit_price or UNKNOWN

    return f'{total_value / amount_value:.2f}'


def first_known_value(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        if text and text.upper() != UNKNOWN:
            return text

    return UNKNOWN


def format_money(value: Decimal | None) -> str:
    if value is None:
        return UNKNOWN

    rounded = value.quantize(Decimal('0.01'))
    us_format = f'{rounded:,.2f}'
    return us_format.replace(',', '\x00').replace('.', ',').replace('\x00', '.')
