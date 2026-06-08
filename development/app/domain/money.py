from decimal import Decimal, InvalidOperation


def parse_decimal(value: str | int | float | Decimal | None) -> Decimal | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() == 'ONBEKEND':
        return None

    # Only include digits or decimal helpers.
    cleaned = ''.join(character for character in text if character.isdigit() or character in ',.-')
    if not cleaned:
        return None

    if ',' in cleaned and '.' in cleaned:
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    elif cleaned.count('.') > 0:
        parts = cleaned.split('.')
        cleaned = ''.join(parts[:-1]) + f"{'.' if len(parts[-1]) <= 2 else ''}"
        cleaned += parts[-1]

    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def calculate_total(amount: str, unit_price: str, fallback_total: str | None = None) -> str:
    amount_value = parse_decimal(amount)
    unit_price_value = parse_decimal(unit_price)
    if amount_value is None or unit_price_value is None:
        return fallback_total or 'ONBEKEND'

    return f'{amount_value * unit_price_value:.2f}'


def first_known_value(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        if text and text.upper() != 'ONBEKEND':
            return text

    return 'ONBEKEND'


def format_money(value: Decimal | None) -> str:
    if value is None:
        return 'ONBEKEND'

    rounded = value.quantize(Decimal('0.01'))
    us_format = f'{rounded:,.2f}'
    return '€' + us_format.replace(',', '\x00').replace('.', ',').replace('\x00', '.')
