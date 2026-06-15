from domain.money import parse_decimal, UNKNOWN


def normalize_unit(value: str | None) -> str:
    text = str(value or '').strip().casefold()
    if not text or text == UNKNOWN.casefold():
        return ''

    return (
        text
        .replace('²', '2')
        .replace('¹', '1')
        .replace(' ', '')
        .replace('.', '')
    )


def units_mismatch(comparison_unit: str | None, matched_unit: str | None) -> bool:
    comparison = normalize_unit(comparison_unit)
    matched = normalize_unit(matched_unit)
    return bool(comparison and matched and comparison != matched)


def post_total_covers_mismatch(comparison_unit: str | None, matched_unit: str | None, offer: dict) -> bool:
    comparison = normalize_unit(comparison_unit)
    matched = normalize_unit(matched_unit)
    if 'post' not in {comparison, matched}:
        return False

    return parse_decimal(offer.get('Totaalbedrag')) is not None
