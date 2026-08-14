from domain.money import parse_decimal, UNKNOWN


CANONICAL_UNITS = ('m2', 'm1', 'st', 'dzd', 'post')

# Keys are the result of normalize_unit() (casefolded, no spaces/dots, ² -> 2, ¹ -> 1).
_UNIT_ALIASES = {
    'm2': 'm2',
    'vierkantemeter': 'm2',
    'm1': 'm1',
    'm': 'm1',
    'meter': 'm1',
    'strekkendemeter': 'm1',
    'lengtemeter': 'm1',
    'lm': 'm1',
    'st': 'st',
    'stuk': 'st',
    'stuks': 'st',
    'stk': 'st',
    'stuk(s)': 'st',
    'dzd': 'dzd',
    'dznd': 'dzd',
    'duizend': 'dzd',
    'post': 'post',
    'stelpost': 'post',
    'pm': 'post',
    'promemori': 'post',
    'stel': 'post',
}


def canonicalize_unit(value: str | None) -> tuple[str, bool]:
    """Map a free-text unit to the canonical extraction vocabulary {m2, m1, st, dzd, post}.

    Returns (value_to_use, recognized). An empty/unknown input is left
    untouched and counted as recognized (nothing to flag). A non-empty,
    unrecognized unit is returned unchanged so callers can still show/store
    it, with recognized=False so it can be flagged for review.
    """
    original = str(value or '').strip()
    if not original or original.upper() == UNKNOWN:
        return original, True

    canonical = _UNIT_ALIASES.get(normalize_unit(original))
    if canonical:
        return canonical, True

    return original, False


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
