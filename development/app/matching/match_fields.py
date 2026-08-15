from decimal import Decimal

from domain.money import calculate_unit_price, first_known_value, parse_decimal, UNKNOWN
from domain.units import normalize_unit


def extracted_categories(extracted_posts: list[dict]) -> list[str]:
    categories = []
    for post in extracted_posts:
        category = str(post.get('Categorie') or '').strip()
        if category and category.upper() != UNKNOWN and category not in categories:
            categories.append(category)

    return categories


def format_categories(categories: list[str]) -> str:
    if not categories:
        return UNKNOWN
    return ', '.join(categories)


def matched_categories(offer_match: dict) -> list[str]:
    raw_categories = offer_match.get('Gematchte categorieen') or offer_match.get('Gematchte categorieën') or []
    if isinstance(raw_categories, list):
        return [
            str(category).strip()
            for category in raw_categories
            if str(category or '').strip() and str(category).strip().upper() != UNKNOWN
        ]

    category = offer_match.get('Gematchte categorie') or offer_match.get('Categorie')
    if not category or str(category).strip().upper() == UNKNOWN:
        return []
    return [str(category).strip()]


def matched_post_descriptions(offer_match: dict) -> list[str]:
    raw_posts = offer_match.get('Gematchte posten') or offer_match.get('Gematchte Posten') or []
    if isinstance(raw_posts, list):
        descriptions = [
            str(description).strip()
            for description in raw_posts
            if str(description or '').strip() and str(description).strip().upper() != UNKNOWN
        ]
        if descriptions:
            return descriptions

    description = offer_match.get('Gematchte omschrijving') or offer_match.get('Omschrijving')
    if not description or str(description).strip().upper() == UNKNOWN:
        return []
    return [str(description).strip()]


def match_type_for_offer(offer_match: dict) -> str:
    match_type = str(offer_match.get('Match type') or offer_match.get('Match Type') or '').strip().casefold()
    if match_type == 'group':
        return 'group'
    if len(matched_post_descriptions(offer_match)) > 1:
        return 'group'
    return 'single'


def sum_extracted_totals(extracted_posts: list[dict]) -> str:
    total = Decimal('0')
    has_total = False
    for post in extracted_posts:
        amount = parse_decimal(post.get('Totaalbedrag'))
        if amount is None:
            continue

        total += amount
        has_total = True

    return f'{total:.2f}' if has_total else UNKNOWN


def aggregate_extracted_amount(extracted_posts: list[dict]) -> tuple[str, str]:
    total = Decimal('0')
    unit = ''
    has_amount = False

    for post in extracted_posts:
        amount = parse_decimal(post.get('Aantal'))
        if amount is None:
            return UNKNOWN, UNKNOWN

        post_unit = str(post.get('Eenheid') or '').strip()
        if not post_unit or post_unit.upper() == UNKNOWN:
            return UNKNOWN, UNKNOWN

        if unit and normalize_unit(post_unit) != normalize_unit(unit):
            return UNKNOWN, UNKNOWN

        unit = post_unit
        total += amount
        has_amount = True

    if not has_amount:
        return UNKNOWN, UNKNOWN

    return f'{total:.2f}', unit


def offer_info_from_extracted_posts(extracted_posts: list[dict], offer_match: dict) -> dict:
    if len(extracted_posts) == 1:
        extracted_post = extracted_posts[0]
        unit_price = calculate_unit_price(
            extracted_post.get('Totaalbedrag'),
            extracted_post.get('Aantal'),
            extracted_post.get('Eenheidsprijs'),
        )
        return {
            'Match type': 'single',
            'Gematchte omschrijving': extracted_post.get('Omschrijving', UNKNOWN),
            'Gematchte categorie': extracted_post.get('Categorie', UNKNOWN),
            'Gematchte eenheid': extracted_post.get('Eenheid', UNKNOWN),
            'Aantal': extracted_post.get('Aantal', UNKNOWN),
            'Eenheidsprijs': unit_price,
            'Totaalbedrag': extracted_post.get('Totaalbedrag', UNKNOWN),
            'Overeenkomst': offer_match.get('Overeenkomst', ''),
            'Ongekoppeld': False,
        }

    total = sum_extracted_totals(extracted_posts)
    amount, amount_unit = aggregate_extracted_amount(extracted_posts)
    descriptions = [
        post.get('Omschrijving', '')
        for post in extracted_posts
        if post.get('Omschrijving')
    ]
    categories = extracted_categories(extracted_posts)
    return {
        'Match type': 'group',
        'Gematchte omschrijving': f'{len(descriptions)} posten',
        'Gematchte posten': descriptions,
        'Gematchte categorie': format_categories(categories),
        'Gematchte categorieen': categories,
        'Gematchte eenheid': 'post',
        'Aantal': amount,
        'Gematchte hoeveelheid eenheid': amount_unit,
        'Eenheidsprijs': UNKNOWN,
        'Totaalbedrag': total,
        'Overeenkomst': offer_match.get('Overeenkomst', ''),
        'Ongekoppeld': False,
    }


def offer_info_from_match(offer_match: dict) -> dict:
    matched_posts = matched_post_descriptions(offer_match)
    info = {
        'Match type': match_type_for_offer(offer_match),
        'Gematchte omschrijving': first_known_value(
            offer_match.get('Gematchte omschrijving'),
            offer_match.get('Omschrijving'),
        ),
        'Gematchte categorie': first_known_value(offer_match.get('Gematchte categorie'), offer_match.get('Categorie')),
        'Gematchte eenheid': first_known_value(offer_match.get('Gematchte eenheid'), offer_match.get('Eenheid')),
        'Aantal': first_known_value(offer_match.get('Aantal')),
        'Eenheidsprijs': calculate_unit_price(
            offer_match.get('Totaalbedrag'),
            offer_match.get('Aantal'),
            offer_match.get('Eenheidsprijs'),
        ),
        'Totaalbedrag': first_known_value(offer_match.get('Totaalbedrag')),
        'Overeenkomst': offer_match.get('Overeenkomst', ''),
    }
    if matched_posts:
        info['Gematchte posten'] = matched_posts
    categories = matched_categories(offer_match)
    if categories:
        info['Gematchte categorieen'] = categories
    return info
