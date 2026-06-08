from decimal import Decimal

from domain.money import first_known_value, parse_decimal


def extracted_categories(extracted_posts: list[dict]) -> list[str]:
    categories = []
    for post in extracted_posts:
        category = str(post.get('Categorie') or '').strip()
        if category and category.upper() != 'ONBEKEND' and category not in categories:
            categories.append(category)

    return categories


def format_categories(categories: list[str]) -> str:
    if not categories:
        return 'ONBEKEND'
    return ', '.join(categories)


def matched_categories(offer_match: dict) -> list[str]:
    raw_categories = offer_match.get('Gematchte categorieen') or offer_match.get('Gematchte categorieën') or []
    if isinstance(raw_categories, list):
        return [
            str(category).strip()
            for category in raw_categories
            if str(category or '').strip() and str(category).strip().upper() != 'ONBEKEND'
        ]

    category = offer_match.get('Gematchte categorie') or offer_match.get('Categorie')
    if not category or str(category).strip().upper() == 'ONBEKEND':
        return []
    return [str(category).strip()]


def matched_post_descriptions(offer_match: dict) -> list[str]:
    raw_posts = offer_match.get('Gematchte posten') or offer_match.get('Gematchte Posten') or []
    if isinstance(raw_posts, list):
        descriptions = [
            str(description).strip()
            for description in raw_posts
            if str(description or '').strip() and str(description).strip().upper() != 'ONBEKEND'
        ]
        if descriptions:
            return descriptions

    description = offer_match.get('Gematchte omschrijving') or offer_match.get('Omschrijving')
    if not description or str(description).strip().upper() == 'ONBEKEND':
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

    return f'{total:.2f}' if has_total else 'ONBEKEND'


def offer_info_from_extracted_posts(extracted_posts: list[dict], offer_match: dict) -> dict:
    if len(extracted_posts) == 1:
        extracted_post = extracted_posts[0]
        return {
            'Match type': 'single',
            'Gematchte omschrijving': extracted_post.get('Omschrijving', 'ONBEKEND'),
            'Gematchte categorie': extracted_post.get('Categorie', 'ONBEKEND'),
            'Gematchte eenheid': extracted_post.get('Eenheid', 'ONBEKEND'),
            'Eenheidsprijs': extracted_post.get('Eenheidsprijs', 'ONBEKEND'),
            'Totaalbedrag': extracted_post.get('Totaalbedrag', 'ONBEKEND'),
            'Overeenkomst': offer_match.get('Overeenkomst', ''),
        }

    total = sum_extracted_totals(extracted_posts)
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
        'Eenheidsprijs': 'ONBEKEND',
        'Totaalbedrag': total,
        'Overeenkomst': offer_match.get('Overeenkomst', ''),
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
        'Eenheidsprijs': first_known_value(offer_match.get('Eenheidsprijs')),
        'Totaalbedrag': first_known_value(offer_match.get('Totaalbedrag')),
        'Overeenkomst': offer_match.get('Overeenkomst', ''),
    }
    if matched_posts:
        info['Gematchte posten'] = matched_posts
    categories = matched_categories(offer_match)
    if categories:
        info['Gematchte categorieen'] = categories
    return info
