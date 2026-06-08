from matching.match_fields import matched_post_descriptions


def normalize_text(value: str | None) -> str:
    return ' '.join(str(value or '').casefold().split())


def find_extracted_post_by_description(offer_result: dict, description: str | None) -> dict:
    if not description or str(description).strip().upper() == 'ONBEKEND':
        return {}

    normalized_description = normalize_text(description)
    for post in offer_result.get('Posten', []):
        if normalize_text(post.get('Omschrijving')) == normalized_description:
            return post

    return {}


def find_offer_result(offer_results: list[dict], offer_name: str) -> dict:
    for offer_result in offer_results:
        if offer_result.get('Bestand') == offer_name:
            return offer_result

    return {}


def find_extracted_posts_for_match(offer_result: dict, offer_match: dict) -> list[dict]:
    extracted_posts = []
    for description in matched_post_descriptions(offer_match):
        extracted_post = find_extracted_post_by_description(offer_result, description)
        if extracted_post:
            extracted_posts.append(extracted_post)

    return extracted_posts


def find_extracted_offer_post(offer_results: list[dict], offer_name: str, offer_match: dict) -> dict:
    offer_result = find_offer_result(offer_results, offer_name)
    extracted_posts = find_extracted_posts_for_match(offer_result, offer_match)
    return extracted_posts[0] if extracted_posts else {}


def find_matching_raw_row(raw_rows: list[dict], comparison_row: dict, index: int) -> dict:
    if index < len(raw_rows):
        return raw_rows[index]

    description = comparison_row.get('Omschrijving')
    for raw_row in raw_rows:
        if raw_row.get('Omschrijving') == description or raw_row.get('Vergelijking omschrijving') == description:
            return raw_row

    return {}


def find_flat_match(flat_rows: list[dict], comparison_row: dict, offer_name: str) -> dict:
    description = comparison_row.get('Omschrijving')
    for raw_row in flat_rows:
        if raw_row.get('Offerte') != offer_name:
            continue

        raw_description = raw_row.get('Vergelijking omschrijving') or raw_row.get('Omschrijving')
        if raw_description == description:
            return raw_row

    return {}
