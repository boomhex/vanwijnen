from domain.money import UNKNOWN
from matching.match_calculation import calculate_offer_total
from matching.match_fields import offer_info_from_extracted_posts
from matching.match_lookup import (
    find_extracted_posts_for_match,
    find_offer_result,
    normalize_text,
)


def deduplicate_offer_matches(normalized_rows: list[dict], offer_results: list[dict]) -> None:
    """Remove offer posts that are claimed by multiple comparison rows.

    For each offer post that appears in more than one comparison row, the post
    is kept only in the best-matching row and removed from all others.

    Priority order (highest first):
    1. Single match over group match
    2. Highest Overeenkomst score
    3. Earliest row (lowest index)

    When a post is removed from a group match, the group total is recalculated
    from the remaining extracted posts. If no posts remain the match is cleared.
    """
    offer_names = {name for row in normalized_rows for name in row.get('Offertes', {})}
    for offer_name in offer_names:
        _deduplicate_for_offer(normalized_rows, offer_results, offer_name)


def _deduplicate_for_offer(
    normalized_rows: list[dict],
    offer_results: list[dict],
    offer_name: str,
) -> None:
    post_to_rows: dict[str, list[tuple[int, int, bool]]] = {}

    for row_index, row in enumerate(normalized_rows):
        offer = row.get('Offertes', {}).get(offer_name, {})
        score = _parse_overeenkomst(offer.get('Overeenkomst'))
        is_single = _is_single_match(offer)
        for key in _match_post_keys(offer):
            post_to_rows.setdefault(key, []).append((row_index, score, is_single))

    for post_key, candidates in post_to_rows.items():
        if len(candidates) <= 1:
            continue

        # Keep the best candidate; remove post from all others
        candidates.sort(key=lambda c: (not c[2], -c[1], c[0]))
        for row_index, _score, _is_single in candidates[1:]:
            _remove_post_from_match(normalized_rows[row_index], offer_name, post_key, offer_results)


def _match_post_keys(offer: dict) -> list[str]:
    """Return normalized description keys for each matched offer post."""
    unknown_key = normalize_text(UNKNOWN)
    if _is_single_match(offer):
        key = normalize_text(offer.get('Gematchte omschrijving') or '')
        return [key] if key and key != unknown_key else []

    return [
        normalize_text(p)
        for p in (offer.get('Gematchte posten') or [])
        if p and normalize_text(p) and normalize_text(p) != unknown_key
    ]


def _is_single_match(offer: dict) -> bool:
    return str(offer.get('Match type') or '').strip().casefold() != 'group'


def _parse_overeenkomst(value) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def _remove_post_from_match(
    row: dict,
    offer_name: str,
    post_key: str,
    offer_results: list[dict],
) -> None:
    offer = row.get('Offertes', {}).get(offer_name)
    if not isinstance(offer, dict):
        return

    if _is_single_match(offer):
        if normalize_text(offer.get('Gematchte omschrijving', '')) == post_key:
            _clear_offer_match(offer)
        return

    remaining = [
        p for p in (offer.get('Gematchte posten') or [])
        if normalize_text(p) != post_key
    ]
    if not remaining:
        _clear_offer_match(offer)
        return

    offer_result = find_offer_result(offer_results, offer_name)
    extracted_posts = find_extracted_posts_for_match(offer_result, {**offer, 'Gematchte posten': remaining})
    if extracted_posts:
        rebuilt = offer_info_from_extracted_posts(extracted_posts, offer)
        offer.clear()
        offer.update(rebuilt)
        offer['Totaalbedrag'] = calculate_offer_total(
            row.get('Aantal', ''),
            offer,
            row.get('Eenheid'),
        )
    else:
        offer['Gematchte posten'] = remaining
        offer['Gematchte omschrijving'] = f'{len(remaining)} posten'


def _clear_offer_match(offer: dict) -> None:
    offer.clear()
    offer.update({
        'Match type': 'single',
        'Gematchte omschrijving': UNKNOWN,
        'Gematchte categorie': UNKNOWN,
        'Gematchte eenheid': UNKNOWN,
        'Aantal': UNKNOWN,
        'Eenheidsprijs': UNKNOWN,
        'Totaalbedrag': UNKNOWN,
        'Overeenkomst': '',
    })
