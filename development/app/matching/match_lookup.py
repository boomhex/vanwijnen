from domain.money import UNKNOWN
from matching.match_fields import matched_post_descriptions

# Imported lazily inside the functions that use them (not at module level):
# services/__init__.py imports ComparisonMatcher, which pulls in this
# matching package, so a top-level `from services.extract_offer import ...`
# here would be a circular import. services/comparison_matcher.py already
# follows this same lazy-import pattern for the same reason.


def normalize_text(value: str | None) -> str:
    return ' '.join(str(value or '').casefold().split())


def build_post_index(offer_result: dict) -> dict[str, dict]:
    """Build a normalized-description → post dict for O(1) lookups."""
    index: dict[str, dict] = {}
    for post in offer_result.get('Posten', []):
        key = normalize_text(post.get('Omschrijving'))
        if key:
            index[key] = post
    return index


def build_post_code_index(offer_result: dict) -> dict[str, dict]:
    """Build a Code/Regelnummer → post dict for O(1) lookups.

    Uses the same post_code() identity as chunk-merge deduplication in
    extract_offer.py, so a code the matching LLM echoes back resolves the
    same post regardless of how its description was worded.
    """
    from services.extract_offer import post_code

    index: dict[str, dict] = {}
    for post in offer_result.get('Posten', []):
        code = post_code(post)
        if code and code not in index:
            index[code] = post
    return index


def find_similar_post(offer_result: dict, description: str | None) -> dict:
    """Fuzzy fallback when no post's description matches exactly.

    Reuses descriptions_similar() — the same heuristic already proven for
    merging duplicate posts across extraction chunks — so a matching LLM
    that paraphrases or slightly truncates a description instead of
    copying it literally still resolves to the right post.
    """
    from services.extract_offer import descriptions_similar

    if not description or str(description).strip().upper() == UNKNOWN:
        return {}

    for post in offer_result.get('Posten', []):
        if descriptions_similar(description, post.get('Omschrijving')):
            return post

    return {}


def find_extracted_post_by_code(offer_result: dict, code: str | None) -> dict:
    if not code or str(code).strip().upper() == UNKNOWN:
        return {}

    return build_post_code_index(offer_result).get(str(code).strip().casefold(), {})


def find_extracted_post_by_description(offer_result: dict, description: str | None) -> dict:
    if not description or str(description).strip().upper() == UNKNOWN:
        return {}

    exact = build_post_index(offer_result).get(normalize_text(description))
    return exact or find_similar_post(offer_result, description)


def find_offer_result(offer_results: list[dict], offer_name: str) -> dict:
    for offer_result in offer_results:
        if offer_result.get('Bestand') == offer_name:
            return offer_result

    return {}


def find_extracted_posts_for_match(offer_result: dict, offer_match: dict) -> list[dict]:
    descriptions = matched_post_descriptions(offer_match)

    # A single-post match may carry a Code/Regelnummer the LLM echoed back
    # (see comparison_match_prompt.txt) — prefer it, since it survives a
    # reworded or truncated description that would otherwise miss below.
    # Group matches don't get a per-item code, only description matching.
    if len(descriptions) == 1:
        by_code = find_extracted_post_by_code(offer_result, offer_match.get('Gematchte code'))
        if by_code:
            return [by_code]

    post_index = build_post_index(offer_result)
    extracted_posts = []
    for description in descriptions:
        if not description or str(description).strip().upper() == UNKNOWN:
            continue
        post = post_index.get(normalize_text(description)) or find_similar_post(offer_result, description)
        if post:
            extracted_posts.append(post)
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
