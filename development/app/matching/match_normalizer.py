from matching.match_calculation import calculate_offer_total
from matching.match_deduplicator import deduplicate_offer_matches
from matching.match_fields import matched_post_descriptions, offer_info_from_extracted_posts, offer_info_from_match
from matching.match_lookup import (
    find_extracted_posts_for_match,
    find_flat_match,
    find_matching_raw_row,
    find_offer_result,
)


def complete_response(json_response: dict, offer_results: list[dict]) -> dict:
    matched_posts = json_response.get('MatchedPosten') or json_response.get('MatchesPosten') or []
    if not isinstance(matched_posts, list):
        matched_posts = []

    json_response['MatchedPosten'] = [
        complete_all_offers(post, offer_results)
        for post in matched_posts
        if isinstance(post, dict)
    ]
    json_response.pop('MatchesPosten', None)
    return json_response


def complete_all_offers(post: dict, offer_results: list[dict]) -> dict:
    raw_offers = normalize_offer_matches(post.get('Offertes'))

    completed_offers = {}
    for offer_result in offer_results:
        offer_name = offer_result.get('Bestand')
        if not offer_name:
            continue

        raw_offer = raw_offers.get(offer_name, {})
        if not isinstance(raw_offer, dict):
            raw_offer = {}

        completed_offers[offer_name] = complete_offer_info(offer_result, raw_offer)

    post['Offertes'] = completed_offers
    return post


def normalize_offer_matches(raw_offers) -> dict:
    if isinstance(raw_offers, dict):
        return raw_offers

    if not isinstance(raw_offers, list):
        return {}

    normalized = {}
    for raw_offer in raw_offers:
        if not isinstance(raw_offer, dict):
            continue

        offer_name = raw_offer.get('Bestand') or raw_offer.get('Offerte')
        if not offer_name:
            continue

        offer_match = dict(raw_offer)
        offer_match.pop('Bestand', None)
        offer_match.pop('Offerte', None)
        normalized[offer_name] = offer_match

    return normalized


def complete_offer_info(offer_result: dict, offer_match: dict) -> dict:
    extracted_posts = find_extracted_posts_for_match(offer_result, offer_match)
    if extracted_posts:
        return offer_info_from_extracted_posts(extracted_posts, offer_match)

    info = offer_info_from_match(offer_match)
    # The LLM claimed a match (a description/code) but it couldn't be
    # resolved to a real extracted post — these values are the raw,
    # unverified LLM output rather than data read from the offer itself.
    # Distinct from "no match claimed at all", which isn't a problem.
    info['Ongekoppeld'] = bool(matched_post_descriptions(offer_match))
    return info


def normalize_matched_posts(comparison: dict, match_result: dict, offer_results: list[dict]) -> list[dict]:
    offer_names = [offer['Bestand'] for offer in offer_results]
    raw_rows = match_result.get('MatchedPosten') or []
    flat_rows = match_result.get('Matches') or []
    normalized_rows = []

    for index, comparison_row in enumerate(comparison.get('Posten', [])):
        raw_row = find_matching_raw_row(raw_rows, comparison_row, index)
        offers = normalize_offer_matches(raw_row.get('Offertes')) if isinstance(raw_row, dict) else {}
        normalized_offers = {}

        for offer_name in offer_names:
            offer_match = offers.get(offer_name, {})
            if not offer_match:
                offer_match = find_flat_match(flat_rows, comparison_row, offer_name)

            offer_result = find_offer_result(offer_results, offer_name)
            normalized_offer = complete_offer_info(offer_result, offer_match)
            normalized_offer['Totaalbedrag'] = calculate_offer_total(
                comparison_row.get('Aantal', ''),
                normalized_offer,
                comparison_row.get('Eenheid'),
            )
            normalized_offers[offer_name] = normalized_offer

        normalized_rows.append({
            'Omschrijving': comparison_row.get('Omschrijving', ''),
            'Aantal': comparison_row.get('Aantal', ''),
            'Eenheid': comparison_row.get('Eenheid', ''),
            'Offertes': normalized_offers,
        })

    deduplicate_offer_matches(normalized_rows, offer_results)
    return normalized_rows


def refresh_offer_match_from_extract(offer_results: list[dict], offer_name: str, offer_match: dict) -> None:
    offer_result = find_offer_result(offer_results, offer_name)
    extracted_posts = find_extracted_posts_for_match(offer_result, offer_match)
    if not extracted_posts:
        return

    offer_match.update(offer_info_from_extracted_posts(extracted_posts, offer_match))

