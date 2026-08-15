import json

from domain.money import UNKNOWN
from matching.prompt_loader import load_prompt

# Fields forwarded to the matching LLM as extra content signals, on top of
# Omschrijving/Beschrijving/Categorie/Aantal/Eenheid (see match_prompt_post).
# Kept deliberately narrower than the full post schema: Inclusief/Exclusief/
# Voorwaarden/DoorOpdrachtgever/Brontekst are verbose (arrays or raw text)
# and would multiply prompt size across every post of every offer for
# comparatively little extra matching signal.
MATCH_CONTEXT_FIELDS = (
    'Code',
    'Regelnummer',
    'PostType',
    'Status',
    'Subcategorie',
    'Werksoort',
    'Prijsbasis',
)

MATCH_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'MatchedPosten': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'Omschrijving': {'type': 'string'},
                    'Offertes': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'Bestand': {'type': 'string'},
                                'Match type': {'type': 'string'},
                                'Gematchte omschrijving': {'type': 'string'},
                                'Gematchte posten': {
                                    'type': 'array',
                                    'items': {'type': 'string'},
                                },
                                'Gematchte code': {'type': 'string'},
                                'Overeenkomst': {'type': 'string'},
                            },
                            'required': [
                                'Bestand',
                                'Match type',
                                'Gematchte omschrijving',
                                'Gematchte posten',
                                'Gematchte code',
                                'Overeenkomst',
                            ],
                        },
                    },
                },
                'required': ['Omschrijving', 'Offertes'],
            },
        },
    },
    'required': ['MatchedPosten'],
}


def build_comparison_match_prompt(comparison: dict, offer_results: list[dict]) -> str:
    template = load_prompt('comparison_match_prompt.txt')
    return template.format(
        comparison_posts=json.dumps(comparison.get('Posten', []), ensure_ascii=False, indent=2),
        offer_results=json.dumps(match_prompt_offer_results(offer_results), ensure_ascii=False, indent=2),
    )


def match_prompt_offer_results(offer_results: list[dict]) -> list[dict]:
    prompt_results = []
    for offer_result in offer_results:
        posts = offer_result.get('Posten', [])
        if not isinstance(posts, list):
            posts = []

        prompt_results.append({
            'Bestand': offer_result.get('Bestand', ''),
            'Posten': [
                match_prompt_post(post)
                for post in posts
                if isinstance(post, dict)
            ],
        })

    return prompt_results


def match_prompt_post(post: dict) -> dict:
    prompt_post = {
        'Omschrijving': post.get('Omschrijving', ''),
        'Beschrijving': post.get('Beschrijving', ''),
        'Categorie': post.get('Categorie', ''),
        'Aantal': post.get('Aantal', ''),
        'Eenheid': post.get('Eenheid', ''),
    }

    for field in MATCH_CONTEXT_FIELDS:
        value = str(post.get(field) or '').strip()
        if value and value.upper() != UNKNOWN:
            prompt_post[field] = value

    match_hints = post.get('MatchHints')
    if isinstance(match_hints, list):
        clean_hints = [str(hint).strip() for hint in match_hints if str(hint or '').strip()]
        if clean_hints:
            prompt_post['MatchHints'] = clean_hints

    return prompt_post
