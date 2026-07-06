import json

from matching.prompt_loader import load_prompt


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
                                'Overeenkomst': {'type': 'string'},
                            },
                            'required': [
                                'Bestand',
                                'Match type',
                                'Gematchte omschrijving',
                                'Gematchte posten',
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
    return {
        'Omschrijving': post.get('Omschrijving', ''),
        'Beschrijving': post.get('Beschrijving', ''),
        'Categorie': post.get('Categorie', ''),
        'Aantal': post.get('Aantal', ''),
        'Eenheid': post.get('Eenheid', ''),
    }
