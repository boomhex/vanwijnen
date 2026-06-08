import json

from matching.prompt_loader import load_prompt


def build_comparison_match_prompt(comparison: dict, offer_results: list[dict]) -> str:
    template = load_prompt('comparison_match_prompt.txt')
    return template.format(
        comparison_posts=json.dumps(comparison.get('Posten', []), ensure_ascii=False, indent=2),
        offer_results=json.dumps(offer_results, ensure_ascii=False, indent=2),
    )
