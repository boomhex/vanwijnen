from application.offer_service import OfferService
from matching.comparison_prompt import build_comparison_match_prompt
from matching.match_calculation import calculate_offer_total
from matching.match_normalizer import (
    complete_response,
    normalize_matched_posts,
    refresh_offer_match_from_extract,
)
from services.folder_handler import FolderHandler
from services.project import Project


class ComparisonMatcher:
    """Match comparison posts to extracted offer posts.

    This class is now mostly an orchestration façade. The detailed money,
    unit, warning, lookup, and normalization rules live in smaller modules.
    """

    def __init__(self, folder_handler: FolderHandler | None) -> None:
        self.folder_handler = folder_handler
        self.offer_service = OfferService(folder_handler) if folder_handler is not None else None

    def project_offer_results(self, project: Project) -> list[dict]:
        if self.offer_service is None:
            return []

        offer_results = []
        for offer in project.offers():
            result = self.offer_service.load_data(offer)
            if result is None:
                continue

            offer_results.append({
                'Bestand': offer.comparison_key,
                'Posten': result.get('Posten', []),
            })

        return offer_results

    def match_comparison_posts(self, project: Project, comparison: dict | None = None) -> dict:
        from services.extract_offer import ask_llm, parse_json_response

        comparison = comparison or project.load_comparison()
        offer_results = self.project_offer_results(project)
        prompt = build_comparison_match_prompt(comparison, offer_results)
        json_response = parse_json_response(ask_llm(prompt))
        return complete_response(json_response, offer_results)

    def normalize_matched_posts(self, project: Project, comparison: dict, match_result: dict) -> list[dict]:
        offer_results = self.project_offer_results(project)
        return normalize_matched_posts(comparison, match_result, offer_results)

    def recalculate_matched_posts(self, comparison: dict, project: Project | None = None) -> list[dict]:
        matched_posts = comparison.get('MatchedPosten', [])
        if not isinstance(matched_posts, list):
            return []

        offer_results = self.project_offer_results(project) if project is not None else []

        for matched_row in matched_posts:
            if not isinstance(matched_row, dict):
                continue

            amount = matched_row.get('Aantal', '')
            comparison_unit = matched_row.get('Eenheid')
            offers = matched_row.get('Offertes', {})
            if not isinstance(offers, dict):
                continue

            for offer_name, offer in offers.items():
                if not isinstance(offer, dict):
                    continue

                refresh_offer_match_from_extract(offer_results, offer_name, offer)
                offer['Totaalbedrag'] = calculate_offer_total(amount, offer, comparison_unit)

        return matched_posts
