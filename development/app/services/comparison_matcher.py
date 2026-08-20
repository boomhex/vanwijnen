import os

from application.offer_service import OfferService

MATCHER_MODEL_ID = os.environ.get('EXTRACT_MATCH_MODEL', 'gemini-3.5-flash')
from domain.money import calculate_unit_price
from matching.comparison_prompt import MATCH_RESPONSE_SCHEMA, build_comparison_match_prompt
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
        answer = ask_llm(prompt, response_schema=MATCH_RESPONSE_SCHEMA, model=MATCHER_MODEL_ID)
        if self.folder_handler is not None:
            self.folder_handler.save_comparison_llm_response(project, answer)

        json_response = parse_json_response(answer)
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
                offer['Eenheidsprijs'] = calculate_unit_price(
                    offer.get('Totaalbedrag'),
                    offer.get('Aantal'),
                    offer.get('Eenheidsprijs'),
                )
                offer['Totaalbedrag'] = calculate_offer_total(amount, offer, comparison_unit)

        return matched_posts
