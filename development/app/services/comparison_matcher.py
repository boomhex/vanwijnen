from decimal import Decimal, InvalidOperation
import json

from services.folder_handler import FolderHandler
from services.project import Project

class ComparisonMatcher:
    def __init__(self, folder_handler: FolderHandler) -> None:
        self.folder_handler = folder_handler

    def project_offer_results(self, project: Project) -> list[dict]:
        offer_results = []
        for offer in project.offers():
            result = offer.load_data()
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

        prompt = f"""
            Je koppelt begrotings-/vergelijkingsregels aan offerteposten.

            Vergelijkingsregels:
            {json.dumps(comparison.get('Posten', []), ensure_ascii=False, indent=2)}

            Offerteposten per bestand:
            {json.dumps(offer_results, ensure_ascii=False, indent=2)}

            Maak per vergelijkingsregel en per offertebestand de beste match.
            Kopieer de gematchte omschrijving letterlijk uit de offerteposten.
            Geef bij iedere regel bij Overeenkomst aan hoe zeker je bent over de match met een score van 1 tot 3. Met 1 het laagst, en 3 het hoogst.
            Als er geen goede match is, vul dan "ONBEKEND" in voor de gematchte velden.
            Vul niet bij 2 posten dezelde post uit de offerte in.
            {{
            "MatchedPosten": [
                {{
                "Omschrijving": "...",
                "Offertes": {{
                    "offerte-bestandsnaam.pdf": {{
                    "Gematchte omschrijving": "...",
                    "Overeenkomst": "..."
                    }}
                }}
                }}
            ]
            }}
        """

        json_response = parse_json_response(ask_llm(prompt))
        complete_json = self.complete_response(json_response, offer_results)

        return complete_json

    def complete_response(self, json_response: dict, offer_results: list[dict]) -> dict:
        matched_posts = json_response.get('MatchedPosten') or json_response.get('MatchesPosten') or []
        if not isinstance(matched_posts, list):
            matched_posts = []

        json_response['MatchedPosten'] = [
            self.complete_all_offers(post, offer_results)
            for post in matched_posts
            if isinstance(post, dict)
        ]
        json_response.pop('MatchesPosten', None)
        return json_response

    def complete_all_offers(self, post: dict, offer_results: list[dict]) -> dict:
        raw_offers = post.get('Offertes', {})
        if not isinstance(raw_offers, dict):
            raw_offers = {}

        completed_offers = {}
        for offer_result in offer_results:
            offer_name = offer_result.get('Bestand')
            if not offer_name:
                continue

            raw_offer = raw_offers.get(offer_name, {})
            if not isinstance(raw_offer, dict):
                raw_offer = {}

            completed_offers[offer_name] = self.complete_offer_info(offer_result, raw_offer)

        post['Offertes'] = completed_offers
        return post
    
    def complete_offer_info(self, offer_result: dict, offer_match: dict) -> dict:
        extracted_post = self.find_extracted_post_by_description(
            offer_result,
            offer_match.get('Gematchte omschrijving') or offer_match.get('Omschrijving'),
        )
        if not extracted_post:
            return {
                'Gematchte omschrijving': 'ONBEKEND',
                'Gematchte eenheid': 'ONBEKEND',
                'Eenheidsprijs': 'ONBEKEND',
                'Totaalbedrag': 'ONBEKEND',
                'Overeenkomst': offer_match.get('Overeenkomst', ''),
            }

        return {
            'Gematchte omschrijving': extracted_post.get('Omschrijving', 'ONBEKEND'),
            'Gematchte eenheid': extracted_post.get('Eenheid', 'ONBEKEND'),
            'Eenheidsprijs': extracted_post.get('Eenheidsprijs', 'ONBEKEND'),
            'Totaalbedrag': extracted_post.get('Totaalbedrag', 'ONBEKEND'),
            'Overeenkomst': offer_match.get('Overeenkomst', ''),
        }

    def normalize_matched_posts(self, project: Project, comparison: dict, match_result: dict) -> list[dict]:
        offer_results = self.project_offer_results(project)
        offer_names = [offer['Bestand'] for offer in offer_results]
        raw_rows = match_result.get('MatchedPosten') or []
        flat_rows = match_result.get('Matches') or []
        normalized_rows = []

        for index, comparison_row in enumerate(comparison.get('Posten', [])):
            raw_row = self.find_matching_raw_row(raw_rows, comparison_row, index)
            offers = raw_row.get('Offertes', {}) if isinstance(raw_row, dict) else {}
            normalized_offers = {}

            for offer_name in offer_names:
                offer_match = offers.get(offer_name, {})
                if not offer_match:
                    offer_match = self.find_flat_match(flat_rows, comparison_row, offer_name)

                extracted_post = self.find_extracted_offer_post(offer_results, offer_name, offer_match)
                unit_price = self.first_known_value(
                    offer_match.get('Eenheidsprijs'),
                    extracted_post.get('Eenheidsprijs'),
                )
                total = self.calculate_total(
                    comparison_row.get('Aantal', ''),
                    unit_price,
                    self.first_known_value(
                        offer_match.get('Totaalbedrag'),
                        extracted_post.get('Totaalbedrag'),
                    ),
                )
                normalized_offers[offer_name] = {
                    'Gematchte omschrijving': self.first_known_value(
                        offer_match.get('Gematchte omschrijving'),
                        extracted_post.get('Omschrijving'),
                    ),
                    'Gematchte eenheid': self.first_known_value(
                        offer_match.get('Gematchte eenheid'),
                        extracted_post.get('Eenheid'),
                    ),
                    'Eenheidsprijs': unit_price,
                    'Totaalbedrag': total,
                    'Overeenkomst': offer_match.get('Overeenkomst', ''),
                }

            normalized_rows.append({
                'Omschrijving': comparison_row.get('Omschrijving', ''),
                'Aantal': comparison_row.get('Aantal', ''),
                'Eenheid': comparison_row.get('Eenheid', ''),
                'Offertes': normalized_offers,
            })

        return normalized_rows

    @staticmethod
    def parse_decimal(value: str | int | float | None) -> Decimal | None:
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.upper() == 'ONBEKEND':
            return None

        # Only include digits or decimal helpers.
        cleaned = ''.join(character for character in text if character.isdigit() or character in ',.-')
        if not cleaned:
            return None

        if ',' in cleaned and '.' in cleaned:
            cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            cleaned = cleaned.replace(',', '.')
        elif cleaned.count('.') > 0:
            parts = cleaned.split('.')
            cleaned = ''.join(parts[:-1]) + \
                      f"{'.' if len(parts[-1]) <= 2 else ''}"
            cleaned += parts[-1]

        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    @classmethod
    def calculate_total(cls, amount: str, unit_price: str, fallback_total: str | None = None) -> str:
        amount_value = cls.parse_decimal(amount)
        unit_price_value = cls.parse_decimal(unit_price)
        if amount_value is None or unit_price_value is None:
            return fallback_total or 'ONBEKEND'

        return f'{amount_value * unit_price_value:.2f}'

    @staticmethod
    def first_known_value(*values: str | None) -> str:
        for value in values:
            if value is None:
                continue

            text = str(value).strip()
            if text and text.upper() != 'ONBEKEND':
                return text

        return 'ONBEKEND'

    @classmethod
    def find_extracted_post_by_description(cls, offer_result: dict, description: str | None) -> dict:
        if not description or str(description).strip().upper() == 'ONBEKEND':
            return {}

        normalized_description = cls.normalize_text(description)
        for post in offer_result.get('Posten', []):
            if cls.normalize_text(post.get('Omschrijving')) == normalized_description:
                return post

        return {}

    @classmethod
    def find_extracted_offer_post(cls, offer_results: list[dict], offer_name: str, offer_match: dict) -> dict:
        matched_description = offer_match.get('Gematchte omschrijving') or offer_match.get('Omschrijving')
        if not matched_description or str(matched_description).strip().upper() == 'ONBEKEND':
            return {}

        for offer_result in offer_results:
            if offer_result.get('Bestand') != offer_name:
                continue

            return cls.find_extracted_post_by_description(offer_result, matched_description)

        return {}

    @staticmethod
    def normalize_text(value: str | None) -> str:
        return ' '.join(str(value or '').casefold().split())

    @staticmethod
    def find_matching_raw_row(raw_rows: list[dict], comparison_row: dict, index: int) -> dict:
        if index < len(raw_rows):
            return raw_rows[index]

        description = comparison_row.get('Omschrijving')
        for raw_row in raw_rows:
            if raw_row.get('Omschrijving') == description or raw_row.get('Vergelijking omschrijving') == description:
                return raw_row

        return {}

    @staticmethod
    def find_flat_match(flat_rows: list[dict], comparison_row: dict, offer_name: str) -> dict:
        description = comparison_row.get('Omschrijving')
        for raw_row in flat_rows:
            if raw_row.get('Offerte') != offer_name:
                continue

            raw_description = raw_row.get('Vergelijking omschrijving') or raw_row.get('Omschrijving')
            if raw_description == description:
                return raw_row

        return {}
