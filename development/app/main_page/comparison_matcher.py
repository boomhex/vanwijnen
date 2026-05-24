from decimal import Decimal, InvalidOperation
from pathlib import Path
import json

from main_page.folder_handler import FolderHandler


class ComparisonMatcher:
    def __init__(self, folder_handler: FolderHandler) -> None:
        self.folder_handler = folder_handler

    def project_offer_results(self, project: Path) -> list[dict]:
        offer_results = []
        for file in self.folder_handler.project_files(project):
            result = self.folder_handler.load_result(file)
            if result is None:
                continue

            offer_results.append({
                'Bestand': file.name,
                'Posten': result.get('Posten', []),
            })

        return offer_results

    def match_comparison_posts(self, project: Path, comparison: dict) -> dict:
        from main_page.extract_offer import ask_llm, parse_json_response

        prompt = f"""
            Je koppelt begrotings-/vergelijkingsregels aan offerteposten.

            Vergelijkingsregels:
            {json.dumps(comparison.get('Posten', []), ensure_ascii=False, indent=2)}

            Offerteposten per bestand:
            {json.dumps(self.project_offer_results(project), ensure_ascii=False, indent=2)}

            Maak per vergelijkingsregel en per offertebestand de beste match.
            Gebruik voor "Aantal" en "Eenheid" de vergelijkingsregel.
            Neem bij iedere offerte de gematchte "Eenheidsprijs" over uit de offertepost als die beschikbaar is.
            Als er geen goede match is, vul dan "ONBEKEND" in voor de gematchte velden.
            Vul niet bij 2 posten dezelde post uit de offerte in.
            {{
            "MatchedPosten": [
                {{
                "Omschrijving": "...",
                "Aantal": "...",
                "Eenheid": "...",
                "Offertes": {{
                    "offerte-bestandsnaam.pdf": {{
                    "Gematchte omschrijving": "...",
                    "Gematchte eenheid": "...",
                    "Eenheidsprijs": "...",
                    "Totaalbedrag": "...",
                    "Match toelichting": "..."
                    }}
                }}
                }}
            ]
            }}
        """
        return parse_json_response(ask_llm(prompt))

    def normalize_matched_posts(self, project: Path, comparison: dict, match_result: dict) -> list[dict]:
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
                    'Match toelichting': offer_match.get('Match toelichting', ''),
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

        cleaned = ''.join(character for character in text if character.isdigit() or character in ',.-')
        if not cleaned:
            return None

        if ',' in cleaned and '.' in cleaned:
            cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            cleaned = cleaned.replace(',', '.')
        elif cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]

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
    def find_extracted_offer_post(cls, offer_results: list[dict], offer_name: str, offer_match: dict) -> dict:
        matched_description = offer_match.get('Gematchte omschrijving') or offer_match.get('Omschrijving')
        if not matched_description or str(matched_description).strip().upper() == 'ONBEKEND':
            return {}

        for offer_result in offer_results:
            if offer_result.get('Bestand') != offer_name:
                continue

            for post in offer_result.get('Posten', []):
                if cls.normalize_text(post.get('Omschrijving')) == cls.normalize_text(matched_description):
                    return post

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
