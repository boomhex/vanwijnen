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
            Gebruik "Match type": "single" als een vergelijkingsregel bij één offertepost hoort.
            Gebruik "Match type": "group" als een vergelijkingsregel bij meerdere offerteposten samen hoort.
            Kopieer gematchte omschrijvingen letterlijk uit de offerteposten.
            Geef bij iedere regel bij Overeenkomst aan hoe zeker je bent over de match met een score van 1 tot 3. Met 1 het laagst, en 3 het hoogst.
            Als er geen goede match is, vul dan "ONBEKEND" in voor de gematchte velden.
            Vul niet bij 2 posten dezelde post uit de offerte in.
            {{
            "MatchedPosten": [
                {{
                "Omschrijving": "...",
                "Offertes": {{
                    "offerte-bestandsnaam.pdf": {{
                    "Match type": "single",
                    "Gematchte omschrijving": "...",
                    "Gematchte posten": ["..."],
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
        extracted_posts = self.find_extracted_posts_for_match(offer_result, offer_match)
        if extracted_posts:
            return self.offer_info_from_extracted_posts(extracted_posts, offer_match)

        return self.offer_info_from_match(offer_match)

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

                offer_result = self.find_offer_result(offer_results, offer_name)
                normalized_offer = self.complete_offer_info(offer_result, offer_match)
                normalized_offer['Totaalbedrag'] = self.calculate_offer_total(
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

        return normalized_rows

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

                self.refresh_offer_match_from_extract(offer_results, offer_name, offer)
                offer['Totaalbedrag'] = self.calculate_offer_total(amount, offer, comparison_unit)

        return matched_posts

    def refresh_offer_match_from_extract(self, offer_results: list[dict], offer_name: str, offer_match: dict) -> None:
        offer_result = self.find_offer_result(offer_results, offer_name)
        extracted_posts = self.find_extracted_posts_for_match(offer_result, offer_match)
        if not extracted_posts:
            return

        offer_match.update(self.offer_info_from_extracted_posts(extracted_posts, offer_match))

    @classmethod
    def calculate_offer_total(cls, amount: str, offer: dict, comparison_unit: str | None = None) -> str:
        units = [
            comparison_unit,
            offer.get('Gematchte eenheid'),
            offer.get('Eenheid'),
        ]
        unit_price = offer.get('Eenheidsprijs')
        fallback_total = offer.get('Totaalbedrag')

        if cls.match_type_for_offer(offer) == 'group':
            return cls.first_known_value(fallback_total, unit_price)

        if any('post' in cls.normalize_unit(unit) for unit in units):
            return cls.first_known_value(fallback_total, unit_price)

        return cls.calculate_total(amount, unit_price, fallback_total)

    @classmethod
    def offer_info_from_extracted_posts(cls, extracted_posts: list[dict], offer_match: dict) -> dict:
        if len(extracted_posts) == 1:
            extracted_post = extracted_posts[0]
            return {
                'Match type': 'single',
                'Gematchte omschrijving': extracted_post.get('Omschrijving', 'ONBEKEND'),
                'Gematchte categorie': extracted_post.get('Categorie', 'ONBEKEND'),
                'Gematchte eenheid': extracted_post.get('Eenheid', 'ONBEKEND'),
                'Eenheidsprijs': extracted_post.get('Eenheidsprijs', 'ONBEKEND'),
                'Totaalbedrag': extracted_post.get('Totaalbedrag', 'ONBEKEND'),
                'Overeenkomst': offer_match.get('Overeenkomst', ''),
            }

        total = cls.sum_extracted_totals(extracted_posts)
        descriptions = [
            post.get('Omschrijving', '')
            for post in extracted_posts
            if post.get('Omschrijving')
        ]
        categories = cls.extracted_categories(extracted_posts)
        return {
            'Match type': 'group',
            'Gematchte omschrijving': f'{len(descriptions)} posten',
            'Gematchte posten': descriptions,
            'Gematchte categorie': cls.format_categories(categories),
            'Gematchte categorieen': categories,
            'Gematchte eenheid': 'post',
            'Eenheidsprijs': 'ONBEKEND',
            'Totaalbedrag': total,
            'Overeenkomst': offer_match.get('Overeenkomst', ''),
        }

    @classmethod
    def offer_info_from_match(cls, offer_match: dict) -> dict:
        matched_posts = cls.matched_post_descriptions(offer_match)
        match_type = cls.match_type_for_offer(offer_match)
        info = {
            'Match type': match_type,
            'Gematchte omschrijving': cls.first_known_value(
                offer_match.get('Gematchte omschrijving'),
                offer_match.get('Omschrijving'),
            ),
            'Gematchte categorie': cls.first_known_value(offer_match.get('Gematchte categorie'), offer_match.get('Categorie')),
            'Gematchte eenheid': cls.first_known_value(offer_match.get('Gematchte eenheid'), offer_match.get('Eenheid')),
            'Eenheidsprijs': cls.first_known_value(offer_match.get('Eenheidsprijs')),
            'Totaalbedrag': cls.first_known_value(offer_match.get('Totaalbedrag')),
            'Overeenkomst': offer_match.get('Overeenkomst', ''),
        }
        if matched_posts:
            info['Gematchte posten'] = matched_posts
        categories = cls.matched_categories(offer_match)
        if categories:
            info['Gematchte categorieen'] = categories
        return info

    @staticmethod
    def extracted_categories(extracted_posts: list[dict]) -> list[str]:
        categories = []
        for post in extracted_posts:
            category = str(post.get('Categorie') or '').strip()
            if category and category.upper() != 'ONBEKEND' and category not in categories:
                categories.append(category)

        return categories

    @staticmethod
    def format_categories(categories: list[str]) -> str:
        if not categories:
            return 'ONBEKEND'
        return ', '.join(categories)

    @staticmethod
    def matched_categories(offer_match: dict) -> list[str]:
        raw_categories = offer_match.get('Gematchte categorieen') or offer_match.get('Gematchte categorieën') or []
        if isinstance(raw_categories, list):
            return [
                str(category).strip()
                for category in raw_categories
                if str(category or '').strip() and str(category).strip().upper() != 'ONBEKEND'
            ]

        category = offer_match.get('Gematchte categorie') or offer_match.get('Categorie')
        if not category or str(category).strip().upper() == 'ONBEKEND':
            return []
        return [str(category).strip()]

    @classmethod
    def sum_extracted_totals(cls, extracted_posts: list[dict]) -> str:
        total = Decimal('0')
        has_total = False
        for post in extracted_posts:
            amount = cls.parse_decimal(post.get('Totaalbedrag'))
            if amount is None:
                continue

            total += amount
            has_total = True

        return f'{total:.2f}' if has_total else 'ONBEKEND'

    @classmethod
    def match_type_for_offer(cls, offer_match: dict) -> str:
        match_type = str(offer_match.get('Match type') or offer_match.get('Match Type') or '').strip().casefold()
        if match_type == 'group':
            return 'group'
        if len(cls.matched_post_descriptions(offer_match)) > 1:
            return 'group'
        return 'single'

    @staticmethod
    def matched_post_descriptions(offer_match: dict) -> list[str]:
        raw_posts = offer_match.get('Gematchte posten') or offer_match.get('Gematchte Posten') or []
        if isinstance(raw_posts, list):
            descriptions = [
                str(description).strip()
                for description in raw_posts
                if str(description or '').strip() and str(description).strip().upper() != 'ONBEKEND'
            ]
            if descriptions:
                return descriptions

        description = offer_match.get('Gematchte omschrijving') or offer_match.get('Omschrijving')
        if not description or str(description).strip().upper() == 'ONBEKEND':
            return []
        return [str(description).strip()]

    @classmethod
    def warning_for_offer(cls, match_row: dict, offer: dict) -> str:
        warnings = []
        comparison_unit = match_row.get('Eenheid', '')
        matched_unit = (
            offer.get('Gematchte eenheid')
            or offer.get('Eenheid')
            or ''
        )

        if cls.units_mismatch(comparison_unit, matched_unit) and not cls.post_total_covers_mismatch(
            comparison_unit,
            matched_unit,
            offer,
        ):
            warnings.append(f'Eenheid wijkt af: vergelijking {comparison_unit}, offerte {matched_unit}.')

        score = str(offer.get('Overeenkomst', '')).strip()
        if score in {'1', '2'}:
            warnings.append(f'Lage overeenkomstscore ({score}). Controleer de match.')

        return ' '.join(warnings)

    @classmethod
    def units_mismatch(cls, comparison_unit: str | None, matched_unit: str | None) -> bool:
        comparison = cls.normalize_unit(comparison_unit)
        matched = cls.normalize_unit(matched_unit)
        return bool(comparison and matched and comparison != matched)

    @classmethod
    def post_total_covers_mismatch(cls, comparison_unit: str | None, matched_unit: str | None, offer: dict) -> bool:
        comparison = cls.normalize_unit(comparison_unit)
        matched = cls.normalize_unit(matched_unit)
        if 'post' not in {comparison, matched}:
            return False

        return cls.parse_decimal(offer.get('Totaalbedrag')) is not None

    @staticmethod
    def normalize_unit(value: str | None) -> str:
        text = str(value or '').strip().casefold()
        if not text or text == 'onbekend':
            return ''

        return (
            text
            .replace('²', '2')
            .replace('¹', '1')
            .replace(' ', '')
            .replace('.', '')
        )

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

    @staticmethod
    def find_offer_result(offer_results: list[dict], offer_name: str) -> dict:
        for offer_result in offer_results:
            if offer_result.get('Bestand') == offer_name:
                return offer_result

        return {}

    @classmethod
    def find_extracted_posts_for_match(cls, offer_result: dict, offer_match: dict) -> list[dict]:
        extracted_posts = []
        for description in cls.matched_post_descriptions(offer_match):
            extracted_post = cls.find_extracted_post_by_description(offer_result, description)
            if extracted_post:
                extracted_posts.append(extracted_post)

        return extracted_posts

    @classmethod
    def find_extracted_offer_post(cls, offer_results: list[dict], offer_name: str, offer_match: dict) -> dict:
        offer_result = cls.find_offer_result(offer_results, offer_name)
        extracted_posts = cls.find_extracted_posts_for_match(offer_result, offer_match)
        return extracted_posts[0] if extracted_posts else {}

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
