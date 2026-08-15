import logging
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from domain.fields import COMPARISON_FIELDS
from domain.money import parse_decimal, UNKNOWN
from matching.match_calculation import calculate_offer_total
from matching.match_fields import offer_info_from_extracted_posts
from matching.match_lookup import find_extracted_post_by_description, find_offer_result
from services.comparison_matcher import ComparisonMatcher
from services.folder_handler import FolderHandler
from services.project import Project
from utils.app_logging import logged_action


logger = logging.getLogger(__name__)


class ComparisonService:
    """Application actions for project comparisons.

    The service keeps the current JSON shape intact while centralizing the
    mutations that used to live in the NiceGUI page.
    """

    comparison_fields = set(COMPARISON_FIELDS)

    def __init__(self, folder_handler: FolderHandler, matcher: ComparisonMatcher | None = None) -> None:
        self.folder_handler = folder_handler
        self.matcher = matcher or ComparisonMatcher(folder_handler)

    def load_comparison(self, project: Project) -> dict[str, Any]:
        return project.load_comparison()

    def save_comparison(self, project: Project, comparison: dict[str, Any]) -> None:
        project.save_comparison(comparison)

    def offer_names(self, project: Project) -> list[str]:
        return [offer['Bestand'] for offer in self.matcher.project_offer_results(project)]

    def offer_post_descriptions(self, project: Project) -> dict[str, list[str]]:
        descriptions_by_offer = {}
        for offer_result in self.matcher.project_offer_results(project):
            offer_name = offer_result.get('Bestand')
            if not offer_name:
                continue

            descriptions = []
            for post in offer_result.get('Posten', []):
                if not isinstance(post, dict):
                    continue

                description = str(post.get('Omschrijving') or '').strip()
                if description and description.upper() != UNKNOWN:
                    descriptions.append(description)

            descriptions_by_offer[offer_name] = descriptions

        return descriptions_by_offer

    def has_offer_results(self, project: Project) -> bool:
        return bool(self.matcher.project_offer_results(project))

    def load_status(self, project: Project) -> dict[str, Any] | None:
        return self.folder_handler.load_comparison_status(project)

    def save_status(
        self,
        project: Project,
        *,
        status: str,
        step: str,
        message: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
    ) -> None:
        payload = {
            'status': status,
            'step': step,
            'updated_at': self.utc_now_iso(),
        }
        if started_at is not None:
            payload['started_at'] = started_at
        if message:
            payload['message'] = message
        if error:
            payload['error'] = error

        self.folder_handler.save_comparison_status(project, payload)
        logger.info(
            'Comparison status for %s: %s/%s%s',
            project.name,
            status,
            step,
            f' - {message}' if message else '',
        )

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @logged_action
    def add_comparison_row(self, project: Project, comparison: dict[str, Any]) -> None:
        comparison.setdefault('Posten', []).append({
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
        })
        self.clear_matches(comparison)
        self.save_comparison(project, comparison)

    # Structural rows that aren't individually comparable line items —
    # seeding from an offer should skip these, not turn them into
    # vergelijkingsregels of their own.
    seed_excluded_post_types = {'subtotal', 'total', 'note'}

    @logged_action
    def seed_comparison_from_offer(self, project: Project, comparison: dict[str, Any], offer_name: str) -> int:
        """Prefill comparison Posten from one offer's extracted posts.

        Lets the user start from real, LLM-extracted phrasing instead of
        typing every row by hand — which also means those rows match that
        offer exactly, and give the LLM concrete wording to match the other
        offers against instead of the user's own free-form rewording.
        Appends to any existing rows rather than replacing them. Returns
        the number of rows added (0 if the offer has nothing to seed from).
        """
        offer_result = find_offer_result(self.matcher.project_offer_results(project), offer_name)
        posts = offer_result.get('Posten', [])
        if not isinstance(posts, list):
            return 0

        seeded_rows = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            if str(post.get('PostType') or '').strip().casefold() in self.seed_excluded_post_types:
                continue

            description = str(post.get('Omschrijving') or '').strip()
            if not description or description.upper() == UNKNOWN:
                continue

            def clean(value: Any) -> str:
                text = str(value or '').strip()
                return '' if text.upper() == UNKNOWN else text

            seeded_rows.append({
                'Omschrijving': description,
                'Aantal': clean(post.get('Aantal')),
                'Eenheid': clean(post.get('Eenheid')),
            })

        if not seeded_rows:
            return 0

        comparison.setdefault('Posten', []).extend(seeded_rows)
        self.clear_matches(comparison)
        self.save_comparison(project, comparison)
        return len(seeded_rows)

    @logged_action
    def update_comparison_row(
        self,
        project: Project,
        comparison: dict[str, Any],
        row_index: int | None,
        field: str | None,
        value: str,
    ) -> bool:
        if field not in self.comparison_fields or row_index is None:
            return False

        posten = comparison.setdefault('Posten', [])
        if row_index < 0 or row_index >= len(posten):
            return False

        posten[row_index][field] = value
        self.clear_matches(comparison)
        self.save_comparison(project, comparison)
        return True

    @logged_action
    def delete_comparison_row(self, project: Project, comparison: dict[str, Any], row_index: int | None) -> bool:
        posten = comparison.setdefault('Posten', [])
        if row_index is None or row_index < 0 or row_index >= len(posten):
            return False

        posten.pop(row_index)
        self.clear_matches(comparison)
        self.save_comparison(project, comparison)
        return True

    @logged_action
    def update_matched_cell(
        self,
        project: Project,
        comparison: dict[str, Any],
        match_rows: list[dict],
        row_id: int | None,
        field: str | None,
        value: Any,
        offer_names: list[str],
    ) -> bool:
        if row_id is None or row_id < 0 or row_id >= len(match_rows):
            return False

        matched_row = match_rows[row_id]

        if field in self.comparison_fields:
            matched_row[field] = value
        elif isinstance(field, str) and field.startswith('offer_'):
            if not self.update_matched_offer_cell(project, matched_row, field, value, offer_names):
                return False
        else:
            return False

        comparison['MatchedPosten'] = match_rows
        comparison.pop('Matches', None)
        self.save_comparison(project, comparison)
        return True

    def update_matched_offer_cell(
        self,
        project: Project,
        matched_row: dict[str, Any],
        field: str,
        value: Any,
        offer_names: list[str],
    ) -> bool:
        parts = field.split('_')
        if len(parts) < 3:
            return False

        try:
            offer_index = int(parts[1])
        except ValueError:
            return False

        if offer_index < 0 or offer_index >= len(offer_names):
            return False

        suffix = parts[2]
        offer_name = offer_names[offer_index]
        offer_entry = matched_row.setdefault('Offertes', {}).setdefault(offer_name, {})

        if suffix == 'omschrijving':
            self.update_matched_offer_description(project, matched_row, offer_name, offer_entry, value)
            return True
        if suffix == 'prijs':
            offer_entry['Eenheidsprijs'] = value
            return True
        if suffix == 'totaal':
            offer_entry['Totaalbedrag'] = value
            return True

        return False

    def update_matched_offer_description(
        self,
        project: Project,
        matched_row: dict[str, Any],
        offer_name: str,
        offer_entry: dict[str, Any],
        description: Any,
    ) -> None:
        clean_descriptions = self.selected_descriptions(description)
        if not clean_descriptions:
            offer_entry.update({
                'Match type': 'single',
                'Gematchte omschrijving': UNKNOWN,
                'Gematchte posten': [],
                'Gematchte categorie': UNKNOWN,
                'Gematchte eenheid': UNKNOWN,
                'Aantal': UNKNOWN,
                'Eenheidsprijs': UNKNOWN,
                'Totaalbedrag': UNKNOWN,
                'Overeenkomst': '',
            })
            return

        offer_results = self.matcher.project_offer_results(project)
        offer_result = find_offer_result(offer_results, offer_name)
        extracted_posts = [
            find_extracted_post_by_description(offer_result, clean_description)
            for clean_description in clean_descriptions
        ]
        # find_extracted_post_by_description() returns {} (not None) on a
        # miss, so this must check falsiness, not identity — an `is None`
        # check here would never trigger, letting an empty {} silently reach
        # offer_info_from_extracted_posts() below and corrupt the total.
        if any(not extracted_post for extracted_post in extracted_posts):
            offer_entry['Match type'] = 'group' if len(clean_descriptions) > 1 else 'single'
            offer_entry['Gematchte omschrijving'] = (
                f'{len(clean_descriptions)} posten'
                if len(clean_descriptions) > 1
                else clean_descriptions[0]
            )
            offer_entry['Gematchte posten'] = clean_descriptions
            return

        offer_entry.update(offer_info_from_extracted_posts(extracted_posts, offer_entry))
        offer_entry['Gematchte posten'] = clean_descriptions
        if len(extracted_posts) == 1:
            offer_entry['Totaalbedrag'] = calculate_offer_total(
                matched_row.get('Aantal', ''),
                offer_entry,
                matched_row.get('Eenheid'),
            )

    @staticmethod
    def selected_descriptions(value: Any) -> list[str]:
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value.startswith('['):
                try:
                    parsed_value = json.loads(stripped_value)
                except json.JSONDecodeError:
                    parsed_value = value
                else:
                    if isinstance(parsed_value, list):
                        value = parsed_value

        values = value if isinstance(value, list) else [value]
        return [
            text
            for item in values
            if (text := str(item or '').strip()) and text.upper() != UNKNOWN
        ]

    @logged_action
    def delete_matched_post_row(
        self,
        project: Project,
        comparison: dict[str, Any],
        match_rows: list[dict],
        row_id: int | None,
    ) -> bool:
        if row_id is None or row_id < 0 or row_id >= len(match_rows):
            return False

        match_rows.pop(row_id)
        comparison['MatchedPosten'] = match_rows
        comparison.pop('Matches', None)
        self.save_comparison(project, comparison)
        return True

    @logged_action
    def add_matched_post_row(
        self,
        project: Project,
        comparison: dict[str, Any],
        offer_names: list[str],
    ) -> None:
        matched_posts = comparison.setdefault('MatchedPosten', [])
        if not isinstance(matched_posts, list):
            matched_posts = []
            comparison['MatchedPosten'] = matched_posts

        matched_posts.append({
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
            'Offertes': {
                offer_name: {
                    'Match type': 'single',
                    'Gematchte omschrijving': UNKNOWN,
                    'Gematchte eenheid': UNKNOWN,
                    'Eenheidsprijs': UNKNOWN,
                    'Totaalbedrag': UNKNOWN,
                    'Overeenkomst': '',
                }
                for offer_name in offer_names
            },
        })
        comparison.pop('Matches', None)
        self.save_comparison(project, comparison)

    @logged_action
    def toggle_warning(
        self,
        project: Project,
        comparison: dict[str, Any],
        warning_id: str,
        checked: bool,
    ) -> None:
        acknowledged = set(comparison.get('AfgevinkteWaarschuwingen', []))
        if checked:
            acknowledged.add(warning_id)
        else:
            acknowledged.discard(warning_id)

        comparison['AfgevinkteWaarschuwingen'] = sorted(acknowledged)
        self.save_comparison(project, comparison)

    def match_project_posts(self, project: Project, comparison: dict[str, Any]) -> dict[str, Any]:
        started_at = self.utc_now_iso()
        self.save_status(
            project,
            status='running',
            step='matching_posts',
            message='Matching comparison posts',
            started_at=started_at,
        )

        try:
            match_result = self.matcher.match_comparison_posts(project, comparison)
            self.save_status(
                project,
                status='running',
                step='normalizing_matches',
                message='Normalizing matched posts',
                started_at=started_at,
            )
            comparison['MatchedPosten'] = self.matcher.normalize_matched_posts(project, comparison, match_result)
            comparison.pop('Matches', None)
            self.save_comparison(project, comparison)
            self.save_status(
                project,
                status='done',
                step='done',
                message='Matching completed',
                started_at=started_at,
            )
            return comparison
        except Exception as error:
            self.save_status(
                project,
                status='failed',
                step='failed',
                message='Matching failed',
                error=str(error),
                started_at=started_at,
            )
            raise

    def recalculate_project_posts(self, project: Project, comparison: dict[str, Any]) -> dict[str, Any]:
        started_at = self.utc_now_iso()
        self.save_status(
            project,
            status='running',
            step='recalculating_posts',
            message='Recalculating matched posts',
            started_at=started_at,
        )

        try:
            comparison['MatchedPosten'] = self.matcher.recalculate_matched_posts(comparison, project)
            comparison.pop('Matches', None)
            self.save_comparison(project, comparison)
            self.save_status(
                project,
                status='done',
                step='done',
                message='Recalculation completed',
                started_at=started_at,
            )
            return comparison
        except Exception as error:
            self.save_status(
                project,
                status='failed',
                step='failed',
                message='Recalculation failed',
                error=str(error),
                started_at=started_at,
            )
            raise

    @staticmethod
    def comparison_total_from_json(comparison: dict[str, Any]) -> tuple[Decimal | None, str]:
        for key in ('Totaalprijs exc. BTW', 'Totaalprijs inc. BTW', 'Totaalbedrag', 'Totaal'):
            if key not in comparison:
                continue

            total = parse_decimal(comparison.get(key))
            if total is not None:
                return total, key

        return None, 'Totaal'

    @staticmethod
    def clear_matches(comparison: dict[str, Any]) -> None:
        comparison.pop('MatchedPosten', None)
        comparison.pop('Matches', None)
