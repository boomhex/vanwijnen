import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from domain.fields import COMPARISON_FIELDS
from domain.money import parse_decimal, UNKNOWN
from services.comparison_matcher import ComparisonMatcher
from services.folder_handler import FolderHandler
from services.project import Project


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

    def add_comparison_row(self, project: Project, comparison: dict[str, Any]) -> None:
        comparison.setdefault('Posten', []).append({
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
        })
        self.clear_matches(comparison)
        self.save_comparison(project, comparison)

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

    def delete_comparison_row(self, project: Project, comparison: dict[str, Any], row_index: int | None) -> bool:
        posten = comparison.setdefault('Posten', [])
        if row_index is None or row_index < 0 or row_index >= len(posten):
            return False

        posten.pop(row_index)
        self.clear_matches(comparison)
        self.save_comparison(project, comparison)
        return True

    def update_matched_cell(
        self,
        project: Project,
        comparison: dict[str, Any],
        match_rows: list[dict],
        row_id: int | None,
        field: str | None,
        value: str,
        offer_names: list[str],
    ) -> bool:
        if row_id is None or row_id < 0 or row_id >= len(match_rows):
            return False

        matched_row = match_rows[row_id]

        if field in self.comparison_fields:
            matched_row[field] = value
        elif isinstance(field, str) and field.startswith('offer_'):
            if not self.update_matched_offer_cell(matched_row, field, value, offer_names):
                return False
        else:
            return False

        comparison['MatchedPosten'] = match_rows
        comparison.pop('Matches', None)
        self.save_comparison(project, comparison)
        return True

    @staticmethod
    def update_matched_offer_cell(
        matched_row: dict[str, Any],
        field: str,
        value: str,
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

        if suffix == 'prijs':
            offer_entry['Eenheidsprijs'] = value
            return True
        if suffix == 'totaal':
            offer_entry['Totaalbedrag'] = value
            return True

        return False

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
