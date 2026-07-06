from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from decimal import Decimal
import json

from nicegui import ui, run
from nicegui_tabulator import tabulator

from application.comparison_service import ComparisonService
from domain.comparison_checks import warnings_for_offer
from domain.fields import COMPARISON_FIELDS
from domain.money import parse_decimal, UNKNOWN
from domain.status import is_active_running, is_stale_running
from matching.match_fields import matched_categories, matched_post_descriptions
from .subpage import SubPage

from services.comparison_matcher import ComparisonMatcher
from services.folder_handler import FolderHandler
from services.project import Project
from .tabulator_table import TabulatorTable
from interface.page_state import MainPageState
from utils.app_logging import log_action


class ComparisonRowsTable(TabulatorTable):
    fields = COMPARISON_FIELDS

    def __init__(self, comparison: dict) -> None:
        self.comparison = comparison
        super().__init__(
            rows=self.rows_from_comparison(),
            columns=[
                self.text_column('Omschrijving', 'Omschrijving', editable=True, width=320, multiline=True),
                self.text_column('Aantal', 'Aantal', editable=True, width=120),
                self.text_column('Eenheid', 'Eenheid', editable=True, width=120),
                {
                    'title': '',
                    'field': '__delete__',
                    'width': 52,
                    'headerSort': False,
                    'hozAlign': 'center',
                    ':formatter': "function(){ return 'x'; }",
                },
            ],
            height='32vh',
        )

    def rows_from_comparison(self) -> list[dict]:
        return [
            {'id': index, **row}
            for index, row in enumerate(self.comparison.get('Posten', []))
        ]


class MatchedPostenTable(TabulatorTable):
    def __init__(
        self,
        *,
        offer_names: list[str],
        match_rows: list[dict],
        offer_post_descriptions: dict[str, list[str]] | None = None,
        acknowledged_warnings: set[str] | None = None,
    ) -> None:
        self.acknowledged_warnings = acknowledged_warnings or set()
        self.offer_post_descriptions = offer_post_descriptions or {}
        offer_totals = self.total_decimals_by_offer(offer_names, match_rows)
        self.offer_names = sorted(
            offer_names,
            key=lambda name: (
                offer_totals.get(name) is None,
                offer_totals.get(name, Decimal('0')),
                name.lower(),
            ),
        )
        self.offer_prefixes = {offer_name: f'offer_{index}' for index, offer_name in enumerate(self.offer_names)}
        super().__init__(
            rows=self.rows_from_matches(match_rows, offer_totals),
            columns=self.columns_from_offers(),
            layout='fitDataStretch',
            reactive=False,
            height='68vh',
        )

    def columns_from_offers(self) -> list[dict]:
        columns = [
            self.text_column('Omschrijving', 'Omschrijving', width=220, multiline=True),
            self.text_column('Aantal', 'Aantal', editable=True, width=100),
            self.text_column('Eenheid', 'Eenheid', editable=True, width=100),
        ]

        # Only show the price and total columns per offer in the UI.
        for offer_name in self.offer_names:
            field_prefix = self.offer_field_prefix(offer_name)
            columns.extend([
                self.matched_description_column(f'{offer_name} post', f'{field_prefix}_omschrijving', width=220),
                self.warning_column(f'{offer_name} prijs', f'{field_prefix}_prijs', editable=True, width=120),
                self.warning_column(f'{offer_name} totaal', f'{field_prefix}_totaal', editable=True, width=130),
                self.warning_column(f'{offer_name} %', f'{field_prefix}_verschil', width=90),
            ])

        columns.append({
            'title': '',
            'field': '__delete__',
            'width': 52,
            'headerSort': False,
            'hozAlign': 'center',
            ':formatter': "function(cell){ return cell.getRow().getData().Omschrijving === 'Totaal' ? '' : 'x'; }",
        })

        return columns

    def matched_description_column(self, title: str, field: str, *, width: int | None = None) -> dict:
        column = self.text_column(title, field, editable=True, width=width, multiline=True)
        column['editor'] = 'list'
        options_field = f'{field}_options'
        column[':editorParams'] = f"""
            function(cell) {{
                const data = cell.getRow().getData();
                return {{
                    values: data[{json.dumps(options_field)}] || [],
                    clearable: false,
                    multiselect: true,
                    maxWidth: true,
                }};
            }}
        """
        column[':formatter'] = """
            function(cell) {
                const element = document.createElement('div');
                const value = cell.getValue();
                element.textContent = Array.isArray(value) ? value.join(', ') : (value || '');
                element.style.whiteSpace = 'normal';
                element.style.overflowWrap = 'break-word';
                element.style.lineHeight = '1.25';
                return element;
            }
        """
        column['variableHeight'] = True
        return column

    def warning_column(
        self,
        title: str,
        field: str,
        *,
        editable: bool = False,
        width: int | None = None,
    ) -> dict:
        column = self.text_column(title, field, editable=editable, width=width)
        warning_field = f'{field}_warning'
        tooltip_field = f'{field}_tooltip'
        column[':formatter'] = f"""
            function(cell) {{
                const data = cell.getRow().getData();
                const warning = data[{json.dumps(warning_field)}];
                const tooltip = data[{json.dumps(tooltip_field)}] || warning || '';
                const element = cell.getElement();

                element.style.backgroundColor = warning ? '#FEF3C7' : '';
                element.style.color = warning ? '#92400E' : '';
                element.style.fontWeight = warning ? '600' : '';
                element.removeAttribute('title');

                const value = cell.getValue();
                return value === null || value === undefined ? '' : value;
            }}
        """
        column[':tooltip'] = f"""
            function(e, cell) {{
                const data = cell.getRow().getData();
                return data[{json.dumps(tooltip_field)}] || data[{json.dumps(warning_field)}] || false;
            }}
        """
        return column

    def offer_field_prefix(self, offer_name: str) -> str:
        return self.offer_prefixes[offer_name]

    def rows_from_matches(self, match_rows: list[dict], offer_totals: dict[str, Decimal | None] | None = None) -> list[dict]:
        rows = []
        for index, match_row in enumerate(match_rows):
            row = {
                'id': index,
                'Omschrijving': match_row.get('Omschrijving', ''),
                'Aantal': match_row.get('Aantal', ''),
                'Eenheid': match_row.get('Eenheid', ''),
            }
            offers = match_row.get('Offertes', {})
            for offer_name in self.offer_names:
                field_prefix = self.offer_field_prefix(offer_name)
                offer = offers.get(offer_name, {})
                matched_descriptions = matched_post_descriptions(offer)
                row[f'{field_prefix}_omschrijving'] = matched_descriptions or [UNKNOWN]
                row[f'{field_prefix}_omschrijving_options'] = self.available_post_options(
                    offer_name,
                    match_rows,
                    index,
                    row[f'{field_prefix}_omschrijving'],
                )

                # Handle posts that are totals (Eenheid == 'post'). In that case the
                # supplier put a total price on the post-level. Prefer to show that
                # value in the totaal column and leave the eenheidsprijs empty.
                eenheidsprijs = offer.get('Eenheidsprijs') or ''
                # Prefer the explicitly matched unit, fall back to the offer's own unit
                gematchte_eenheid = (
                    (offer.get('Gematchte eenheid') or '')
                    or (offer.get('Eenheid') or '')
                ).strip().casefold()

                # If the supplier marked the post as a 'post' unit (i.e. a total
                # rather than a per-unit price), move the unit price into the
                # totaal column when a total isn't already provided.
                # Prefer to show totals for 'post' unit types
                raw_een = offer.get('Eenheidsprijs') or ''
                raw_tot = offer.get('Totaalbedrag') or ''
                if 'post' in gematchte_eenheid:
                    if raw_een and not raw_tot:
                        raw_tot = raw_een
                    raw_een = ''

                def _fmt_money(val):
                    dec = parse_decimal(val)
                    if dec is None:
                        return UNKNOWN
                    return self.format_money(dec)

                row[f'{field_prefix}_prijs'] = _fmt_money(raw_een) if raw_een not in (None, '') else UNKNOWN
                row[f'{field_prefix}_totaal'] = _fmt_money(raw_tot) if raw_tot not in (None, '') else UNKNOWN
                warnings = warnings_for_offer(match_row, offer)
                warning_ids = [
                    self.warning_id(index, offer_name, warning)
                    for warning in warnings
                ]
                active_warnings = [
                    warning
                    for warning, warning_id in zip(warnings, warning_ids, strict=False)
                    if warning_id not in self.acknowledged_warnings
                ]
                warning = ' '.join(active_warnings)
                tooltip = self.tooltip_for_offer(offer, row[f'{field_prefix}_omschrijving'], warning)
                row[f'{field_prefix}_prijs_warning'] = warning
                row[f'{field_prefix}_totaal_warning'] = warning
                row[f'{field_prefix}_verschil_warning'] = warning
                row[f'{field_prefix}_prijs_tooltip'] = tooltip
                row[f'{field_prefix}_totaal_tooltip'] = tooltip
                row[f'{field_prefix}_verschil_tooltip'] = tooltip
                row[f'{field_prefix}_verschil'] = ''

            self.add_difference_percentages(row, self.offer_names)
            rows.append(row)

        # Append a totals row which sums the per-offer subtotals.
        if offer_totals is None:
            offer_totals = self.total_decimals_by_offer(self.offer_names, match_rows)

        totals_row = {
            'id': len(rows),
            'Omschrijving': 'Totaal',
            'Aantal': '',
            'Eenheid': '',
        }

        for offer_name in self.offer_names:
            field_prefix = self.offer_field_prefix(offer_name)
            total = offer_totals.get(offer_name)
            totals_row[f'{field_prefix}_omschrijving'] = 'Totaal'
            totals_row[f'{field_prefix}_omschrijving_options'] = ['Totaal']
            totals_row[f'{field_prefix}_prijs'] = ''
            totals_row[f'{field_prefix}_totaal'] = self.format_money(total) if total is not None else UNKNOWN
            totals_row[f'{field_prefix}_verschil'] = ''
            totals_row[f'{field_prefix}_prijs_tooltip'] = 'Totaal'
            totals_row[f'{field_prefix}_totaal_tooltip'] = 'Totaal'
            totals_row[f'{field_prefix}_verschil_tooltip'] = 'Totaal'

        self.add_difference_percentages(totals_row, self.offer_names)
        rows.append(totals_row)

        return rows

    def available_post_options(
        self,
        offer_name: str,
        match_rows: list[dict],
        current_row_index: int,
        current_value,
    ) -> list[str]:
        all_descriptions = self.offer_post_descriptions.get(offer_name, [])
        used_descriptions = set()

        for row_index, match_row in enumerate(match_rows):
            if row_index == current_row_index:
                continue

            offer = match_row.get('Offertes', {}).get(offer_name, {})
            if not isinstance(offer, dict):
                continue

            for description in matched_post_descriptions(offer):
                if description and str(description).strip().upper() != UNKNOWN:
                    used_descriptions.add(str(description).strip())

        options = [
            description
            for description in all_descriptions
            if description not in used_descriptions
        ]
        current_values = self.current_description_values(current_value)
        for current_text in reversed(current_values):
            if current_text and current_text not in options:
                options.insert(0, current_text)
        if UNKNOWN not in options:
            options.insert(0, UNKNOWN)

        return options

    @staticmethod
    def current_description_values(value) -> list[str]:
        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item or '').strip()
            ]

        text = str(value or '').strip()
        return [text] if text else []

    @staticmethod
    def warning_id(row_index: int, offer_name: str, warning: str) -> str:
        return f'{row_index}|{offer_name}|{warning}'

    @staticmethod
    def tooltip_for_offer(offer: dict, matched_description, warning: str) -> str:
        matched_posts = matched_post_descriptions(offer)
        categories = matched_categories(offer)
        if len(matched_posts) > 1:
            tooltip = 'Gematchte posten:\n' + '\n'.join(f'- {description}' for description in matched_posts)
        else:
            descriptions = MatchedPostenTable.current_description_values(matched_description)
            tooltip_description = descriptions[0] if descriptions else UNKNOWN
            tooltip = f'Gematchte omschrijving: {tooltip_description}'

        if categories:
            tooltip += '\nCategorie: ' + ', '.join(categories)

        if warning:
            tooltip += f'\nWaarschuwing: {warning}'

        return tooltip

    @classmethod
    def total_decimals_by_offer(cls, offer_names: list[str], match_rows: list[dict]) -> dict[str, Decimal | None]:
        totals: dict[str, Decimal] = {}
        has_value: dict[str, bool] = {}

        for offer_name in offer_names:
            totals[offer_name] = Decimal('0')
            has_value[offer_name] = False

        for match_row in match_rows:
            if not isinstance(match_row, dict):
                continue

            amount = parse_decimal(match_row.get('Aantal'))
            offers = match_row.get('Offertes', {})
            if not isinstance(offers, dict):
                continue

            for offer_name in offer_names:
                offer = offers.get(offer_name, {})
                if not isinstance(offer, dict):
                    continue

                value = parse_decimal(offer.get('Totaalbedrag'))
                if value is None and amount is not None:
                    unit_price = parse_decimal(offer.get('Eenheidsprijs'))
                    if unit_price is not None:
                        value = amount * unit_price

                if value is None:
                    continue

                totals[offer_name] += value
                has_value[offer_name] = True

        return {
            offer_name: totals[offer_name] if has_value[offer_name] else None
            for offer_name in offer_names
        }

    @classmethod
    def add_difference_percentages(cls, row: dict, offer_names: list[str]) -> None:
        totals = {
            index: parse_decimal(row.get(f'offer_{index}_totaal'))
            for index, _offer_name in enumerate(offer_names)
        }
        known_totals = [total for total in totals.values() if total is not None]
        if not known_totals:
            return

        lowest_total = min(known_totals)
        for index, total in totals.items():
            field_prefix = f'offer_{index}'
            if total is None:
                row[f'{field_prefix}_verschil'] = UNKNOWN
                continue

            row[f'{field_prefix}_verschil'] = cls.format_percentage_difference(total, lowest_total)

    @staticmethod
    def format_percentage_difference(total: Decimal, lowest_total: Decimal) -> str:
        if lowest_total == 0:
            return '0,0%' if total == 0 else UNKNOWN

        percentage = ((total - lowest_total) / lowest_total * Decimal('100')).quantize(Decimal('0.1'))
        if percentage == 0:
            return '0,0%'

        return f'+{str(percentage).replace(".", ",")}%'

    def totals_by_offer(self) -> dict[str, str]:
        totals: dict[str, Decimal] = {}

        for offer_name in self.offer_names:
            field_prefix = self.offer_field_prefix(offer_name)
            total = Decimal('0')
            has_known_value = False

            for row in self.rows:
                row_total = parse_decimal(row.get(f'{field_prefix}_totaal'))
                if row_total is None:
                    continue

                total += row_total
                has_known_value = True

            if has_known_value:
                totals[offer_name] = total

        return {
            offer_name: self.format_money(total) if offer_name in totals else UNKNOWN
            for offer_name, total in ((name, totals.get(name)) for name in self.offer_names)
        }

    def to_excel_clipboard_text(self) -> str:
        export_columns = [
            column
            for column in self.columns
            if column.get('field') and not str(column.get('field')).startswith('__')
        ]
        lines = [
            '\t'.join(self.clean_clipboard_cell(column.get('title', '')) for column in export_columns)
        ]

        for row in self.rows:
            lines.append(
                '\t'.join(
                    self.clean_clipboard_cell(row.get(column['field'], ''))
                    for column in export_columns
                )
            )

        return '\n'.join(lines)

    @staticmethod
    def clean_clipboard_cell(value) -> str:
        if isinstance(value, list):
            return ', '.join(' '.join(str(item or '').split()) for item in value)
        return ' '.join(str(value or '').split())




class ComparisonPage(SubPage):
    def __init__(
        self,
        state: MainPageState,
        folder_handler: FolderHandler,
        projects_dir: Path,
        refresh: Callable[[], None] | None = None,
        matcher: ComparisonMatcher | None = None,
    ) -> None:
        super().__init__(state, folder_handler, projects_dir, refresh)
        self.matcher = matcher or ComparisonMatcher(folder_handler)
        self.comparison_service = ComparisonService(folder_handler, self.matcher)

    def render(self):
        with ui.column().classes('w-full h-full'):
            with ui.column().classes('w-full p-4'):
                self.show()

    def show(self) -> None:
        # Check if project is selected
        project = self.state.comparison_project
        if project is None:
            ui.label('No project selected').classes('text-gray-500')
            return

        # Show title
        ui.label(f'Comparison: {project.name}').classes('text-xl font-bold')

        # Show invoer
        comparison = self.comparison_service.load_comparison(project)
        self.input_table(project, comparison)

        # Show match button
        self.match_button(project, comparison)

        # Show comparison
        match_rows = comparison.get('MatchedPosten', [])
        if not match_rows:
            return

        # Show match table
        ui.label('Gematchte Posten').classes('text-lg font-bold mt-4')
        self.render_side_by_side_match_table(project, comparison, match_rows)

    def match_button(self, project: Project, comparison: dict) -> None:
        with ui.row().classes('items-center gap-2 mt-4'):
            status = self.comparison_service.load_status(project)
            is_running = is_active_running(status)
            is_stale = is_stale_running(status)
            failed = bool(status and status.get('status') == 'failed')
            status_message = status.get('message') if status else None
            status_step = status.get('step') if status else None
            status_error = status.get('error') if status else None

            match_button = ui.button(
                'Matching' if is_running else 'Match Posten',
                icon='eva-bulb-outline',
            ).props('dense no-caps')
            recalculate_button = ui.button(
                'Recalculating' if is_running and status_step == 'recalculating_posts' else 'Recalculate',
                icon='calculate',
                on_click=lambda selected_project=project, data=comparison: self.recalculate_project_posts(
                    selected_project,
                    data,
                ),
            ).props('dense no-caps')

            if not comparison.get('MatchedPosten'):
                recalculate_button.props('disable')

            if is_running:
                match_button.props('loading disable')
                recalculate_button.props('loading disable')
                if status_message or status_step:
                    match_button.tooltip(status_message or status_step)
                    recalculate_button.tooltip(status_message or status_step)

            if failed:
                warning_icon = ui.icon('warning').classes('text-red-700')
                warning_icon.tooltip(status_error or status_message or 'Comparison failed')
            elif is_stale:
                warning_icon = ui.icon('warning').classes('text-orange-700')
                warning_icon.tooltip('Comparison status is stale. You can retry matching or recalculating.')

            async def request_match(_event, selected_project=project, data=comparison, button=match_button):
                await self.match_project_posts(selected_project, data, button)

            if not is_running:
                match_button.on('click', request_match)

    def input_table(self, project: Project, comparison: dict) -> None:
        comparison_table = ComparisonRowsTable(comparison)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten voor Vergelijking').classes('text-lg font-bold')
            ui.button(
                'Add row',
                icon='add',
                on_click=lambda: self.add_comparison_row(project, comparison),
            ).props('dense no-caps size=sm')

        comparison_tabulator = tabulator(comparison_table.options(), row_key='id').classes('w-full')

        def update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            self.comparison_service.update_comparison_row(
                project, comparison, row.get('id'), column.get('field'), cell.get('value', '')
            )

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return
            row = cell.get('row', {})
            if not self.comparison_service.delete_comparison_row(project, comparison, row.get('id')):
                return
            comparison_table.rows = comparison_table.rows_from_comparison()
            comparison_tabulator.set_data(comparison_table.rows)

        comparison_tabulator.on_event('cellEdited', update_cell)
        comparison_tabulator.on_event('cellClick', delete_row)

    def add_comparison_row(self, project: Project, comparison: dict) -> None:
        self.comparison_service.add_comparison_row(project, comparison)
        self.refresh()

    def delete_comparison_row(self, project: Project, comparison: dict, row_index: int) -> None:
        if not self.comparison_service.delete_comparison_row(project, comparison, row_index):
            ui.notify('Row no longer exists')
            self.refresh()
            return

        self.refresh()

    def render_side_by_side_match_table(self, project: Project, comparison: dict, match_rows: list[dict]) -> None:
        offer_names = self.comparison_service.offer_names(project)
        offer_post_descriptions = self.comparison_service.offer_post_descriptions(project)
        acknowledged_warnings = set(comparison.get('AfgevinkteWaarschuwingen', []))

        matched_table = MatchedPostenTable(
            offer_names=offer_names,
            match_rows=match_rows,
            offer_post_descriptions=offer_post_descriptions,
            acknowledged_warnings=acknowledged_warnings,
        )
        offer_names = matched_table.offer_names
        # per-offer decimals for comparison
        offer_totals_dec: dict[str, Decimal | None] = {}
        for offer_name in offer_names:
            field_prefix = matched_table.offer_field_prefix(offer_name)
            # look at the totals row (last row)
            if matched_table.rows:
                last = matched_table.rows[-1]
                offer_totals_dec[offer_name] = parse_decimal(last.get(f'{field_prefix}_totaal'))
            else:
                offer_totals_dec[offer_name] = None

        comparison_total, comparison_total_label = self.comparison_service.comparison_total_from_json(comparison)

        with ui.row().classes('items-center gap-2 mt-2'):
            ui.button(
                'Add row',
                icon='add',
                on_click=lambda selected_project=project, data=comparison, names=offer_names: self.add_matched_post_row(
                    selected_project,
                    data,
                    names,
                ),
            ).props('dense no-caps')
            ui.button(
                'Copy for Excel',
                icon='content_copy',
                on_click=lambda table=matched_table: self.copy_match_table_to_clipboard(table),
            ).props('dense no-caps')

        matched_tab = tabulator(matched_table.options(), row_key='id').classes('w-full')
        self.render_warning_checklist(project, comparison, match_rows, offer_names)

        def matched_update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            field = column.get('field')
            value = cell.get('value', '')

            row_id = row.get('id')
            if row_id is None:
                return

            if not self.comparison_service.update_matched_cell(
                project,
                comparison,
                match_rows,
                row_id,
                field,
                value,
                offer_names,
            ):
                return

            if self.is_offer_description_field(field):
                return

            matched_table.rows = matched_table.rows_from_matches(match_rows)
            try:
                matched_tab.set_data(matched_table.rows)
            except Exception:
                # best-effort: ignore UI refresh errors
                pass

        matched_tab.on_event('cellEdited', matched_update_cell)

        def matched_delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return

            row = cell.get('row', {})
            row_id = row.get('id')
            if not self.comparison_service.delete_matched_post_row(project, comparison, match_rows, row_id):
                return

            matched_table.rows = matched_table.rows_from_matches(match_rows)
            matched_tab.set_data(matched_table.rows)

        matched_tab.on_event('cellClick', matched_delete_row)

        # If comparison JSON contains an optional total, warn when it does not match
        # the totals row inside the table for any offer.
        if comparison_total is not None:
            mismatched = [
                name for name, val in offer_totals_dec.items()
                if val is not None and abs(val - comparison_total) > Decimal('0.02')
            ]

            if mismatched:
                ui.label(
                    'Warning: the summed subtotals do not match the comparison total for '
                    + ', '.join(mismatched)
                ).classes('text-xs text-red-700 font-semibold mt-2')

    @staticmethod
    def is_offer_description_field(field: str | None) -> bool:
        parts = str(field or '').split('_')
        return len(parts) >= 3 and parts[0] == 'offer' and parts[2] == 'omschrijving'

    def render_warning_checklist(
        self,
        project: Project,
        comparison: dict,
        match_rows: list[dict],
        offer_names: list[str],
    ) -> None:
        warning_items = self.warning_checklist_items(comparison, match_rows, offer_names)
        if not warning_items:
            return

        with ui.expansion('Warnings checklist', icon='fact_check').classes('w-full mt-2'):
            with ui.column().classes('gap-1 w-full'):
                for item in warning_items:
                    checkbox = ui.checkbox(
                        item['label'],
                        value=item['checked'],
                        on_change=lambda event, warning_id=item['id']: self.toggle_warning(
                            project,
                            comparison,
                            warning_id,
                            bool(event.value),
                        ),
                    ).classes('text-sm')
                    checkbox.tooltip(item['tooltip'])

    def warning_checklist_items(
        self,
        comparison: dict,
        match_rows: list[dict],
        offer_names: list[str],
    ) -> list[dict]:
        acknowledged = set(comparison.get('AfgevinkteWaarschuwingen', []))
        items = []

        for row_index, match_row in enumerate(match_rows):
            offers = match_row.get('Offertes', {})
            if not isinstance(offers, dict):
                continue

            row_description = match_row.get('Omschrijving', f'Rij {row_index + 1}')
            for offer_name in offer_names:
                offer = offers.get(offer_name, {})
                if not isinstance(offer, dict):
                    continue

                for warning in warnings_for_offer(match_row, offer):
                    warning_id = MatchedPostenTable.warning_id(row_index, offer_name, warning)
                    items.append({
                        'id': warning_id,
                        'checked': warning_id in acknowledged,
                        'label': f'{row_description} | {offer_name}: {warning}',
                        'tooltip': warning,
                    })

        return items

    def toggle_warning(self, project: Project, comparison: dict, warning_id: str, checked: bool) -> None:
        self.comparison_service.toggle_warning(project, comparison, warning_id, checked)
        self.refresh()

    def copy_match_table_to_clipboard(self, matched_table) -> None:
        clipboard_text = matched_table.to_excel_clipboard_text()
        ui.run_javascript(f'''
            navigator.clipboard.writeText({json.dumps(clipboard_text)})
                .catch(() => {{
                    const textarea = document.createElement('textarea');
                    textarea.value = {json.dumps(clipboard_text)};
                    textarea.style.position = 'fixed';
                    textarea.style.opacity = '0';
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                }});
        ''')
        ui.notify('Copied table for Excel')

    async def match_project_posts(self, project: Project, comparison: dict, button) -> None:
        if not comparison.get('Posten'):
            ui.notify('Add comparison rows before matching')
            return

        if not self.comparison_service.has_offer_results(project):
            ui.notify('No extracted offer results available for this project')
            return

        button.set_text('Matching')
        button.props('loading disable')
        button.update()

        log_action('match_posts_requested', project=project.name)
        try:
            await run.io_bound(self.comparison_service.match_project_posts, project, comparison)
        except Exception as error:
            log_action('match_posts_failed', project=project.name, error=str(error))
            self.notify_safe(f'Could not match posts: {error}')
            self.refresh()
            return

        log_action('match_posts_finished', project=project.name)
        self.notify_safe('Matched posts')
        self.refresh()

    def recalculate_project_posts(self, project: Project, comparison: dict) -> None:
        if not comparison.get('MatchedPosten'):
            ui.notify('Match posts before recalculating')
            return

        log_action('recalculate_posts', project=project.name)
        self.comparison_service.recalculate_project_posts(project, comparison)
        self.notify_safe('Recalculated posts')
        self.refresh()

    def add_matched_post_row(self, project: Project, comparison: dict, offer_names: list[str]) -> None:
        self.comparison_service.add_matched_post_row(project, comparison, offer_names)
        self.notify_safe('Added matched row')
        self.refresh()
