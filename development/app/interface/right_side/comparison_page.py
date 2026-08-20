from __future__ import annotations

import copy
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
from interface.left_drawer.dialogs import render_status_fields
from interface.left_drawer.utils import format_elapsed_seconds
from matching.match_fields import matched_categories, matched_post_descriptions
from .subpage import SubPage

from services.comparison_matcher import ComparisonMatcher
from services.folder_handler import FolderHandler
from services.project import Project
from .tabulator_table import TabulatorTable
from interface.page_state import MainPageState, PendingUndo
from interface.theme import WARNING_BG, WARNING_TEXT
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
        omschrijving_column = self.text_column('Omschrijving', 'Omschrijving', width=220, multiline=True)
        omschrijving_column['frozen'] = True
        columns = [
            omschrijving_column,
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

                element.style.backgroundColor = warning ? {json.dumps(WARNING_BG)} : '';
                element.style.color = warning ? {json.dumps(WARNING_TEXT)} : '';
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
        lines = []
        for row in self.rows:
            cells = ['', row.get('Omschrijving', ''), row.get('Aantal', ''), row.get('Eenheid', '')]
            for offer_name in self.offer_names:
                field_prefix = self.offer_field_prefix(offer_name)
                cells += ['', row.get(f'{field_prefix}_prijs', ''), row.get(f'{field_prefix}_totaal', '')]
            lines.append('\t\t'.join(self.clean_clipboard_cell(cell) for cell in cells))

        return '\n'.join(lines)

    @staticmethod
    def clean_clipboard_cell(value) -> str:
        if isinstance(value, list):
            return ', '.join(' '.join(str(item or '').split()) for item in value)
        return ' '.join(str(value or '').split())

    @staticmethod
    def spiegel_line(values) -> str:
        CELL_SEP = '\t'
        result = CELL_SEP * 2 + values[0] + CELL_SEP * 2 + \
        values[1]
        

        return result
        




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
            ui.label('Geen project geselecteerd').classes('text-gray-500')
            return

        # Show title
        ui.label(f'Vergelijking: {project.name}').classes('text-xl font-bold')

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
                'Bezig met matchen' if is_running else 'Posten matchen',
                icon='eva-bulb-outline',
            ).props('dense no-caps')
            recalculate_button = ui.button(
                'Bezig met herberekenen' if is_running and status_step == 'recalculating_posts' else 'Herberekenen',
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

                status_text = status_message or status_step or 'Bezig'
                elapsed = format_elapsed_seconds(status.get('started_at') if status else None)
                if elapsed:
                    status_text += f' ({elapsed})'
                ui.label(status_text).classes('text-xs text-gray-600')

            if failed:
                warning_icon = ui.icon('warning').classes('text-red-700')
                warning_icon.tooltip(status_error or status_message or 'Vergelijking mislukt')
            elif is_stale:
                warning_icon = ui.icon('warning').classes('text-orange-700')
                warning_icon.tooltip('Vergelijkingsstatus is verouderd. U kunt opnieuw matchen of herberekenen.')

            if status is not None:
                ui.button(icon='info', on_click=lambda data=status: self.show_status_dialog(project, data)) \
                    .props('flat dense round size=sm').tooltip('Matchstatus bekijken')

            async def request_match(_event, selected_project=project, data=comparison, button=match_button):
                await self.match_project_posts(selected_project, data, button)

            if not is_running:
                match_button.on('click', request_match)

    @staticmethod
    def show_status_dialog(project: Project, status: dict | None) -> None:
        with ui.dialog() as dialog, ui.card().classes('gap-2 min-w-[20rem]'):
            ui.label(f'Matchstatus voor {project.name}').classes('font-medium')
            render_status_fields(status)

            with ui.row().classes('justify-end w-full'):
                ui.button('Sluiten', on_click=dialog.close).props('flat dense no-caps size=sm')

        dialog.open()

    def input_table(self, project: Project, comparison: dict) -> None:
        comparison_table = ComparisonRowsTable(comparison)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten voor Vergelijking').classes('text-lg font-bold')
            ui.button(
                'Regel toevoegen',
                icon='add',
                on_click=lambda: self.add_comparison_row(project, comparison),
            ).props('dense no-caps size=sm')
            ui.button(
                'Vul vanuit offerte',
                icon='playlist_add',
                on_click=lambda: self.open_seed_from_offer_dialog(project, comparison),
            ).props('dense no-caps size=sm outline')
            ui.button(
                'Plakken vanuit Excel',
                icon='content_paste',
                on_click=lambda: self.open_paste_rows_dialog(project, comparison),
            ).props('dense no-caps size=sm outline')

        with ui.element('div').classes('w-full overflow-x-auto'):
            comparison_tabulator = tabulator(comparison_table.options(), row_key='id').classes('w-full')

        def update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            try:
                self.comparison_service.update_comparison_row(
                    project, comparison, row.get('id'), column.get('field'), cell.get('value', '')
                )
            except Exception as error:  # noqa: BLE001 - surface any save failure to the user
                ui.notify(f'Kon niet opslaan: {error}', type='negative')
                return

            ui.notify('Opgeslagen', type='positive')

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return
            row = cell.get('row', {})
            self.delete_comparison_row(project, comparison, row.get('id'))

        comparison_tabulator.on_event('cellEdited', update_cell)
        comparison_tabulator.on_event('cellClick', delete_row)

    def add_comparison_row(self, project: Project, comparison: dict) -> None:
        self.comparison_service.add_comparison_row(project, comparison)
        self.refresh()

    def open_seed_from_offer_dialog(self, project: Project, comparison: dict) -> None:
        offer_names = self.comparison_service.offer_names(project)
        if not offer_names:
            ui.notify('Geen offertes met extractieresultaat gevonden.', type='warning')
            return

        with ui.dialog() as dialog, ui.card().classes('gap-2 min-w-[20rem]'):
            ui.label('Vergelijkingsregels vullen vanuit offerte').classes('font-medium')
            ui.label(
                'Voegt de posten van de gekozen offerte toe als nieuwe vergelijkingsregels, '
                'met de omschrijving zoals die uit die offerte is gehaald.'
            ).classes('text-xs text-gray-600')
            offer_select = ui.select(offer_names, value=offer_names[0], label='Offerte') \
                .classes('w-full').props('dense outlined')

            with ui.row().classes('justify-end w-full gap-2'):
                ui.button('Annuleren', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button(
                    'Vul in',
                    on_click=lambda: self.seed_from_offer(project, comparison, offer_select.value, dialog),
                ).props('dense no-caps size=sm')

        dialog.open()

    def seed_from_offer(self, project: Project, comparison: dict, offer_name: str, dialog) -> None:
        added = self.comparison_service.seed_comparison_from_offer(project, comparison, offer_name)
        dialog.close()
        if added:
            ui.notify(f'{added} regel(s) toegevoegd vanuit {offer_name}.', type='positive')
        else:
            ui.notify(f'Geen bruikbare posten gevonden in {offer_name}.', type='warning')
        self.refresh()

    def open_paste_rows_dialog(self, project: Project, comparison: dict) -> None:
        with ui.dialog() as dialog, ui.card().classes('gap-2 min-w-[28rem]'):
            ui.label('Vergelijkingsregels plakken').classes('font-medium')
            ui.label(
                'Plak rijen vanuit Excel: Omschrijving, Aantal en Eenheid per kolom, '
                'zonder kopregel. Een regel zonder tabs wordt als omschrijving gebruikt.'
            ).classes('text-xs text-gray-600')
            paste_area = ui.textarea(placeholder='Omschrijving\tAantal\tEenheid') \
                .classes('w-full font-mono text-xs').props('outlined rows=10')

            with ui.row().classes('justify-end w-full gap-2'):
                ui.button('Annuleren', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button(
                    'Toevoegen',
                    on_click=lambda: self.paste_comparison_rows(project, comparison, paste_area.value, dialog),
                ).props('dense no-caps size=sm')

        dialog.open()

    def paste_comparison_rows(self, project: Project, comparison: dict, text: str, dialog) -> None:
        added = self.comparison_service.add_comparison_rows_from_text(project, comparison, text or '')
        dialog.close()
        if added:
            ui.notify(f'{added} regel(s) toegevoegd.', type='positive')
        else:
            ui.notify('Geen bruikbare rijen gevonden om te plakken.', type='warning')
        self.refresh()

    def delete_comparison_row(self, project: Project, comparison: dict, row_index: int | None) -> None:
        posten = comparison.get('Posten', [])
        if row_index is None or row_index < 0 or row_index >= len(posten):
            ui.notify('Regel bestaat niet meer', type='negative')
            self.refresh()
            return

        removed_row = copy.deepcopy(posten[row_index])
        removed_matched = copy.deepcopy(comparison.get('MatchedPosten'))
        removed_matches = copy.deepcopy(comparison.get('Matches'))

        if not self.comparison_service.delete_comparison_row(project, comparison, row_index):
            ui.notify('Regel bestaat niet meer', type='negative')
            self.refresh()
            return

        label = str(removed_row.get('Omschrijving') or '').strip() or f'row {row_index + 1}'

        def restore(
            project=project,
            row_index=row_index,
            removed_row=removed_row,
            removed_matched=removed_matched,
            removed_matches=removed_matches,
        ) -> None:
            fresh_comparison = self.comparison_service.load_comparison(project)
            fresh_posten = fresh_comparison.setdefault('Posten', [])
            fresh_posten.insert(min(row_index, len(fresh_posten)), removed_row)
            if removed_matched is not None:
                fresh_comparison['MatchedPosten'] = removed_matched
            if removed_matches is not None:
                fresh_comparison['Matches'] = removed_matches
            self.comparison_service.save_comparison(project, fresh_comparison)

        self.state.pending_undo = PendingUndo(label=f'"{label}" verwijderd', restore=restore)
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
                'Regel toevoegen',
                icon='add',
                on_click=lambda selected_project=project, data=comparison, names=offer_names: self.add_matched_post_row(
                    selected_project,
                    data,
                    names,
                ),
            ).props('dense no-caps')
            ui.button(
                'Kopiëren voor Excel',
                icon='content_copy',
                on_click=lambda: ui.notify('Tabel gekopieerd voor Excel'),
            ).props('dense no-caps').on(
                'click',
                js_handler=self.copy_match_table_js_handler(matched_table),
            )

        with ui.element('div').classes('w-full overflow-x-auto'):
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

            try:
                updated = self.comparison_service.update_matched_cell(
                    project,
                    comparison,
                    match_rows,
                    row_id,
                    field,
                    value,
                    offer_names,
                )
            except Exception as error:  # noqa: BLE001 - surface any save failure to the user
                ui.notify(f'Kon niet opslaan: {error}', type='negative')
                return

            if not updated:
                return

            ui.notify('Opgeslagen', type='positive')
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
            self.delete_matched_post_row(project, comparison, match_rows, row.get('id'))

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
                    'Waarschuwing: de opgetelde subtotalen komen niet overeen met het vergelijkingstotaal voor '
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

        open_count = sum(1 for item in warning_items if not item['checked'])
        with ui.expansion(
            f'Waarschuwingenchecklist ({open_count}/{len(warning_items)} open)',
            icon='fact_check',
        ).classes('w-full mt-2'):
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

    def copy_match_table_js_handler(self, matched_table) -> str:
        # Must run entirely client-side (no server round-trip) so the clipboard write stays
        # inside the same synchronous user gesture as the click — Safari refuses clipboard
        # access otherwise, even though Chrome/Firefox/Edge tolerate the round-trip delay.
        clipboard_text = matched_table.to_excel_clipboard_text()
        return f'''() => {{
            function copyWithTextarea() {{
                const textarea = document.createElement('textarea');
                textarea.value = {json.dumps(clipboard_text)};
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
            }}
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText({json.dumps(clipboard_text)}).catch(copyWithTextarea);
            }} else {{
                copyWithTextarea();
            }}
        }}'''

    async def match_project_posts(self, project: Project, comparison: dict, button) -> None:
        if not comparison.get('Posten'):
            ui.notify('Voeg vergelijkingsregels toe voordat u matcht')
            return

        if not self.comparison_service.has_offer_results(project):
            ui.notify('Geen geëxtraheerde offerteresultaten beschikbaar voor dit project')
            return

        button.set_text('Bezig met matchen')
        button.props('loading disable')
        button.update()

        log_action('match_posts_requested', project=project.name)
        try:
            await run.io_bound(self.comparison_service.match_project_posts, project, comparison)
        except Exception as error:
            log_action('match_posts_failed', project=project.name, error=str(error))
            self.notify_safe(f'Kon posten niet matchen: {error}')
            self.refresh()
            return

        log_action('match_posts_finished', project=project.name)
        self.notify_safe('Posten gematcht')
        self.refresh()

    def recalculate_project_posts(self, project: Project, comparison: dict) -> None:
        if not comparison.get('MatchedPosten'):
            ui.notify('Match eerst de posten voordat u herberekent')
            return

        log_action('recalculate_posts', project=project.name)
        self.comparison_service.recalculate_project_posts(project, comparison)
        self.notify_safe('Posten herberekend')
        self.refresh()

    def add_matched_post_row(self, project: Project, comparison: dict, offer_names: list[str]) -> None:
        self.comparison_service.add_matched_post_row(project, comparison, offer_names)
        self.notify_safe('Gematchte regel toegevoegd')
        self.refresh()

    def delete_matched_post_row(
        self,
        project: Project,
        comparison: dict,
        match_rows: list[dict],
        row_id: int | None,
    ) -> None:
        if row_id is None or row_id < 0 or row_id >= len(match_rows):
            return

        removed_row = copy.deepcopy(match_rows[row_id])

        if not self.comparison_service.delete_matched_post_row(project, comparison, match_rows, row_id):
            return

        label = str(removed_row.get('Omschrijving') or '').strip() or f'row {row_id + 1}'

        def restore(project=project, row_id=row_id, removed_row=removed_row) -> None:
            fresh_comparison = self.comparison_service.load_comparison(project)
            matched = fresh_comparison.get('MatchedPosten')
            if not isinstance(matched, list):
                return
            matched.insert(min(row_id, len(matched)), removed_row)
            fresh_comparison['MatchedPosten'] = matched
            self.comparison_service.save_comparison(project, fresh_comparison)

        self.state.pending_undo = PendingUndo(label=f'"{label}" verwijderd', restore=restore)
        self.refresh()
