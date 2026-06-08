from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from decimal import Decimal
import json

from nicegui import ui, run
from nicegui_tabulator import tabulator

from .subpage import SubPage

from services.comparison_matcher import ComparisonMatcher
from services.folder_handler import FolderHandler
from services.project import Project
from .tabulator_table import TabulatorTable
from interface.page_state import MainPageState


class ComparisonRowsTable(TabulatorTable):
    fields = ['Omschrijving', 'Aantal', 'Eenheid']

    def __init__(self, comparison: dict) -> None:
        self.comparison = comparison
        super().__init__(
            rows=self.rows_from_comparison(),
            columns=[
                self.text_column('Omschrijving', 'Omschrijving', editable=True),
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
        )

    def rows_from_comparison(self) -> list[dict]:
        return [
            {'id': index, **row}
            for index, row in enumerate(self.comparison.get('Posten', []))
        ]

    def add_row(self) -> dict:
        row = {
            'id': len(self.comparison.setdefault('Posten', [])),
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
        }
        self.comparison['Posten'].append({field: row[field] for field in self.fields})
        self.rows.append(row)
        self.clear_matches()
        return row

    def update_cell(self, row_id: int | None, field: str | None, value: str) -> None:
        if field not in self.fields:
            return

        posten = self.comparison.setdefault('Posten', [])
        if row_id is None:
            return
        if row_id >= len(posten):
            return

        posten[row_id][field] = value
        self.clear_matches()

    def delete_row(self, row_id: int | None) -> None:
        posten = self.comparison.setdefault('Posten', [])
        if row_id is None:
            return
        if row_id >= len(posten):
            return

        posten.pop(row_id)
        self.clear_matches()
        self.rows = self.rows_from_comparison()

    def clear_matches(self) -> None:
        self.comparison.pop('MatchedPosten', None)
        self.comparison.pop('Matches', None)


class MatchedPostenTable(TabulatorTable):
    def __init__(self, *, offer_names: list[str], match_rows: list[dict]) -> None:
        self.offer_names = self.offer_names_by_total(offer_names, match_rows)
        self.offer_prefixes = {offer_name: f'offer_{index}' for index, offer_name in enumerate(self.offer_names)}
        super().__init__(
            rows=self.rows_from_matches(match_rows),
            columns=self.columns_from_offers(),
            layout='fitDataStretch',
            reactive=False,
        )

    def columns_from_offers(self) -> list[dict]:
        columns = [
            self.text_column('Omschrijving', 'Omschrijving', width=260),
            self.text_column('Aantal', 'Aantal', editable=True, width=100),
            self.text_column('Eenheid', 'Eenheid', editable=True, width=100),
        ]

        # Only show the price and total columns per offer in the UI.
        for offer_name in self.offer_names:
            field_prefix = self.offer_field_prefix(offer_name)
            columns.extend([
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
                element.title = tooltip;

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

    @classmethod
    def offer_names_by_total(cls, offer_names: list[str], match_rows: list[dict]) -> list[str]:
        totals = cls.total_decimals_by_offer(offer_names, match_rows)
        return sorted(
            offer_names,
            key=lambda offer_name: (
                totals.get(offer_name) is None,
                totals.get(offer_name, Decimal('0')),
                offer_name.lower(),
            ),
        )

    def rows_from_matches(self, match_rows: list[dict]) -> list[dict]:
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
                # Keep the matched description in the data for reference, but do not
                # show it in the table columns.
                row[f'{field_prefix}_omschrijving'] = offer.get('Gematchte omschrijving', offer.get('Omschrijving', 'ONBEKEND'))

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
                    dec = ComparisonMatcher.parse_decimal(val)
                    if dec is None:
                        return 'ONBEKEND'
                    return self.format_money(dec)

                row[f'{field_prefix}_prijs'] = _fmt_money(raw_een) if raw_een not in (None, '') else 'ONBEKEND'
                row[f'{field_prefix}_totaal'] = _fmt_money(raw_tot) if raw_tot not in (None, '') else 'ONBEKEND'
                warning = ComparisonMatcher.warning_for_offer(match_row, offer)
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
        totals_row = {
            'id': len(rows),
            'Omschrijving': 'Totaal',
            'Aantal': '',
            'Eenheid': '',
        }

        for offer_name in self.offer_names:
            field_prefix = self.offer_field_prefix(offer_name)
            # sum known totals across all rows
            total_sum = Decimal('0')
            has_value = False
            for r in rows:
                value = ComparisonMatcher.parse_decimal(r.get(f'{field_prefix}_totaal'))
                if value is None:
                    # if a row has no totaal, try to compute from aantal * prijs
                    aantal = ComparisonMatcher.parse_decimal(r.get('Aantal'))
                    prijs = ComparisonMatcher.parse_decimal(r.get(f'{field_prefix}_prijs'))
                    if aantal is not None and prijs is not None:
                        value = aantal * prijs

                if value is None:
                    continue

                total_sum += value
                has_value = True

            totals_row[f'{field_prefix}_omschrijving'] = 'Totaal'
            totals_row[f'{field_prefix}_prijs'] = ''
            totals_row[f'{field_prefix}_totaal'] = self.format_money(total_sum) if has_value else 'ONBEKEND'
            totals_row[f'{field_prefix}_verschil'] = ''
            totals_row[f'{field_prefix}_prijs_tooltip'] = 'Totaal'
            totals_row[f'{field_prefix}_totaal_tooltip'] = 'Totaal'
            totals_row[f'{field_prefix}_verschil_tooltip'] = 'Totaal'

        self.add_difference_percentages(totals_row, self.offer_names)
        rows.append(totals_row)

        return rows

    @staticmethod
    def tooltip_for_offer(offer: dict, matched_description: str, warning: str) -> str:
        matched_posts = ComparisonMatcher.matched_post_descriptions(offer)
        categories = ComparisonMatcher.matched_categories(offer)
        if len(matched_posts) > 1:
            tooltip = 'Gematchte posten:\n' + '\n'.join(f'- {description}' for description in matched_posts)
        else:
            tooltip = f'Gematchte omschrijving: {str(matched_description or "ONBEKEND").strip() or "ONBEKEND"}'

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

            amount = ComparisonMatcher.parse_decimal(match_row.get('Aantal'))
            offers = match_row.get('Offertes', {})
            if not isinstance(offers, dict):
                continue

            for offer_name in offer_names:
                offer = offers.get(offer_name, {})
                if not isinstance(offer, dict):
                    continue

                value = ComparisonMatcher.parse_decimal(offer.get('Totaalbedrag'))
                if value is None and amount is not None:
                    unit_price = ComparisonMatcher.parse_decimal(offer.get('Eenheidsprijs'))
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
            index: ComparisonMatcher.parse_decimal(row.get(f'offer_{index}_totaal'))
            for index, _offer_name in enumerate(offer_names)
        }
        known_totals = [total for total in totals.values() if total is not None]
        if not known_totals:
            return

        lowest_total = min(known_totals)
        for index, total in totals.items():
            field_prefix = f'offer_{index}'
            if total is None:
                row[f'{field_prefix}_verschil'] = 'ONBEKEND'
                continue

            row[f'{field_prefix}_verschil'] = cls.format_percentage_difference(total, lowest_total)

    @staticmethod
    def format_percentage_difference(total: Decimal, lowest_total: Decimal) -> str:
        if lowest_total == 0:
            return '0,0%' if total == 0 else 'ONBEKEND'

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
                row_total = ComparisonMatcher.parse_decimal(row.get(f'{field_prefix}_totaal'))
                if row_total is None:
                    continue

                total += row_total
                has_known_value = True

            if has_known_value:
                totals[offer_name] = total

        return {
            offer_name: self.format_money(total) if offer_name in totals else 'ONBEKEND'
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
        comparison = project.load_comparison()
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
            match_button = ui.button('Match Posten', icon='eva-bulb-outline').props('dense no-caps')
            recalculate_button = ui.button(
                'Recalculate',
                icon='calculate',
                on_click=lambda selected_project=project, data=comparison: self.recalculate_project_posts(
                    selected_project,
                    data,
                ),
            ).props('dense no-caps')

            if not comparison.get('MatchedPosten'):
                recalculate_button.props('disable')

            async def request_match(_event, selected_project=project, data=comparison, button=match_button):
                await self.match_project_posts(selected_project, data, button)
                self.refresh()

            match_button.on('click', request_match)

    def input_table(self, project: Project, comparison: dict) -> None:

        comparison_table = ComparisonRowsTable(comparison)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten voor Vergelijking').classes('text-lg font-bold')
            ui.button(
                'Add row',
                icon='add',
                on_click=lambda: self.add_comparison_row(project, comparison, comparison_table),
            ).props('dense no-caps size=sm')

        comparison_tabulator = tabulator(comparison_table.options(), row_key='id').classes('w-full')

        def update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            comparison_table.update_cell(row.get('id'), column.get('field'), cell.get('value', ''))
            project.save_comparison(comparison)

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return

            row = cell.get('row', {})
            comparison_table.delete_row(row.get('id'))
            project.save_comparison(comparison)
            comparison_tabulator.set_data(comparison_table.rows)

        comparison_tabulator.on_event('cellEdited', update_cell)
        comparison_tabulator.on_event('cellClick', delete_row)

    def update_comparison_value(self, project: Project, comparison: dict, row_index: int, field: str, value: str) -> None:
        comparison['Posten'][row_index][field] = value
        comparison.pop('MatchedPosten', None)
        comparison.pop('Matches', None)
        project.save_comparison(comparison)

    def add_comparison_row(
        self,
        project: Project,
        comparison: dict,
        comparison_table=None,
    ) -> None:
        if comparison_table is None:
            comparison.setdefault('Posten', [])
            comparison['Posten'].append({
                'Omschrijving': '',
                'Aantal': '',
                'Eenheid': '',
            })
            comparison.pop('MatchedPosten', None)
            comparison.pop('Matches', None)
        else:
            comparison_table.add_row()

        project.save_comparison(comparison)
        self.refresh()

    def delete_comparison_row(self, project: Project, comparison: dict, row_index: int) -> None:
        if 'Posten' not in comparison or row_index >= len(comparison['Posten']):
            ui.notify('Row no longer exists')
            self.refresh()
            return

        comparison['Posten'].pop(row_index)
        comparison.pop('MatchedPosten', None)
        comparison.pop('Matches', None)
        project.save_comparison(comparison)
        self.refresh()

    def render_side_by_side_match_table(self, project: Project, comparison: dict, match_rows: list[dict]) -> None:
        offer_names = [offer['Bestand'] for offer in self.matcher.project_offer_results(project)]

        matched_table = MatchedPostenTable(offer_names=offer_names, match_rows=match_rows)
        offer_names = matched_table.offer_names
        # per-offer decimals for comparison
        offer_totals_dec: dict[str, Decimal | None] = {}
        for offer_name in offer_names:
            field_prefix = matched_table.offer_field_prefix(offer_name)
            # look at the totals row (last row)
            if matched_table.rows:
                last = matched_table.rows[-1]
                offer_totals_dec[offer_name] = ComparisonMatcher.parse_decimal(last.get(f'{field_prefix}_totaal'))
            else:
                offer_totals_dec[offer_name] = None

        comparison_total, comparison_total_label = self.comparison_total_from_json(comparison)

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

        def matched_update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            field = column.get('field')
            value = cell.get('value', '')

            row_id = row.get('id')
            if row_id is None:
                return

            # Ensure the underlying comparison structure exists
            comparison.setdefault('MatchedPosten', match_rows)
            if row_id >= len(match_rows):
                return

            matched_row = match_rows[row_id]

            # Handle editing of top-level fields
            if field in ('Aantal', 'Eenheid', 'Omschrijving'):
                matched_row[field] = value
            elif isinstance(field, str) and field.startswith('offer_'):
                parts = field.split('_')
                if len(parts) < 3:
                    return
                try:
                    offer_index = int(parts[1])
                except ValueError:
                    return
                suffix = parts[2]
                if offer_index < 0 or offer_index >= len(offer_names):
                    return

                offer_name = offer_names[offer_index]
                offers = matched_row.setdefault('Offertes', {})
                offer_entry = offers.setdefault(offer_name, {})

                if suffix == 'prijs':
                    offer_entry['Eenheidsprijs'] = value
                elif suffix == 'totaal':
                    offer_entry['Totaalbedrag'] = value

            # Persist changes and recompute the displayed rows (including totals row)
            project.save_comparison(comparison)
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
            if row_id is None or row_id >= len(match_rows):
                return

            match_rows.pop(row_id)
            comparison['MatchedPosten'] = match_rows
            comparison.pop('Matches', None)
            project.save_comparison(comparison)
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
    def comparison_total_from_json(comparison: dict) -> tuple[Decimal | None, str]:
        for key in ('Totaalprijs exc. BTW', 'Totaalprijs inc. BTW', 'Totaalbedrag', 'Totaal'):
            if key not in comparison:
                continue

            total = ComparisonMatcher.parse_decimal(comparison.get(key))
            if total is not None:
                return total, key

        return None, 'Totaal'

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

        if not self.matcher.project_offer_results(project):
            ui.notify('No extracted offer results available for this project')
            return

        button.set_text('Matching')
        button.props('loading disable')
        button.update()

        try:
            match_result = await run.io_bound(self.matcher.match_comparison_posts, project, comparison)
        except Exception as error:
            self.notify_safe(f'Could not match posts: {error}')
            return

        comparison['MatchedPosten'] = self.matcher.normalize_matched_posts(project, comparison, match_result)
        comparison.pop('Matches', None)
        project.save_comparison(comparison)
        self.notify_safe('Matched posts')
        self.refresh()

    def recalculate_project_posts(self, project: Project, comparison: dict) -> None:
        if not comparison.get('MatchedPosten'):
            ui.notify('Match posts before recalculating')
            return

        comparison['MatchedPosten'] = self.matcher.recalculate_matched_posts(comparison, project)
        comparison.pop('Matches', None)
        project.save_comparison(comparison)
        self.notify_safe('Recalculated posts')
        self.refresh()

    def add_matched_post_row(self, project: Project, comparison: dict, offer_names: list[str]) -> None:
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
                    'Gematchte omschrijving': 'ONBEKEND',
                    'Gematchte eenheid': 'ONBEKEND',
                    'Eenheidsprijs': 'ONBEKEND',
                    'Totaalbedrag': 'ONBEKEND',
                    'Overeenkomst': '',
                }
                for offer_name in offer_names
            },
        })
        comparison.pop('Matches', None)
        project.save_comparison(comparison)
        self.notify_safe('Added matched row')
        self.refresh()
