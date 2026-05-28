from pathlib import Path
from urllib.parse import quote
from collections.abc import Callable
from decimal import Decimal
import json

from abc import ABC, abstractmethod

from nicegui_tabulator import tabulator
from nicegui import run, ui

from services.comparison_matcher import ComparisonMatcher
from ui.editable_table_helper import render_editable_summary
from services.folder_handler import FolderHandler
from ui.page_state import MainPageState


class SubPage(ABC):
    def __init__(
        self,
        state: MainPageState,
        folder_handler: FolderHandler,
        pdf_dir: Path,
        refresh: Callable[[], None] | None = None,
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.pdf_dir = pdf_dir
        self.refresh_callback = refresh

        self.container = None

    @abstractmethod
    def render(self) -> None:
       pass

    @abstractmethod
    def show(self) -> None:
        pass

    def save_result(self, file: Path, result: dict) -> None:
        self.folder_handler.save_result(file, result)

    def refresh(self) -> None:
        if self.refresh_callback is not None:
            self.refresh_callback()

    @staticmethod
    def notify_safe(message: str) -> None:
        try:
            ui.notify(message)
        except RuntimeError:
            print(message)


class TabulatorTable:
    def __init__(
        self,
        *,
        rows: list[dict],
        columns: list[dict],
        layout: str = 'fitColumns',
        reactive: bool = True,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.layout = layout
        self.reactive = reactive

    def options(self) -> dict:
        return {
            'data': self.rows,
            'layout': self.layout,
            'reactiveData': self.reactive,
            'columns': self.columns,
        }

    @staticmethod
    def text_column(
        title: str,
        field: str,
        *,
        editable: bool = False,
        width: int | None = None,
    ) -> dict:
        column = {
            'title': title,
            'field': field,
        }
        if editable:
            column['editor'] = 'input'
        if width is not None:
            column['width'] = width

        return column


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
        self.offer_names = offer_names
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
                self.text_column(f'{offer_name} prijs', f'{field_prefix}_prijs', editable=True, width=120),
                self.text_column(f'{offer_name} totaal', f'{field_prefix}_totaal', editable=True, width=130),
            ])

        return columns

    def offer_field_prefix(self, offer_name: str) -> str:
        return f'offer_{self.offer_names.index(offer_name)}'

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
                totaalprijs = offer.get('Totaalbedrag') or ''
                # Prefer the explicitly matched unit, fall back to the offer's own unit
                gematchte_eenheid = (
                    (offer.get('Gematchte eenheid') or '')
                    or (offer.get('Eenheid') or '')
                ).strip().casefold()

                # If the supplier marked the post as a 'post' unit (i.e. a total
                # rather than a per-unit price), move the unit price into the
                # totaal column when a total isn't already provided.
                if 'post' in gematchte_eenheid and eenheidsprijs and not totaalprijs:
                    # move the value to totaal
                    totaalprijs = eenheidsprijs
                    eenheidsprijs = ''

                row[f'{field_prefix}_prijs'] = eenheidsprijs if eenheidsprijs not in (None, '') else 'ONBEKEND'
                row[f'{field_prefix}_totaal'] = totaalprijs if totaalprijs not in (None, '') else 'ONBEKEND'
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

        rows.append(totals_row)

        return rows

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

    @staticmethod
    def format_money(value: Decimal | None) -> str:
        if value is None:
            return 'ONBEKEND'

        rounded = value.quantize(Decimal('0.01'))
        text = f'{rounded:,.2f}'
        return f'€ {text.replace(",", "_").replace(".", ",").replace("_", ".")}'

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


class OfferRowsTable(TabulatorTable):
    fields = ['Omschrijving', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag']

    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__(
            rows=self.rows_from_result(),
            columns=[
                self.text_column('Omschrijving', 'Omschrijving', editable=True),
                self.text_column('Aantal', 'Aantal', editable=True, width=120),
                self.text_column('Eenheid', 'Eenheid', editable=True, width=120),
                self.text_column('Eenheidsprijs', 'Eenheidsprijs', editable=True, width=140),
                self.text_column('Totaalbedrag', 'Totaalbedrag', editable=True, width=140),
                {
                    'title': '',
                    'field': '__delete__',
                    'width': 52,
                    'headerSort': False,
                    'hozAlign': 'center',
                    ':formatter': "function(){ return 'x'; }",
                },
            ],
            layout='fitData',
            reactive=True,
        )

    def rows_from_result(self) -> list[dict]:
        return [
            {'id': index, **row}
            for index, row in enumerate(self.result.get('Posten', []))
        ]

    def add_row(self) -> dict:
        row = {
            'id': len(self.result.setdefault('Posten', [])),
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
            'Eenheidsprijs': '',
            'Totaalbedrag': '',
        }
        self.result['Posten'].append({field: row[field] for field in self.fields})
        self.rows.append(row)
        return row

    def update_cell(self, row_id: int | None, field: str | None, value: str) -> None:
        if field not in self.fields:
            return

        posten = self.result.setdefault('Posten', [])
        if row_id is None:
            return
        if row_id >= len(posten):
            return

        posten[row_id][field] = value

    def delete_row(self, row_id: int | None) -> None:
        posten = self.result.setdefault('Posten', [])
        if row_id is None:
            return
        if row_id >= len(posten):
            return

        posten.pop(row_id)
        self.rows = self.rows_from_result()


class ComparisonPage(SubPage):
    def __init__(
        self,
        state: MainPageState,
        folder_handler: FolderHandler,
        pdf_dir: Path,
        refresh: Callable[[], None] | None = None,
        matcher: ComparisonMatcher | None = None,
    ) -> None:
        super().__init__(state, folder_handler, pdf_dir, refresh)
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
        comparison = self.folder_handler.load_comparison(project)
        self.input_table(project, comparison)

        # Show match button
        self.match_button(project, comparison)

        # Show comparison
        match_rows = comparison.get('MatchedPosten', [])
        if not match_rows:
            return

        # Show match table
        ui.label('Matched posten').classes('text-lg font-bold mt-4')
        self.render_side_by_side_match_table(project, comparison, match_rows)

    def match_button(self, project, comparison) -> None:
        with ui.row().classes('items-center gap-2 mt-4'):
            match_button = ui.button('Match posten', icon='auto_fix_high').props('dense no-caps')

            async def request_match(_event, selected_project=project, data=comparison, button=match_button):
                await self.match_project_posts(selected_project, data, button)
                self.refresh()

            match_button.on('click', request_match)

    def input_table(self, project, comparison) -> None:
        comparison_table = ComparisonRowsTable(comparison)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Comparison rows').classes('text-lg font-bold')
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
            self.folder_handler.save_comparison(project, comparison)

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return

            row = cell.get('row', {})
            comparison_table.delete_row(row.get('id'))
            self.folder_handler.save_comparison(project, comparison)
            comparison_tabulator.set_data(comparison_table.rows)

        comparison_tabulator.on_event('cellEdited', update_cell)
        comparison_tabulator.on_event('cellClick', delete_row)

    def update_comparison_value(self, project: Path, comparison: dict, row_index: int, field: str, value: str) -> None:
        comparison['Posten'][row_index][field] = value
        comparison.pop('MatchedPosten', None)
        comparison.pop('Matches', None)
        self.folder_handler.save_comparison(project, comparison)

    def add_comparison_row(
        self,
        project: Path,
        comparison: dict,
        comparison_table: ComparisonRowsTable | None = None,
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

        self.folder_handler.save_comparison(project, comparison)
        self.refresh()

    def delete_comparison_row(self, project: Path, comparison: dict, row_index: int) -> None:
        if 'Posten' not in comparison or row_index >= len(comparison['Posten']):
            ui.notify('Row no longer exists')
            self.refresh()
            return

        comparison['Posten'].pop(row_index)
        comparison.pop('MatchedPosten', None)
        comparison.pop('Matches', None)
        self.folder_handler.save_comparison(project, comparison)
        self.refresh()

    def render_side_by_side_match_table(self, project: Path, comparison: dict, match_rows: list[dict]) -> None:
        offer_names = [offer['Bestand'] for offer in self.matcher.project_offer_results(project)]
        matched_table = MatchedPostenTable(offer_names=offer_names, match_rows=match_rows)
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
            self.folder_handler.save_comparison(project, comparison)
            matched_table.rows = matched_table.rows_from_matches(match_rows)
            try:
                matched_tab.set_data(matched_table.rows)
            except Exception:
                # best-effort: ignore UI refresh errors
                pass

        matched_tab.on_event('cellEdited', matched_update_cell)

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

    def copy_match_table_to_clipboard(self, matched_table: MatchedPostenTable) -> None:
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

    async def match_project_posts(self, project: Path, comparison: dict, button) -> None:
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
        self.folder_handler.save_comparison(project, comparison)
        self.notify_safe('Matched posts')
        self.refresh()


class OfferPage(SubPage):

    def render(self) -> None:
        self.show()

    def opened_file(self):
        if self.state.opened_file is None:
            ui.label('No file selected')
            return

        relative_pdf_path = self.state.opened_file.relative_to(self.pdf_dir).as_posix()
        pdf_url = f'/pdfs/{quote(relative_pdf_path, safe="/")}'

        ui.html(f'''
            <iframe
                src="{pdf_url}"
                style="width: 100%; height: 100%; border: none;"
            ></iframe>
        ''', sanitize=False).classes('w-full h-full overflow-hidden')

    def opened_file_result(self):
        if self.state.opened_file is None:
            return

        result = self.folder_handler.load_result(self.state.opened_file)
        if result is None:
            ui.label('No result found for this file').classes('text-gray-500')
            return

        from services.extract_offer import validate_offer_json

        ui.label('Result').classes('text-lg font-bold')
        validation_warnings = validate_offer_json(result)
        if validation_warnings:
            with ui.column().classes('w-full gap-1 bg-yellow-50 border border-yellow-300 p-2 rounded'):
                ui.label('Checks').classes('font-medium text-yellow-900')
                for warning in validation_warnings:
                    ui.label(warning).classes('text-xs text-yellow-900')

        opened_file = self.state.opened_file
        render_editable_summary(
            result,
            on_update=lambda field, value: self.update_summary_value(opened_file, result, field, value),
            on_add=lambda field, value: self.add_summary_field(opened_file, result, field, value),
        )

        self.input_table(opened_file, result)

    def input_table(self, file: Path, result: dict) -> None:
        offer_table = OfferRowsTable(result)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten').classes('text-lg font-bold')
            ui.button(
                'Add row',
                icon='add',
                on_click=lambda: self.add_post_row(file, result, offer_table),
            ).props('dense no-caps size=sm')

        offer_tabulator = tabulator(offer_table.options(), row_key='id').classes('w-full')

        def update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            offer_table.update_cell(row.get('id'), column.get('field'), cell.get('value', ''))
            self.folder_handler.save_result(file, result)

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return

            row = cell.get('row', {})
            offer_table.delete_row(row.get('id'))
            self.folder_handler.save_result(file, result)
            offer_tabulator.set_data(offer_table.rows)

        offer_tabulator.on_event('cellEdited', update_cell)
        offer_tabulator.on_event('cellClick', delete_row)

    def show(self) -> None:
        with ui.splitter(value=50, limits=(25, 75)).classes('w-full h-screen max-h-screen').props(
            'before-class=overflow-hidden after-class=overflow-hidden'
        ) as splitter:
            with splitter.before:
                with ui.column().classes('w-full h-full'):
                    self.opened_file()

            with splitter.after:
                with ui.scroll_area().classes('w-full h-full p-4'):
                    self.opened_file_result()

    def update_summary_value(self, file: Path, result: dict, field: str, value: str) -> None:
        result[field] = value
        self.save_result(file, result)
        self.refresh()

    def add_summary_field(self, file: Path, result: dict, field: str | None, value: str | None) -> None:
        if not field:
            ui.notify('Enter a field name')
            return

        clean_field = field.strip()
        if not clean_field or clean_field == 'Posten':
            ui.notify('Invalid field name')
            return

        if clean_field in result:
            ui.notify(f'{clean_field} already exists')
            return

        result[clean_field] = value or ''
        self.save_result(file, result)
        self.refresh()

    def add_post_row(self, file: Path, result: dict, offer_table: OfferRowsTable | None = None) -> None:
        if offer_table is None:
            result.setdefault('Posten', [])
            result['Posten'].append({
                'Omschrijving': '',
                'Aantal': '',
                'Eenheid': '',
                'Eenheidsprijs': '',
                'Totaalbedrag': '',
            })
        else:
            offer_table.add_row()

        self.save_result(file, result)
        self.refresh()


class RightSide(SubPage):
    def __init__(self, *, state: MainPageState, folder_handler: FolderHandler, pdf_dir: Path) -> None:
        super().__init__(state, folder_handler, pdf_dir, None)

        self.comparison_page = ComparisonPage(
            state, folder_handler, pdf_dir, self.schedule_refresh_safe
        )
        self.offer_page = OfferPage(
            state, folder_handler, pdf_dir, self.schedule_refresh_safe
        )

    def render(self) -> None:
        self.container = ui.column().classes('w-full overflow-hidden')
        with self.container:
            self.show()

    def refresh(self) -> None:
        if self.container is None:
            return

        self.container.clear()
        with self.container:
            self.show()

    def schedule_refresh(self) -> None:
        ui.timer(0.05, self.refresh, once=True)

    def schedule_refresh_safe(self) -> None:
        try:
            self.schedule_refresh()
        except RuntimeError:
            pass

    def show(self) -> None:
        if self.state.current_view == 'comparison':
            self.comparison_page.render()
            return

        if self.state.current_view == "offer":
            self.offer_page.render()
            return
