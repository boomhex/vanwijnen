from pathlib import Path
from urllib.parse import quote
from collections.abc import Callable

from abc import ABC, abstractmethod

from nicegui_tabulator import tabulator
from nicegui import run, ui

from services.comparison_matcher import ComparisonMatcher
from ui.editable_table_helper import render_editable_table
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
            self.text_column('Aantal', 'Aantal', width=100),
            self.text_column('Eenheid', 'Eenheid', width=100),
        ]

        for offer_name in self.offer_names:
            field_prefix = self.offer_field_prefix(offer_name)
            columns.extend([
                self.text_column(f'{offer_name} post', f'{field_prefix}_omschrijving', width=260),
                self.text_column(f'{offer_name} prijs', f'{field_prefix}_prijs', width=120),
                self.text_column(f'{offer_name} totaal', f'{field_prefix}_totaal', width=130),
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
                row[f'{field_prefix}_omschrijving'] = offer.get('Gematchte omschrijving', 'ONBEKEND')
                row[f'{field_prefix}_prijs'] = offer.get('Eenheidsprijs', 'ONBEKEND')
                row[f'{field_prefix}_totaal'] = offer.get('Totaalbedrag', 'ONBEKEND')
            rows.append(row)

        return rows


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
        self.render_side_by_side_match_table(project, match_rows)

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

    def render_side_by_side_match_table(self, project: Path, match_rows: list[dict]) -> None:
        offer_names = [offer['Bestand'] for offer in self.matcher.project_offer_results(project)]
        matched_table = MatchedPostenTable(offer_names=offer_names, match_rows=match_rows)
        tabulator(matched_table.options(), row_key='id').classes('w-full')

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
        with ui.row().classes('w-full h-screen max-h-screen flex-nowrap'):
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
                style="width: 100%; height: 100vh; border: none;"
            ></iframe>
        ''', sanitize=False).classes('w-full h-full')

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
        render_editable_table(
            result,
            row_collection_key='Posten',
            row_fields=['Omschrijving', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag'],
            on_summary_update=lambda field, value: self.update_summary_value(opened_file, result, field, value),
            on_summary_add=lambda field, value: self.add_summary_field(opened_file, result, field, value),
            on_row_update=lambda index, field, value: self.update_post_value(opened_file, result, index, field, value),
            on_row_add=lambda: self.add_post_row(opened_file, result),
            on_row_delete=lambda index: self.delete_post_row(opened_file, result, index),
        )

    def show(self) -> None:
        with ui.column().classes('w-1/2 h-full'):
            self.opened_file()

        with ui.column().classes('w-1/2 h-full'):
            with ui.scroll_area().classes('w-full h-full p-4'):
                    self.opened_file_result()

    def update_summary_value(self, file: Path, result: dict, field: str, value: str) -> None:
        result[field] = value
        self.save_result(file, result)
        self.refresh()

    def update_post_value(self, file: Path, result: dict, post_index: int, field: str, value: str) -> None:
        result['Posten'][post_index][field] = value
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

    def add_post_row(self, file: Path, result: dict) -> None:
        result.setdefault('Posten', [])
        result['Posten'].append({
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
            'Eenheidsprijs': '',
            'Totaalbedrag': '',
        })
        self.save_result(file, result)
        self.refresh()

    def delete_post_row(self, file: Path, result: dict, post_index: int) -> None:
        if 'Posten' not in result or post_index >= len(result['Posten']):
            ui.notify('Row no longer exists')
            self.refresh()
            return

        result['Posten'].pop(post_index)
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
