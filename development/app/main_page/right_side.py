from pathlib import Path
from urllib.parse import quote
import json

from nicegui import run, ui

from main_page.editable_table_helper import render_editable_rows, render_editable_table
from main_page.folder_handler import FolderHandler
from main_page.page_state import MainPageState


class RightSide:
    def __init__(self, *, state: MainPageState, folder_handler: FolderHandler, pdf_dir: Path) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.pdf_dir = pdf_dir
        self.container = None

    def render(self) -> None:
        self.container = ui.column().classes('w-full h-full')
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
            with ui.column().classes('w-full h-full p-4'):
                self.comparison_page()
            return

        with ui.row().classes('w-full h-full flex-nowrap'):
            with ui.column().classes('w-1/2 h-full'):
                self.opened_file()
            with ui.column().classes('w-1/2 h-full p-4 overflow-auto'):
                self.opened_file_result()

    def opened_file(self) -> None:
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

    def opened_file_result(self) -> None:
        if self.state.opened_file is None:
            return

        result = self.folder_handler.load_result(self.state.opened_file)
        if result is None:
            ui.label('No result found for this file').classes('text-gray-500')
            return

        from main_page.extract_offer import validate_offer_json

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

    def save_result(self, file: Path, result: dict) -> None:
        self.folder_handler.save_result(file, result)

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

    def update_comparison_value(self, project: Path, comparison: dict, row_index: int, field: str, value: str) -> None:
        comparison['Posten'][row_index][field] = value
        comparison.pop('Matches', None)
        self.folder_handler.save_comparison(project, comparison)

    def add_comparison_row(self, project: Path, comparison: dict) -> None:
        comparison.setdefault('Posten', [])
        comparison['Posten'].append({
            'Omschrijving': '',
            'Aantal': '',
            'Eenheid': '',
        })
        comparison.pop('Matches', None)
        self.folder_handler.save_comparison(project, comparison)
        self.schedule_refresh()

    def delete_comparison_row(self, project: Path, comparison: dict, row_index: int) -> None:
        if 'Posten' not in comparison or row_index >= len(comparison['Posten']):
            ui.notify('Row no longer exists')
            self.refresh()
            return

        comparison['Posten'].pop(row_index)
        comparison.pop('Matches', None)
        self.folder_handler.save_comparison(project, comparison)
        self.schedule_refresh()

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
            Als er geen goede match is, vul dan "ONBEKEND" in voor de gematchte velden.
            Reageer ALLEEN met geldige JSON, zonder markdown, in exact dit formaat:
            {{
            "Matches": [
                {{
                "Vergelijking omschrijving": "...",
                "Vergelijking aantal": "...",
                "Vergelijking eenheid": "...",
                "Offerte": "...",
                "Gematchte omschrijving": "...",
                "Gematcht aantal": "...",
                "Gematchte eenheid": "...",
                "Totaalbedrag": "...",
                "Match toelichting": "..."
                }}
            ]
            }}
        """
        return parse_json_response(ask_llm(prompt))

    async def match_project_posts(self, project: Path, comparison: dict, button) -> None:
        if not comparison.get('Posten'):
            ui.notify('Add comparison rows before matching')
            return

        if not self.project_offer_results(project):
            ui.notify('No extracted offer results available for this project')
            return

        button.set_text('Matching')
        button.props('loading disable')
        button.update()

        try:
            match_result = await run.io_bound(self.match_comparison_posts, project, comparison)
        except Exception as error:
            self.notify_safe(f'Could not match posts: {error}')
            return

        comparison['Matches'] = match_result.get('Matches', [])
        self.folder_handler.save_comparison(project, comparison)
        self.notify_safe('Matched posts')
        self.schedule_refresh_safe()

    @staticmethod
    def notify_safe(message: str) -> None:
        try:
            ui.notify(message)
        except RuntimeError:
            print(message)

    def comparison_page(self) -> None:
        project = self.state.comparison_project
        if project is None:
            ui.label('No project selected').classes('text-gray-500')
            return

        ui.label(f'Comparison: {project.name}').classes('text-xl font-bold')

        result_files = [
            file
            for file in self.folder_handler.project_files(project)
            if self.folder_handler.result_path_for_file(file).exists()
        ]

        ui.label(f'{len(result_files)} extracted file(s) available for comparison').classes('text-sm text-gray-600')

        comparison = self.folder_handler.load_comparison(project)
        comparison_rows = [
            {'id': index, **row}
            for index, row in enumerate(comparison.get('Posten', []))
        ]

        ui.label('Comparison rows').classes('text-lg font-bold mt-4')
        render_editable_rows(
            comparison_rows,
            ['Omschrijving', 'Aantal', 'Eenheid'],
            on_update=lambda index, field, value: self.update_comparison_value(project, comparison, index, field, value),
            on_add=lambda: self.add_comparison_row(project, comparison),
            on_delete=lambda index: self.delete_comparison_row(project, comparison, index),
        )

        with ui.row().classes('items-center gap-2 mt-4'):
            match_button = ui.button('Match posten', icon='auto_fix_high').props('dense no-caps')

            async def request_match(_event, selected_project=project, data=comparison, button=match_button):
                await self.match_project_posts(selected_project, data, button)

            match_button.on('click', request_match)

        match_rows = [
            {'id': index, **row}
            for index, row in enumerate(comparison.get('Matches', []))
        ]
        if not match_rows:
            return

        ui.label('Matched posten').classes('text-lg font-bold mt-4')
        ui.table(
            columns=[
                {'name': 'Vergelijking omschrijving', 'label': 'Vergelijking', 'field': 'Vergelijking omschrijving'},
                {'name': 'Vergelijking aantal', 'label': 'Aantal', 'field': 'Vergelijking aantal'},
                {'name': 'Vergelijking eenheid', 'label': 'Eenheid', 'field': 'Vergelijking eenheid'},
                {'name': 'Offerte', 'label': 'Offerte', 'field': 'Offerte'},
                {'name': 'Gematchte omschrijving', 'label': 'Gematchte post', 'field': 'Gematchte omschrijving'},
                {'name': 'Gematcht aantal', 'label': 'Gem. aantal', 'field': 'Gematcht aantal'},
                {'name': 'Gematchte eenheid', 'label': 'Gem. eenheid', 'field': 'Gematchte eenheid'},
                {'name': 'Totaalbedrag', 'label': 'Totaal', 'field': 'Totaalbedrag'},
                {'name': 'Match toelichting', 'label': 'Toelichting', 'field': 'Match toelichting'},
            ],
            rows=match_rows,
            row_key='id',
        ).classes('w-full')
