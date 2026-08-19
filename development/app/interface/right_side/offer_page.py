import json

from .subpage import SubPage
from nicegui import ui
from nicegui_tabulator import tabulator
from domain.status import extraction_progress_fraction, is_active_running
from interface.editable_table_helper import render_editable_summary
from interface.left_drawer.utils import format_elapsed_seconds
from interface.page_state import PendingUndo
from application.offer_service import OfferService
from domain.offer import Offer
from .tabulator_table import PostenTable


class OfferPage(SubPage):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.offer_service = OfferService(self.folder_handler)

    def render(self) -> None:
        self.show()

    def opened_file(self):
        if self.state.opened_offer is None:
            ui.label('Geen bestand geselecteerd')
            return

        pdf_url = self.state.opened_offer.pdf_url(self.projects_dir)
        ui.button(
            'PDF openen',
            icon='picture_as_pdf',
            on_click=lambda url=pdf_url: ui.run_javascript(
                f'window.open({json.dumps(url)}, "_blank", "noopener,noreferrer")'
            ),
        ).props('no-caps color=primary')

    def opened_file_result(self):
        if self.state.opened_offer is None:
            return

        offer = self.state.opened_offer
        extraction_status = self.folder_handler.load_extraction_status(offer.document)
        if is_active_running(extraction_status):
            self.render_extraction_progress(extraction_status)

        result = self.offer_service.load_data(offer)
        if result is None:
            ui.label('Geen resultaat gevonden voor dit bestand').classes('text-gray-500')
            return

        from services.extract_offer import validate_offer_json

        ui.label('Resultaat').classes('text-lg font-bold')
        validation_warnings = validate_offer_json(result)
        if validation_warnings:
            with ui.column().classes('w-full gap-1 bg-yellow-50 border border-yellow-300 p-2 rounded'):
                ui.label('Controles').classes('font-medium text-yellow-900')
                for warning in validation_warnings:
                    ui.label(warning).classes('text-xs text-yellow-900')

        opened_offer = self.state.opened_offer
        render_editable_summary(
            result,
            on_update=lambda field, value: self.update_summary_value(opened_offer, result, field, value),
            on_add=lambda field, value: self.add_summary_field(opened_offer, result, field, value),
        )

        self.render_post_table(opened_offer, result)

    def render_extraction_progress(self, status: dict) -> None:
        step = status.get('step')
        fraction = extraction_progress_fraction(step)
        with ui.column().classes('w-full gap-1 bg-blue-50 border border-blue-200 p-2 rounded'):
            with ui.row().classes('items-center gap-2 w-full no-wrap'):
                ui.label(status.get('message') or step or 'Bezig met extraheren…').classes('text-sm text-blue-900 grow')
                elapsed = format_elapsed_seconds(status.get('started_at'))
                if elapsed:
                    ui.label(elapsed).classes('text-xs text-blue-700 shrink-0')
            progress = ui.linear_progress(value=fraction or 0.0, show_value=False, size='6px').classes('w-full')
            if fraction is None:
                progress.props('indeterminate')

    def render_post_table(self, offer: 'Offer', result: dict) -> None:
        posten = self.offer_service.posten_list(result)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten').classes('text-lg font-bold')
            ui.button(
                'Post toevoegen',
                icon='add',
                on_click=lambda: self.add_post_row(offer, result),
            ).props('dense no-caps size=sm')

        if not posten:
            ui.label('Geen posten gevonden').classes('text-gray-500')
            return

        pdf_url = offer.pdf_url(self.projects_dir)
        posten_table = PostenTable(posten, pdf_url)
        with ui.element('div').classes('w-full overflow-x-auto'):
            posten_tab = tabulator(posten_table.options(), row_key='id').classes('w-full')

        def update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            self.commit_post_field(offer, result, row.get('id'), column.get('field'), cell.get('value', ''))

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return
            row = cell.get('row', {})
            self.delete_post_row(offer, result, row.get('id'))

        def open_pdf_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__open_pdf__':
                return
            row = cell.get('row', {})
            pagina = str(row.get('Pagina') or '').strip()
            if not pagina or pagina.upper() == 'ONBEKEND':
                return
            ui.run_javascript(
                f'window.open({json.dumps(f"{pdf_url}#page={pagina}")}, "_blank", "noopener,noreferrer")'
            )

        posten_tab.on_event('cellEdited', update_cell)
        posten_tab.on_event('cellClick', delete_row)
        posten_tab.on_event('cellClick', open_pdf_row)

    def commit_post_field(self, offer: 'Offer', result: dict, row_id: int | None, field: str | None, value: str) -> None:
        if row_id is None or field is None:
            return

        try:
            self.offer_service.update_post_row(offer, result, row_id, field, value)
        except Exception as error:  # noqa: BLE001 - surface any save failure to the user
            ui.notify(f'Kon {field} niet opslaan: {error}', type='negative')
            return

        ui.notify('Opgeslagen', type='positive')

    def show(self) -> None:
        with ui.scroll_area().classes('w-full h-screen p-4'):
            with ui.column().classes('w-full gap-4'):
                self.opened_file()
                self.opened_file_result()

    def update_summary_value(self, offer: 'Offer', result: dict, field: str, value: str) -> None:
        try:
            self.offer_service.update_summary_value(offer, result, field, value)
        except Exception as error:  # noqa: BLE001 - surface any save failure to the user
            ui.notify(f'Kon {field} niet opslaan: {error}', type='negative')
            return

        ui.notify('Opgeslagen', type='positive')
        self.refresh()

    def add_summary_field(self, offer: 'Offer', result: dict, field: str | None, value: str | None) -> None:
        if not field:
            ui.notify('Voer een veldnaam in', type='negative')
            return

        clean_field = field.strip()
        if not clean_field or clean_field == 'Posten':
            ui.notify('Ongeldige veldnaam', type='negative')
            return

        if clean_field in result:
            ui.notify(f'{clean_field} bestaat al', type='negative')
            return

        self.offer_service.add_summary_field(offer, result, clean_field, value)
        ui.notify(f'{clean_field} toegevoegd', type='positive')
        self.refresh()

    def add_post_row(self, offer: 'Offer', result: dict) -> None:
        self.offer_service.add_post_row(offer, result)
        self.refresh()

    def delete_post_row(self, offer: 'Offer', result: dict, row_id: int | None) -> None:
        if row_id is None:
            return

        posten = self.offer_service.posten_list(result)
        if row_id < 0 or row_id >= len(posten):
            return

        removed = posten[row_id]
        self.offer_service.delete_post_row(offer, result, row_id)

        label = removed.omschrijving.strip() or f'post {row_id + 1}'

        def restore(offer=offer, row_id=row_id, removed=removed) -> None:
            fresh_result = self.offer_service.load_data(offer)
            if fresh_result is None:
                return
            fresh_posten = self.offer_service.posten_list(fresh_result)
            fresh_posten.insert(min(row_id, len(fresh_posten)), removed)
            self.offer_service.save_posten_list(offer, fresh_result, fresh_posten)

        self.state.pending_undo = PendingUndo(label=f'"{label}" verwijderd', restore=restore)
        self.refresh()
