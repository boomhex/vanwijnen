import json

from .subpage import SubPage
from nicegui import ui
from interface.editable_table_helper import render_editable_summary
from application.offer_service import OfferService
from domain.offer import Offer, Posten


class OfferPage(SubPage):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.offer_service = OfferService(self.folder_handler)

    def render(self) -> None:
        self.show()

    def opened_file(self):
        if self.state.opened_offer is None:
            ui.label('No file selected')
            return

        pdf_url = self.state.opened_offer.pdf_url(self.projects_dir)
        ui.button(
            'Open PDF',
            icon='picture_as_pdf',
            on_click=lambda url=pdf_url: ui.run_javascript(
                f'window.open({json.dumps(url)}, "_blank", "noopener,noreferrer")'
            ),
        ).props('no-caps color=primary')

    def opened_file_result(self):
        if self.state.opened_offer is None:
            return

        result = self.offer_service.load_data(self.state.opened_offer)
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

        opened_offer = self.state.opened_offer
        render_editable_summary(
            result,
            on_update=lambda field, value: self.update_summary_value(opened_offer, result, field, value),
            on_add=lambda field, value: self.add_summary_field(opened_offer, result, field, value),
        )

        self.render_post_cards(opened_offer, result)

    def render_post_cards(self, offer: 'Offer', result: dict) -> None:
        posten = self.offer_service.posten_list(result)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten').classes('text-lg font-bold')
            ui.button(
                'Add post',
                icon='add',
                on_click=lambda: self.add_post_row(offer, result),
            ).props('dense no-caps size=sm')

        if not posten:
            ui.label('No posts found').classes('text-gray-500')
            return

        with ui.column().classes('w-full gap-3'):
            for index, post in enumerate(posten):
                self.render_post_card(offer, result, index, post)

    def render_post_card(self, offer: 'Offer', result: dict, index: int, post: Posten) -> None:
        with ui.card().classes('w-full p-3 gap-3 rounded-lg border border-gray-200 shadow-none'):
            with ui.row().classes('w-full items-start justify-between gap-3 no-wrap'):
                ui.label(f'Post {index + 1}').classes('text-sm font-semibold text-gray-600')
                ui.button(
                    icon='delete',
                    on_click=lambda row_id=index: self.delete_post_row(offer, result, row_id),
                ).props('flat dense round color=negative size=sm')

            self.post_textarea(offer, result, index, 'Omschrijving', post.omschrijving, min_rows=1)
            self.post_textarea(offer, result, index, 'Beschrijving', post.beschrijving, min_rows=2)

            with ui.element('div').classes('grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-2 w-full'):
                self.post_input(offer, result, index, 'Categorie', post.categorie)
                self.post_input(offer, result, index, 'Aantal', post.aantal)
                self.post_input(offer, result, index, 'Eenheid', post.eenheid)
                self.post_input(offer, result, index, 'Eenheidsprijs', post.eenheidsprijs)
                self.post_input(offer, result, index, 'Totaalbedrag', post.totaalbedrag)

    def post_input(self, offer: 'Offer', result: dict, index: int, field: str, value: str) -> None:
        input_field = ui.input(field, value=str(value)).props('dense outlined').classes('w-full')
        self.commit_on_blur_and_enter(
            input_field,
            lambda field_input=input_field, row_id=index, key=field: self.offer_service.update_post_row(
                offer,
                result,
                row_id,
                key,
                field_input.value,
            ),
        )

    def post_textarea(
        self,
        offer: 'Offer',
        result: dict,
        index: int,
        field: str,
        value: str,
        *,
        min_rows: int,
    ) -> None:
        input_field = ui.textarea(field, value=str(value)).props(f'outlined autogrow rows={min_rows}').classes('w-full')
        self.commit_on_blur_and_enter(
            input_field,
            lambda field_input=input_field, row_id=index, key=field: self.offer_service.update_post_row(
                offer,
                result,
                row_id,
                key,
                field_input.value,
            ),
        )

    @staticmethod
    def commit_on_blur_and_enter(input_field, on_commit) -> None:
        input_field.on('blur', lambda _event: on_commit())
        input_field.on('keydown.enter', lambda _event: on_commit())

    def show(self) -> None:
        with ui.scroll_area().classes('w-full h-screen p-4'):
            with ui.column().classes('w-full gap-4'):
                self.opened_file()
                self.opened_file_result()

    def update_summary_value(self, offer: 'Offer', result: dict, field: str, value: str) -> None:
        self.offer_service.update_summary_value(offer, result, field, value)
        self.refresh()

    def add_summary_field(self, offer: 'Offer', result: dict, field: str | None, value: str | None) -> None:
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

        self.offer_service.add_summary_field(offer, result, clean_field, value)
        self.refresh()

    def add_post_row(self, offer: 'Offer', result: dict) -> None:
        self.offer_service.add_post_row(offer, result)
        self.refresh()

    def delete_post_row(self, offer: 'Offer', result: dict, row_id: int) -> None:
        self.offer_service.delete_post_row(offer, result, row_id)
        self.refresh()
