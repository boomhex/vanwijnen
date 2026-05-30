from pathlib import Path
from .subpage import SubPage
from nicegui import ui
from interface.editable_table_helper import render_editable_summary
from nicegui_tabulator import tabulator
from .tabulator_table import TabulatorTable
from services.comparison_matcher import ComparisonMatcher
from services.offer import Offer, Posten

class OfferRowsTable(TabulatorTable):
    fields = ['Omschrijving', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag']

    def __init__(self, posten: list[Posten]) -> None:
        self.posten = posten
        super().__init__(
            rows=self.rows_from_posten(),
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

    def rows_from_posten(self) -> list[dict]:
        rows = []
        for index, row in enumerate(self.posten):
            formatted_row = {
                'id': index,
                'Omschrijving': row.omschrijving,
                'Aantal': row.aantal,
                'Eenheid': row.eenheid,
                'Eenheidsprijs': row.eenheidsprijs,
                'Totaalbedrag': row.totaalbedrag,
            }
            # Format monetary fields for display
            for field in ('Eenheidsprijs', 'Totaalbedrag'):
                if field in formatted_row and formatted_row[field]:
                    try:
                        decimal_val = ComparisonMatcher.parse_decimal(formatted_row[field])
                        if decimal_val is not None:
                            formatted_row[field] = self.format_money(decimal_val)
                    except (ValueError, AttributeError):
                        pass  # Keep original if formatting fails
            rows.append(formatted_row)
        return rows

    def add_row(self) -> dict:
        row = Posten()
        self.posten.append(row)
        display_row = {
            'id': len(self.posten) - 1,
            'Omschrijving': row.omschrijving,
            'Aantal': row.aantal,
            'Eenheid': row.eenheid,
            'Eenheidsprijs': row.eenheidsprijs,
            'Totaalbedrag': row.totaalbedrag,
        }
        self.rows.append(display_row)
        return display_row

    def update_cell(self, row_id: int | None, field: str | None, value: str) -> None:
        if field not in self.fields:
            return

        if row_id is None:
            return
        if row_id >= len(self.posten):
            return

        setattr(self.posten[row_id], self.field_to_attr(field), value)

    def delete_row(self, row_id: int | None) -> None:
        if row_id is None:
            return
        if row_id >= len(self.posten):
            return

        self.posten.pop(row_id)
        self.rows = self.rows_from_posten()

    @staticmethod
    def field_to_attr(field: str) -> str:
        return {
            'Omschrijving': 'omschrijving',
            'Aantal': 'aantal',
            'Eenheid': 'eenheid',
            'Eenheidsprijs': 'eenheidsprijs',
            'Totaalbedrag': 'totaalbedrag',
        }[field]


class OfferPage(SubPage):

    def render(self) -> None:
        self.show()

    def opened_file(self):
        if self.state.opened_offer is None:
            ui.label('No file selected')
            return

        pdf_url = self.state.opened_offer.pdf_url(self.projects_dir)

        ui.html(f'''
            <iframe
                src="{pdf_url}"
                style="width: 100%; height: 100%; border: none;"
            ></iframe>
        ''', sanitize=False).classes('w-full h-full overflow-hidden')

    def opened_file_result(self):
        if self.state.opened_offer is None:
            return

        result = self.state.opened_offer.load_data()
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

        self.input_table(opened_offer, result)

    def input_table(self, offer: 'Offer', result: dict) -> None:
        posten = offer.posten_list(result)
        offer_table = OfferRowsTable(posten)

        with ui.row().classes('items-center gap-2 mt-4'):
            ui.label('Posten').classes('text-lg font-bold')
            ui.button(
                'Add row',
                icon='add',
                on_click=lambda: self.add_post_row(offer, result, offer_table),
            ).props('dense no-caps size=sm')

        offer_tabulator = tabulator(offer_table.options(), row_key='id').classes('w-full')

        def update_cell(event) -> None:
            cell = event.args.get('cell', {})
            row = cell.get('row', {})
            column = cell.get('column', {})
            offer.update_post_row(result, row.get('id'), column.get('field'), cell.get('value', ''))
            offer_tabulator.set_data(offer_table.rows)

        def delete_row(event) -> None:
            cell = event.args.get('cell', {})
            column = cell.get('column', {})
            if column.get('field') != '__delete__':
                return

            row = cell.get('row', {})
            offer.delete_post_row(result, row.get('id'))
            offer_table.rows = offer_table.rows_from_posten()
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

    def update_summary_value(self, offer: 'Offer', result: dict, field: str, value: str) -> None:
        offer.update_summary_value(result, field, value)
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

        offer.add_summary_field(result, clean_field, value)
        self.refresh()

    def add_post_row(self, offer: 'Offer', result: dict, offer_table: OfferRowsTable | None = None) -> None:
        offer.add_post_row(result)
        if offer_table is not None:
            offer_table.rows = offer_table.rows_from_posten()
        self.refresh()

