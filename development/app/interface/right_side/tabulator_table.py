from decimal import Decimal

from domain.fields import OFFER_FIELDS
from domain.money import format_money as _format_money
from domain.offer import Posten, grouping_fields


class TabulatorTable:
    def __init__(
        self,
        *,
        rows: list[dict],
        columns: list[dict],
        layout: str = 'fitColumns',
        reactive: bool = True,
        height: str | None = None,
        group_by: list[str] | None = None,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.layout = layout
        self.reactive = reactive
        self.height = height
        self.group_by = group_by

    def options(self) -> dict:
        options = {
            'data': self.rows,
            'layout': self.layout,
            'reactiveData': self.reactive,
            'columns': self.columns,
        }
        if self.height is not None:
            options['height'] = self.height
        if self.group_by:
            options['groupBy'] = self.group_by

        return options

    @staticmethod
    def text_column(
        title: str,
        field: str,
        *,
        editable: bool = False,
        width: int | None = None,
        multiline: bool = False,
    ) -> dict:
        column = {
            'title': title,
            'field': field,
        }
        if editable:
            column['editor'] = 'textarea' if multiline else 'input'
        if width is not None:
            column['width'] = width
        if multiline:
            column[':formatter'] = """
                function(cell) {
                    const element = document.createElement('div');
                    element.textContent = cell.getValue() || '';
                    element.style.whiteSpace = 'normal';
                    element.style.textWrap = 'balance';
                    element.style.overflowWrap = 'break-word';
                    element.style.lineHeight = '1.25';
                    return element;
                }
            """
            column['variableHeight'] = True

        return column

    @staticmethod
    def format_money(value: Decimal | None) -> str:
        return _format_money(value)


class PostenTable(TabulatorTable):
    fields = OFFER_FIELDS

    def __init__(self, posten: list[Posten], pdf_url: str | None = None) -> None:
        self.posten = posten
        self.pdf_url = pdf_url
        super().__init__(
            rows=self.rows_from_posten(),
            columns=[
                self.text_column('Omschrijving', 'Omschrijving', editable=True, width=220, multiline=True),
                self.text_column('Beschrijving', 'Beschrijving', editable=True, width=280, multiline=True),
                self.text_column('Categorie', 'Categorie', editable=True, width=140),
                self.text_column('Aantal', 'Aantal', editable=True, width=90),
                self.text_column('Eenheid', 'Eenheid', editable=True, width=90),
                self.text_column('Eenheidsprijs', 'Eenheidsprijs', editable=True, width=110),
                self.text_column('Totaalbedrag', 'Totaalbedrag', editable=True, width=120),
                {
                    'title': '',
                    'field': '__open_pdf__',
                    'width': 56,
                    'headerSort': False,
                    'hozAlign': 'center',
                    'tooltip': 'Open bij deze pagina in de PDF',
                    ':formatter': """
                        function(cell) {
                            const value = cell.getRow().getData().Pagina;
                            if (!value || value === 'ONBEKEND') { return ''; }
                            return 'p.' + value;
                        }
                    """,
                },
                {
                    'title': '',
                    'field': '__delete__',
                    'width': 52,
                    'headerSort': False,
                    'hozAlign': 'center',
                    ':formatter': "function(){ return 'x'; }",
                },
            ],
            layout='fitDataStretch',
            reactive=False,
            height='60vh',
            group_by=grouping_fields(posten),
        )

    def rows_from_posten(self) -> list[dict]:
        return [
            {'id': index, **post.to_dict()}
            for index, post in enumerate(self.posten)
        ]
