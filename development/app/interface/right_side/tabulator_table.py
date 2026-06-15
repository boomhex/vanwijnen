from decimal import Decimal

from domain.money import format_money as _format_money


class TabulatorTable:
    def __init__(
        self,
        *,
        rows: list[dict],
        columns: list[dict],
        layout: str = 'fitColumns',
        reactive: bool = True,
        height: str | None = None,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.layout = layout
        self.reactive = reactive
        self.height = height

    def options(self) -> dict:
        options = {
            'data': self.rows,
            'layout': self.layout,
            'reactiveData': self.reactive,
            'columns': self.columns,
        }
        if self.height is not None:
            options['height'] = self.height

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
