from decimal import Decimal

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

    @staticmethod
    def format_money(value: Decimal | None) -> str:
        """Format decimal value as European format (1.234,56 without euro sign)."""
        if value is None:
            return 'ONBEKEND'

        rounded = value.quantize(Decimal('0.01'))
        # Format with thousands separator (US style: 1,234.56)
        us_format = f'{rounded:,.2f}'
        # Convert to European style (1.234,56)
        european_format = '€' + us_format.replace(',', '\x00').replace('.', ',').replace('\x00', '.')
        return european_format
