from collections.abc import Callable, Sequence
from typing import Any

from nicegui import ui


SummaryUpdateHandler = Callable[[str, str], None]
SummaryAddHandler = Callable[[str | None, str | None], None]
RowUpdateHandler = Callable[[int, str, str], None]
RowDeleteHandler = Callable[[int], None]
RowAddHandler = Callable[[], None]


def render_editable_table(
    table_dict: dict[str, Any],
    *,
    row_collection_key: str,
    row_fields: Sequence[str],
    on_summary_update: SummaryUpdateHandler,
    on_summary_add: SummaryAddHandler | None = None,
    on_row_update: RowUpdateHandler,
    on_row_add: RowAddHandler | None = None,
    on_row_delete: RowDeleteHandler | None = None,
) -> None:
    render_editable_summary(
        table_dict,
        on_update=on_summary_update,
        on_add=on_summary_add,
        excluded_fields=(row_collection_key,),
    )

    ui.label(row_collection_key).classes('text-lg font-bold mt-4')
    rows = [
        {'id': index, **row}
        for index, row in enumerate(table_dict.get(row_collection_key, []))
    ]
    render_editable_rows(
        rows,
        row_fields,
        on_update=on_row_update,
        on_add=on_row_add,
        on_delete=on_row_delete,
    )


def render_editable_summary(
    table_dict: dict[str, Any],
    *,
    on_update: SummaryUpdateHandler,
    on_add: SummaryAddHandler | None = None,
    excluded_fields: Sequence[str] = ('Posten',),
) -> None:
    with ui.column().classes('w-full gap-1'):
        for field, value in table_dict.items():
            if field in excluded_fields:
                continue

            with ui.row().classes('items-center w-full gap-2 no-wrap'):
                ui.label(field).classes('w-40 text-xs font-medium')
                input_field = ui.input(value=str(value)).props('dense outlined').classes('grow')
                _commit_on_blur_and_enter(
                    input_field,
                    lambda key=field, field_input=input_field: on_update(key, field_input.value),
                )

        if on_add is not None:
            _render_add_summary_field(on_add)


def render_editable_rows(
    rows: list[dict[str, Any]],
    fields: Sequence[str],
    *,
    on_update: RowUpdateHandler,
    on_add: RowAddHandler | None = None,
    on_delete: RowDeleteHandler | None = None,
) -> None:
    with ui.column().classes('w-full gap-2'):
        if on_add is not None:
            ui.button(
                'Add row',
                icon='add',
                on_click=on_add,
            ).props('dense no-caps size=sm').classes('self-start')

        with ui.row().classes('w-full gap-1 no-wrap text-xs font-medium text-gray-600'):
            for field in fields:
                ui.label(field).classes('grow basis-0')
            if on_delete is not None:
                ui.label('').classes('w-8')

        for index, row in enumerate(rows):
            row_index = row.get('id', index)
            with ui.row().classes('w-full gap-1 no-wrap'):
                for field in fields:
                    input_field = ui.input(value=str(row.get(field, ''))).props('dense outlined').classes('grow basis-0')
                    _commit_on_blur_and_enter(
                        input_field,
                        lambda index=row_index, key=field, field_input=input_field: on_update(
                            index,
                            key,
                            field_input.value,
                        ),
                    )

                if on_delete is not None:
                    ui.button(
                        icon='close',
                        on_click=lambda index=row_index: on_delete(index),
                    ).props('flat dense round color=negative size=sm').classes('w-8')


def _render_add_summary_field(on_add: SummaryAddHandler) -> None:
    with ui.row().classes('items-end w-full gap-2 no-wrap mt-2'):
        field_input = ui.input('New field').props('dense outlined').classes('w-40')
        value_input = ui.input('Value').props('dense outlined').classes('grow')
        ui.button(
            'Add field',
            icon='add',
            on_click=lambda: on_add(field_input.value, value_input.value),
        ).props('dense no-caps size=sm')


def _commit_on_blur_and_enter(input_field, on_commit: Callable[[], None]) -> None:
    input_field.on('blur', lambda _event: on_commit())
    input_field.on('keydown.enter', lambda _event: on_commit())
