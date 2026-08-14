from collections.abc import Callable

from nicegui import ui

from services.offer import Offer
from services.project import Project


STATUS_FIELD_LABELS = [
    ('Status', 'status'),
    ('Step', 'step'),
    ('Message', 'message'),
    ('Error', 'error'),
    ('Started at', 'started_at'),
    ('Updated at', 'updated_at'),
]


def offer_status_dialog(offer: Offer, status: dict | None) -> None:
    with ui.dialog() as dialog, ui.card().classes('gap-2 min-w-[20rem]'):
        ui.label(f'Extraction status for {offer.name}').classes('font-medium')
        render_status_fields(status)

        with ui.row().classes('justify-end w-full'):
            ui.button('Close', on_click=dialog.close).props('flat dense no-caps size=sm')

    dialog.open()


def render_status_fields(status: dict | None) -> None:
    if not status:
        ui.label('No status recorded yet.').classes('text-sm text-gray-600')
        return

    for label, key in STATUS_FIELD_LABELS:
        value = status.get(key)
        if not value:
            continue
        with ui.row().classes('items-start gap-2 w-full no-wrap'):
            ui.label(label).classes('w-24 text-xs font-medium text-gray-500 shrink-0')
            ui.label(str(value)).classes('text-xs break-words')


def reset_extraction_button(offer: Offer, on_confirm: Callable[[], None]) -> None:
    with ui.dialog() as dialog, ui.card().classes('gap-3'):
        ui.label(f'Reset extraction for {offer.name}?').classes('font-medium')
        ui.label(
            'This removes extract.json, status.json, and saved LLM responses. The PDF and raw.txt stay in place.'
        ).classes('text-sm text-gray-600')

        def confirm() -> None:
            dialog.close()
            on_confirm()

        with ui.row().classes('justify-end gap-2 w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Reset', icon='restart_alt', on_click=confirm).props('dense no-caps size=sm color=warning')

    dialog.open()


def rename_offer_dialog(offer: Offer, on_save: Callable[[str | None], bool]) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Rename {offer.name}').classes('font-medium')
        name_input = ui.input('Filename', value=offer.name).classes('w-80')

        def save() -> None:
            if on_save(name_input.value):
                dialog.close()

        name_input.on('keydown.enter', save)

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Save', on_click=save).props('dense no-caps size=sm')

    dialog.open()


def rename_project_dialog(project: Project, on_save: Callable[[str | None], bool]) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Rename {project.name}').classes('font-medium')
        name_input = ui.input('Project folder', value=project.name).classes('w-80')

        def save() -> None:
            if on_save(name_input.value):
                dialog.close()

        name_input.on('keydown.enter', save)

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Save', on_click=save).props('dense no-caps size=sm')

    dialog.open()


def delete_project_dialog(project: Project, on_confirm: Callable[[], bool]) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Delete {project.name}?').classes('font-medium')
        ui.label('This removes the project folder, all offers, and the comparison.').classes('text-sm text-gray-600')

        def confirm() -> None:
            if on_confirm():
                dialog.close()

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Delete', icon='delete', on_click=confirm).props('dense no-caps size=sm color=negative')

    dialog.open()


def bulk_delete_dialog(offers: list[Offer], on_confirm: Callable[[], None]) -> None:
    with ui.dialog() as dialog, ui.card().classes('gap-3'):
        ui.label(f'Delete {len(offers)} offer(s)?').classes('font-medium')
        preview_names = ', '.join(offer.name for offer in offers[:5])
        if len(offers) > 5:
            preview_names += f', +{len(offers) - 5} more'
        ui.label(preview_names).classes('text-sm text-gray-600')
        ui.label('This permanently removes each offer folder, including the PDF.').classes('text-xs text-gray-500')

        def confirm() -> None:
            dialog.close()
            on_confirm()

        with ui.row().classes('justify-end gap-2 w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Delete', icon='delete', on_click=confirm).props('dense no-caps size=sm color=negative')

    dialog.open()


def bulk_move_dialog(
    offers: list[Offer],
    *,
    project_options: list[str],
    on_save: Callable[[str | None], bool],
) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Move {len(offers)} offer(s)').classes('font-medium')
        project_select = ui.select(project_options, label='Project').classes('w-80')

        def save() -> None:
            if on_save(project_select.value):
                dialog.close()

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Move', on_click=save).props('dense no-caps size=sm')

    dialog.open()


def move_offer_dialog(
    offer: Offer,
    *,
    project_options: list[str],
    current_project: str,
    on_save: Callable[[str | None], bool],
) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Move {offer.name}').classes('font-medium')
        project_select = ui.select(
            project_options,
            label='Project',
            value=current_project,
        ).classes('w-80')

        def save() -> None:
            if on_save(project_select.value):
                dialog.close()

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Move', on_click=save).props('dense no-caps size=sm')

    dialog.open()

