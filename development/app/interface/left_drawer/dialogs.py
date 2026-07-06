from collections.abc import Callable

from nicegui import ui

from services.offer import Offer
from services.project import Project


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

