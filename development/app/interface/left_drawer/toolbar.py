from collections.abc import Callable

from nicegui import ui

from interface.page_state import MainPageState
from services.offer import Offer
from services.project import Project

from . import dialogs
from .actions import DrawerActions


class DrawerToolbar:
    def __init__(
        self,
        *,
        state: MainPageState,
        actions: DrawerActions,
        open_offer: Callable[[Offer], None],
        open_project: Callable[[Project], None],
        extract_offer: Callable[[Offer], None],
        reset_offer: Callable[[Offer], None],
        rename_offer: Callable[[Offer, str | None], bool],
        rename_project: Callable[[Project, str | None], bool],
        move_offer: Callable[[Offer], None],
        delete_offer: Callable[[Offer], None],
        delete_project: Callable[[Project], bool],
        toggle_selection_mode: Callable[[], None],
        bulk_extract: Callable[[], None],
        bulk_move: Callable[[], None],
        bulk_delete: Callable[[], None],
    ) -> None:
        self.state = state
        self.actions = actions
        self.open_offer = open_offer
        self.open_project = open_project
        self.extract_offer = extract_offer
        self.reset_offer = reset_offer
        self.rename_offer = rename_offer
        self.rename_project = rename_project
        self.move_offer = move_offer
        self.delete_offer = delete_offer
        self.delete_project = delete_project
        self.toggle_selection_mode = toggle_selection_mode
        self.bulk_extract = bulk_extract
        self.bulk_move = bulk_move
        self.bulk_delete = bulk_delete

    def render(self) -> None:
        selected_offer = self.selected_offer()
        selected_project = self.selected_project()
        selected_label = self.selected_label(selected_offer, selected_project)

        with ui.column().classes('w-full gap-1 mt-1 mb-2'):
            with ui.row().classes('items-center justify-between gap-2 w-full no-wrap'):
                selected_text = ui.label(selected_label).classes(
                    'text-xs text-gray-700 min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap'
                )
                if selected_label != 'No selection':
                    selected_text.tooltip(selected_label)
                with ui.row().classes('items-center gap-1 no-wrap shrink-0'):
                    self.toolbar_button(
                        icon='close' if self.state.selection_mode else 'checklist',
                        tooltip='Exit selection mode' if self.state.selection_mode else 'Select multiple offers',
                        on_click=self.toggle_selection_mode,
                        enabled=True,
                    )
                    self.toolbar_button(
                        icon='visibility' if selected_offer is not None else 'compare_arrows',
                        tooltip='Open offer' if selected_offer is not None else 'Compare project',
                        on_click=self.open_selected,
                        enabled=selected_offer is not None or selected_project is not None,
                    )
                    self.toolbar_button(
                        icon='text_snippet',
                        tooltip='Extract offer',
                        on_click=self.extract_selected,
                        enabled=self.can_extract_selected(selected_offer),
                    )
                    self.toolbar_button(
                        icon='restart_alt',
                        tooltip='Reset extraction',
                        on_click=self.reset_selected,
                        enabled=self.can_reset_selected(selected_offer),
                        color='warning',
                    )
                    self.toolbar_button(
                        icon='edit',
                        tooltip='Rename',
                        on_click=self.rename_selected,
                        enabled=selected_offer is not None or selected_project is not None,
                    )
                    self.toolbar_button(
                        icon='drive_file_move',
                        tooltip='Move offer',
                        on_click=self.move_selected,
                        enabled=selected_offer is not None,
                    )
                    self.toolbar_button(
                        icon='delete',
                        tooltip='Delete',
                        on_click=self.delete_selected,
                        enabled=selected_offer is not None or selected_project is not None,
                        color='negative',
                    )

            if self.state.selection_mode:
                self.render_bulk_row()

    def render_bulk_row(self) -> None:
        count = len(self.state.selected_offers)
        with ui.row().classes('items-center gap-2 w-full no-wrap text-xs text-gray-700'):
            ui.label(f'{count} selected')
            self.toolbar_button(
                icon='text_snippet',
                tooltip='Extract selected offers',
                on_click=self.bulk_extract,
                enabled=count > 0,
            )
            self.toolbar_button(
                icon='drive_file_move',
                tooltip='Move selected offers',
                on_click=self.bulk_move,
                enabled=count > 0,
            )
            self.toolbar_button(
                icon='delete',
                tooltip='Delete selected offers',
                on_click=self.bulk_delete,
                enabled=count > 0,
                color='negative',
            )

    @staticmethod
    def toolbar_button(
        *,
        icon: str,
        tooltip: str,
        on_click,
        enabled: bool,
        color: str | None = None,
    ) -> None:
        button = ui.button(icon=icon, on_click=on_click).props('flat dense round size=sm')
        if color:
            button.props(f'color={color}')
        if not enabled:
            button.props('disable')
        button.tooltip(tooltip)

    @staticmethod
    def selected_label(offer: Offer | None, project: Project | None) -> str:
        if offer is not None:
            return offer.name
        if project is not None:
            return project.name
        return 'No selection'

    def selected_offer(self) -> Offer | None:
        offer = self.state.selected_offer
        if offer is None or not offer.document.exists():
            return None
        return offer

    def selected_project(self) -> Project | None:
        if self.selected_offer() is not None:
            return None

        project = self.state.selected_project
        if project is None or not project.path.exists():
            return None
        return project

    def can_extract_selected(self, offer: Offer | None) -> bool:
        if offer is None:
            return False
        status = self.actions.offer_item_status(offer)
        return not status.result_exists and not status.extract_requested

    def can_reset_selected(self, offer: Offer | None) -> bool:
        if offer is None:
            return False
        status = self.actions.offer_item_status(offer)
        return bool(status.result_exists or status.extraction_status) and not status.extract_requested

    def open_selected(self) -> None:
        selected_offer = self.selected_offer()
        if selected_offer is not None:
            self.open_offer(selected_offer)
            return

        selected_project = self.selected_project()
        if selected_project is not None:
            self.open_project(selected_project)

    def extract_selected(self) -> None:
        selected_offer = self.selected_offer()
        if selected_offer is not None:
            self.extract_offer(selected_offer)

    def reset_selected(self) -> None:
        selected_offer = self.selected_offer()
        if selected_offer is not None:
            self.reset_offer(selected_offer)

    def rename_selected(self) -> None:
        selected_offer = self.selected_offer()
        if selected_offer is not None:
            dialogs.rename_offer_dialog(
                selected_offer,
                on_save=lambda new_name, offer=selected_offer: self.rename_offer(offer, new_name),
            )
            return

        selected_project = self.selected_project()
        if selected_project is not None:
            dialogs.rename_project_dialog(
                selected_project,
                on_save=lambda new_name, project=selected_project: self.rename_project(project, new_name),
            )

    def move_selected(self) -> None:
        selected_offer = self.selected_offer()
        if selected_offer is not None:
            self.move_offer(selected_offer)

    def delete_selected(self) -> None:
        selected_offer = self.selected_offer()
        if selected_offer is not None:
            self.delete_offer(selected_offer)
            return

        selected_project = self.selected_project()
        if selected_project is not None:
            dialogs.delete_project_dialog(
                selected_project,
                on_confirm=lambda project=selected_project: self.delete_project(project),
            )
