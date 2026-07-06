from collections.abc import Callable

from nicegui import ui

from application.project_service import ProjectService
from domain.status import is_active_running, is_stale_running
from interface.page_state import MainPageState
from services.offer import Offer
from services.project import Project

from .actions import DrawerActions
from .utils import add_name_tooltip, compact_name


class DrawerTree:
    def __init__(
        self,
        *,
        state: MainPageState,
        actions: DrawerActions,
        project_service: ProjectService,
        open_offer: Callable[[Offer], None],
        select_project: Callable[[Project, bool], None],
    ) -> None:
        self.state = state
        self.actions = actions
        self.project_service = project_service
        self.open_offer = open_offer
        self.select_project = select_project

    def render(self, projects: list[Project]) -> None:
        ui.label('Projects').classes('text-lg font-bold mb-2')
        if not projects:
            ui.label('No projects or PDFs yet').classes('text-gray-500')
            return

        for project in projects:
            self.project_item(project)

    def project_item(self, project: Project) -> None:
        project_label = compact_name(project.name, max_length=24, mode='end')
        expansion = ui.expansion(
            project_label,
            icon='folder',
            value=project.name in self.state.expanded_project_names,
            on_value_change=lambda event, selected_project=project: self.select_project(
                selected_project,
                bool(event.value),
            ),
        ).classes(self.project_item_classes(project)).props('dense')
        if project_label != project.name:
            expansion.tooltip(project.name)

        with expansion:
            self.project_status(project)
            offers = project.offers()
            if not offers:
                ui.label('No offers in this project').classes('text-gray-500 pl-8')
                return

            for offer in offers:
                self.file_item(offer)

    def project_status(self, project: Project) -> None:
        comparison_status = self.project_service.load_status(project)
        comparison_running = is_active_running(comparison_status)
        comparison_stale = is_stale_running(comparison_status)
        if not comparison_running and not comparison_stale:
            return

        with ui.row().classes('items-center gap-1 pl-8 text-xs text-gray-700'):
            icon = ui.icon('hourglass_empty' if comparison_running else 'warning').classes(
                'text-orange-700 text-sm'
            )
            icon.tooltip(
                comparison_status.get('message')
                or comparison_status.get('step')
                or 'Comparison status is stale. You can retry matching.'
            )
            ui.label('Matching' if comparison_running else 'Comparison status is stale')

    def project_item_classes(self, project: Project) -> str:
        classes = 'w-full max-w-full rounded text-sm'
        if self.state.selected_project == project:
            classes += ' bg-red-50'
        return classes

    def file_item(self, offer: Offer) -> None:
        status = self.actions.offer_item_status(offer)
        selected = self.state.selected_offer == offer
        card_classes = 'w-[calc(100%-1.5rem)] ml-6 my-0.5 px-2 py-1 gap-0 cursor-pointer border rounded shadow-none'
        card_classes += ' bg-red-50 border-red-300' if selected else ' border-transparent'

        card = ui.card().classes(card_classes)
        card.on('click', lambda _event, selected_offer=offer: self.open_offer(selected_offer))
        with card:
            with ui.row().classes('items-center gap-2 w-full no-wrap min-h-0'):
                ui.icon('description').classes('text-gray-600 text-sm shrink-0')
                with ui.column().classes('gap-0 grow min-w-0 leading-tight'):
                    displayed_offer_name = compact_name(offer.name, max_length=34)
                    ui.label(displayed_offer_name).classes('font-medium text-xs leading-tight')
                    add_name_tooltip(displayed_offer_name, offer.name)
                    displayed_project_name = compact_name(offer.project_name, max_length=30)
                    ui.label(displayed_project_name).classes('text-[11px] leading-tight text-gray-500')
                    add_name_tooltip(displayed_project_name, offer.project_name)
                self.offer_status_icon(status)

    @staticmethod
    def offer_status_icon(status) -> None:
        if status.result_exists:
            ui.icon('check_circle').classes('text-green-700 text-sm shrink-0')
        elif status.extract_requested:
            status_icon = ui.icon('hourglass_empty').classes('text-orange-700 text-sm shrink-0')
            if status.status_step or status.status_message:
                status_icon.tooltip(status.status_message or status.status_step)
        elif status.extraction_failed:
            status_icon = ui.icon('warning').classes('text-red-700 text-sm shrink-0')
            status_icon.tooltip(status.status_error or status.status_message or 'Extraction failed')
        elif status.status_stale:
            status_icon = ui.icon('warning').classes('text-orange-700 text-sm shrink-0')
            status_icon.tooltip('Extraction status is stale. You can retry extraction.')
