from collections.abc import Callable

from nicegui import ui

from application.project_service import ProjectService
from domain.status import extraction_progress_fraction, is_active_running, is_stale_running
from interface.page_state import MainPageState
from services.offer import Offer
from services.project import Project

from . import dialogs
from .actions import DrawerActions
from .utils import add_name_tooltip, compact_name, format_elapsed_seconds


SEARCH_INPUT_HTML_ID = 'drawer-tree-search'


class DrawerTree:
    def __init__(
        self,
        *,
        state: MainPageState,
        actions: DrawerActions,
        project_service: ProjectService,
        open_offer: Callable[[Offer], None],
        select_project: Callable[[Project, bool], None],
        search_changed: Callable[[str], None],
        toggle_offer_selected: Callable[[Offer, bool], None],
        cancel_extraction: Callable[[Offer], None],
    ) -> None:
        self.state = state
        self.actions = actions
        self.project_service = project_service
        self.open_offer = open_offer
        self.select_project = select_project
        self.search_changed = search_changed
        self.toggle_offer_selected = toggle_offer_selected
        self.cancel_extraction = cancel_extraction

    def render(self, projects: list[Project]) -> None:
        ui.label('Projecten').classes('text-lg font-bold mb-2')
        self.search_box()

        if not projects:
            ui.label('Nog geen projecten of PDF\'s').classes('text-gray-500')
            return

        query = self.state.tree_search.strip().lower()
        visible_projects = [project for project in projects if self.project_matches(project, query)] if query else projects
        if query and not visible_projects:
            ui.label('Geen resultaten gevonden').classes('text-gray-500')
            return

        for project in visible_projects:
            self.project_item(project, query)

    def search_box(self) -> None:
        ui.input(
            placeholder='Zoek projecten of offertes…',
            value=self.state.tree_search,
            on_change=lambda event: self.search_changed(event.value or ''),
        ).props(
            f'dense outlined clearable debounce=400 prepend-icon=search id={SEARCH_INPUT_HTML_ID}'
        ).classes('w-full mb-2')

    @staticmethod
    def project_matches(project: Project, query: str) -> bool:
        if query in project.name.lower():
            return True
        return any(query in offer.name.lower() for offer in project.offers())

    def project_item(self, project: Project, query: str = '') -> None:
        project_label = compact_name(project.name, max_length=24, mode='end')
        force_expanded = bool(query)
        expansion = ui.expansion(
            project_label,
            icon='folder',
            value=force_expanded or project.name in self.state.expanded_project_names,
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
            if query and query not in project.name.lower():
                offers = [offer for offer in offers if query in offer.name.lower()]
            if not offers:
                ui.label('Geen offertes in dit project').classes('text-gray-500 pl-8')
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
                or 'Vergelijkingsstatus is verouderd. U kunt het matchen opnieuw proberen.'
            )
            ui.label('Bezig met matchen' if comparison_running else 'Vergelijkingsstatus is verouderd')

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
                if self.state.selection_mode:
                    checkbox = ui.checkbox(
                        value=offer in self.state.selected_offers,
                        on_change=lambda event, offer=offer: self.toggle_offer_selected(offer, bool(event.value)),
                    ).props('dense').classes('shrink-0')
                    checkbox.on('click.stop', lambda: None)
                ui.icon('description').classes('text-gray-600 text-sm shrink-0')
                with ui.column().classes('gap-0 grow min-w-0 leading-tight'):
                    displayed_offer_name = compact_name(offer.name, max_length=34)
                    ui.label(displayed_offer_name).classes('font-medium text-xs leading-tight')
                    add_name_tooltip(displayed_offer_name, offer.name)
                    displayed_project_name = compact_name(offer.project_name, max_length=30)
                    ui.label(displayed_project_name).classes('text-[11px] leading-tight text-gray-500')
                    add_name_tooltip(displayed_project_name, offer.project_name)
                self.offer_status_icon(offer, status)

            if status.extract_requested:
                self.extraction_progress_row(offer, status)

    def extraction_progress_row(self, offer: Offer, status) -> None:
        fraction = extraction_progress_fraction(status.status_step)
        with ui.row().classes('items-center gap-1 w-full no-wrap mt-1'):
            progress = ui.linear_progress(value=fraction or 0.0, show_value=False, size='4px').classes('grow')
            if fraction is None:
                progress.props('indeterminate')

            elapsed = format_elapsed_seconds(
                status.extraction_status.get('started_at') if status.extraction_status else None
            )
            if elapsed:
                ui.label(elapsed).classes('text-[10px] text-gray-500 shrink-0')

            cancel_button = ui.button(
                icon='cancel',
                on_click=lambda offer=offer: self.cancel_extraction(offer),
            ).props('flat dense round size=xs color=negative')
            cancel_button.tooltip('Extractie annuleren')
            cancel_button.on('click.stop', lambda: None)

    def offer_status_icon(self, offer: Offer, status) -> None:
        icon_name, icon_color, tooltip = self.status_icon_spec(status)
        if icon_name is None:
            return

        status_icon = ui.icon(icon_name).classes(f'{icon_color} text-sm shrink-0 cursor-pointer')
        if tooltip:
            status_icon.tooltip(tooltip)
        status_icon.on(
            'click.stop',
            lambda offer=offer, status=status: dialogs.offer_status_dialog(offer, status.extraction_status),
        )

    @staticmethod
    def status_icon_spec(status) -> tuple[str | None, str, str | None]:
        if status.result_exists:
            return 'check_circle', 'text-green-700', 'Extractiestatus bekijken'
        if status.extract_requested:
            return 'hourglass_empty', 'text-orange-700', status.status_message or status.status_step or 'Extractiestatus bekijken'
        if status.extraction_failed:
            return 'warning', 'text-red-700', status.status_error or status.status_message or 'Extractie mislukt — status bekijken'
        if status.status_stale:
            return 'warning', 'text-orange-700', 'Extractiestatus is verouderd — status bekijken'
        if status.extraction_status and status.extraction_status.get('status') == 'cancelled':
            return 'block', 'text-gray-500', 'Extractie geannuleerd — status bekijken'
        return None, '', None
