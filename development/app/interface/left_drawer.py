import logging
from pathlib import Path
from typing import Callable

from nicegui import events, ui

logger = logging.getLogger(__name__)

from application.offer_service import OfferService
from application.extraction_job_service import ExtractionJobService
from domain.status import is_active_running, is_stale_running
from services.folder_handler import FolderHandler, UNASSIGNED_PROJECT
from services.offer import Offer
from services.project import Project
from interface.page_state import MainPageState, View


class LeftDrawer:
    def __init__(
        self,
        *,
        state: MainPageState,
        folder_handler: FolderHandler,
        extraction_job_service: ExtractionJobService,
        projects_dir: Path,
        refresh_right_side: Callable[[], None],
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.extraction_job_service = extraction_job_service
        self.offer_service = OfferService(folder_handler)
        self.projects_dir = projects_dir
        self.refresh_right_side = refresh_right_side
        self.file_list_container = None

    def render(self) -> None:
        ui.upload(on_upload=self.handle_upload, auto_upload=True).classes('w-full')
        self.file_list_container = ui.column().classes('w-full')
        with self.file_list_container:
            self.file_list()
        ui.timer(2.0, self.poll_extraction_statuses)

    async def handle_upload(self, event: events.UploadEventArguments) -> None:
        await self.folder_handler.add_uploaded_file(event, self.state.upload_project)
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if not self.file_list_container_is_live():
            return

        try:
            ui.timer(0.05, self.refresh_safe, once=True)
        except RuntimeError:
            pass

    def refresh(self) -> None:
        if not self.file_list_container_is_live():
            return

        self.file_list_container.clear()
        with self.file_list_container:
            self.file_list()

    def refresh_safe(self) -> None:
        try:
            self.refresh()
        except RuntimeError:
            pass

    def poll_extraction_statuses(self) -> None:
        if not self.file_list_container_is_live():
            return

        should_refresh = bool(self.state.extract_requested_offers)
        for project in self.folder_handler.projects():
            for offer in project.offers():
                status = self.load_extraction_status(offer)
                if self.extraction_job_service.is_running(offer) or is_active_running(status):
                    should_refresh = True
                    break
            if should_refresh:
                break

        if should_refresh:
            self.refresh_safe()

    def file_list_container_is_live(self) -> bool:
        return self.element_is_live(self.file_list_container)

    @staticmethod
    def element_is_live(element) -> bool:
        if element is None:
            return False
        if getattr(element, 'is_deleted', False):
            return False

        client = getattr(element, 'client', None)
        if client is None:
            return False

        return not getattr(client, '_deleted', False)

    @staticmethod
    def notify_safe(message: str) -> None:
        try:
            ui.notify(message)
        except RuntimeError:
            logger.info(message)

    @staticmethod
    def compact_name(name: str, max_length: int = 32, *, mode: str = 'middle') -> str:
        if len(name) <= max_length:
            return name

        if max_length <= 3:
            return name[:max_length]

        if mode == 'end':
            return f'{name[:max_length - 3]}...'

        keep = max_length - 3
        left = keep // 2
        right = keep - left
        return f'{name[:left]}...{name[-right:]}'

    @staticmethod
    def add_name_tooltip(displayed_name: str, full_name: str) -> None:
        if displayed_name != full_name:
            ui.tooltip(full_name)

    def file_list(self) -> None:
        project_names = [project.name for project in self.folder_handler.projects()]
        upload_options = project_names if project_names else [UNASSIGNED_PROJECT]

        with ui.row().classes('items-end gap-2 w-full no-wrap'):
            project_input = ui.input('New project').classes('grow')
            ui.button(icon='eva-folder-add-outline', on_click=lambda: self.create_project(project_input.value))

        ui.select(
            upload_options,
            label='Upload to project',
            value=self.state.upload_project,
            on_change=lambda event: setattr(self.state, 'upload_project', event.value),
        ).classes('w-full')

        ui.label('Projects').classes('text-lg font-bold mb-2')
        projects = self.folder_handler.projects()

        if not projects:
            ui.label('No projects or PDFs yet').classes('text-gray-500')
            return

        for project in projects:
            self.project_item(project)

    def create_project(self, project_name: str | None) -> None:
        try:
            project = self.folder_handler.create_project(project_name)
        except ValueError as error:
            ui.notify(str(error))
            return

        self.state.upload_project = project.name
        self.schedule_refresh()

    def project_item(self, project: Project) -> None:
        project_label = self.compact_name(project.name, max_length=24, mode='end')
        expansion = ui.expansion(project_label, icon='folder').classes('w-full')
        if project_label != project.name:
            expansion.tooltip(project.name)

        with expansion:
            with ui.row().classes('items-center justify-between gap-2 w-full no-wrap'):
                with ui.row().classes('items-center gap-1 no-wrap'):
                    comparison_status = self.load_comparison_status(project)
                    comparison_running = is_active_running(comparison_status)
                    comparison_stale = is_stale_running(comparison_status)
                    compare_button = ui.button(
                        'Compare',
                        icon='compare_arrows',
                        on_click=lambda selected_project=project: self.open_project_comparison(selected_project),
                    ).props('flat dense no-caps size=sm')
                    if comparison_running:
                        compare_button.set_text('Matching')
                        compare_button.props('loading disable')
                        compare_button.tooltip(comparison_status.get('message') or comparison_status.get('step'))
                    elif comparison_stale:
                        stale_icon = ui.icon('warning').classes('text-orange-700')
                        stale_icon.tooltip('Comparison status is stale. You can retry matching.')
                    self.rename_project_button(project)
                    self.delete_project_button(project)

            offers = project.offers()
            if not offers:
                ui.label('No offers in this project').classes('text-gray-500 pl-8')
                return

            for offer in offers:
                self.file_item(offer)

    def file_item(self, offer: Offer) -> None:
        result_exists = offer.extract_path.exists()
        extraction_status = self.load_extraction_status(offer)
        status = extraction_status.get('status') if extraction_status else None
        status_step = extraction_status.get('step') if extraction_status else None
        status_message = extraction_status.get('message') if extraction_status else None
        status_error = extraction_status.get('error') if extraction_status else None
        status_stale = is_stale_running(extraction_status)
        job_running = self.extraction_job_service.is_running(offer)
        status_running = is_active_running(extraction_status)
        if not job_running and not status_running:
            self.state.extract_requested_offers.discard(offer)

        extract_requested = job_running or status_running
        extraction_failed = status == 'failed'

        with ui.card().classes('w-full ml-6 my-1 p-2 gap-1'):
            with ui.row().classes('items-start gap-2 w-full no-wrap'):
                ui.icon('description').classes('text-gray-600 mt-1 text-sm')
                with ui.column().classes('gap-0 grow min-w-0'):
                    displayed_offer_name = self.compact_name(offer.name, max_length=34)
                    ui.label(displayed_offer_name).classes('font-medium text-sm')
                    self.add_name_tooltip(displayed_offer_name, offer.name)
                    displayed_project_name = self.compact_name(offer.project_name, max_length=30)
                    ui.label(displayed_project_name).classes('text-xs text-gray-500')
                    self.add_name_tooltip(displayed_project_name, offer.project_name)
                if result_exists:
                    ui.icon('check_circle').classes('text-green-700 mt-1 text-sm')
                elif extract_requested:
                    status_icon = ui.icon('hourglass_empty').classes('text-orange-700 mt-1 text-sm')
                    if status_step or status_message:
                        status_icon.tooltip(status_message or status_step)
                elif extraction_failed:
                    status_icon = ui.icon('warning').classes('text-red-700 mt-1 text-sm')
                    status_icon.tooltip(status_error or status_message or 'Extraction failed')
                elif status_stale:
                    status_icon = ui.icon('warning').classes('text-orange-700 mt-1 text-sm')
                    status_icon.tooltip('Extraction status is stale. You can retry extraction.')

            with ui.row().classes('items-center gap-1 w-full no-wrap'):
                ui.button('Open', icon='visibility', on_click=lambda selected_offer=offer: self.open_file(selected_offer)).props('flat dense no-caps size=sm')
                extract_button = ui.button(
                    'Requested' if extract_requested else 'Extract',
                    icon='task_alt' if result_exists else 'text_snippet',
                ).props('flat dense no-caps size=sm')
                if result_exists or extract_requested:
                    extract_button.props('disable')
                else:
                    def request_extract(_event, selected_offer=offer):
                        self.request_extract(selected_offer)

                    extract_button.on('click', request_extract)
                self.rename_button(offer)
                self.move_button(offer)
                ui.button('Delete', icon='delete', on_click=lambda selected_offer=offer: self.delete_file(selected_offer)).props('flat dense no-caps size=sm color=negative')

    def load_extraction_status(self, offer: Offer) -> dict | None:
        try:
            return self.folder_handler.load_extraction_status(offer.document)
        except (FileNotFoundError, ValueError, OSError):
            return None

    def load_comparison_status(self, project: Project) -> dict | None:
        try:
            return self.folder_handler.load_comparison_status(project)
        except (FileNotFoundError, ValueError, OSError):
            return None

    def open_file(self, offer: Offer) -> None:
        self.state.opened_offer = offer
        self.state.current_view = View.OFFER
        self.refresh_right_side()
        ui.notify(f'Opened {offer.name}')

    def open_project_comparison(self, project: Project) -> None:
        self.state.comparison_project = project
        self.state.current_view = View.COMPARISON
        self.refresh_right_side()

    def delete_file(self, offer: Offer) -> None:
        try:
            self.offer_service.delete(offer)
        except FileNotFoundError as error:
            ui.notify(str(error))
            self.schedule_refresh()
            return

        if self.state.opened_offer == offer:
            self.state.opened_offer = None
            self.schedule_right_side_refresh()

        ui.notify(f'Deleted {offer.name}')
        self.schedule_refresh()

    def rename_file(self, offer: Offer, new_name: str | None) -> bool:
        try:
            new_offer = self.offer_service.rename(offer, new_name)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            return False

        if self.state.opened_offer == offer:
            self.state.opened_offer = new_offer
            self.schedule_right_side_refresh()

        ui.notify(f'Renamed to {new_offer.name}')
        self.schedule_refresh()
        return True

    def rename_project(self, project: Project, new_name: str | None) -> bool:
        old_name = project.name

        try:
            new_project = project.rename(new_name)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            return False

        if self.state.upload_project == old_name:
            self.state.upload_project = new_project.name

        if self.state.comparison_project == project:
            self.state.comparison_project = new_project
            self.schedule_right_side_refresh()

        if self.state.opened_offer is not None and self.state.opened_offer.project_name == old_name:
            new_offer_path = new_project.path / self.state.opened_offer.name
            self.state.opened_offer = self.folder_handler.offer_from_path(new_offer_path)
            self.schedule_right_side_refresh()

        updated_requested_offers = set()
        for offer in self.state.extract_requested_offers:
            if offer.project_name == old_name:
                updated_requested_offers.add(self.folder_handler.offer_from_path(new_project.path / offer.name))
            else:
                updated_requested_offers.add(offer)
        self.state.extract_requested_offers = updated_requested_offers

        ui.notify(f'Renamed {old_name} to {new_project.name}')
        self.schedule_refresh()
        return True

    def delete_project(self, project: Project) -> bool:
        project_name = project.name

        try:
            project.delete()
        except (FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            self.schedule_refresh()
            return False

        if self.state.upload_project == project_name:
            remaining_projects = self.folder_handler.projects()
            self.state.upload_project = remaining_projects[0].name if remaining_projects else UNASSIGNED_PROJECT

        if self.state.comparison_project == project:
            self.state.comparison_project = None
            if self.state.current_view == View.COMPARISON:
                self.state.current_view = View.OFFER
            self.schedule_right_side_refresh()

        if self.state.opened_offer is not None and self.state.opened_offer.project_name == project_name:
            self.state.opened_offer = None
            self.schedule_right_side_refresh()

        self.state.extract_requested_offers = {
            offer
            for offer in self.state.extract_requested_offers
            if offer.project_name != project_name
        }

        ui.notify(f'Deleted {project_name}')
        self.schedule_refresh()
        return True

    def move_file(self, offer: Offer, target_project: str | None) -> bool:
        try:
            new_offer = self.offer_service.move_to_project(offer, target_project)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            return False

        if self.state.opened_offer == offer:
            self.state.opened_offer = new_offer
            self.schedule_right_side_refresh()

        if offer in self.state.extract_requested_offers:
            self.state.extract_requested_offers.discard(offer)
            self.state.extract_requested_offers.add(new_offer)

        ui.notify(f'Moved {new_offer.name}')
        self.schedule_refresh()
        return True

    def schedule_right_side_refresh(self) -> None:
        try:
            ui.timer(0.05, self.refresh_right_side, once=True)
        except RuntimeError:
            pass

    def rename_button(self, offer: Offer) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Rename {offer.name}').classes('font-medium')
            name_input = ui.input('Filename', value=offer.name).classes('w-80')

            def save():
                if self.rename_file(offer, name_input.value):
                    dialog.close()

            with ui.row().classes('justify-end w-full'):
                ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button('Save', on_click=save).props('dense no-caps size=sm')

        ui.button('Rename', icon='edit', on_click=dialog.open).props('flat dense no-caps size=sm')

    def rename_project_button(self, project: Project) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Rename {project.name}').classes('font-medium')
            name_input = ui.input('Project folder', value=project.name).classes('w-80')

            def save():
                if self.rename_project(project, name_input.value):
                    dialog.close()

            with ui.row().classes('justify-end w-full'):
                ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button('Save', on_click=save).props('dense no-caps size=sm')

        ui.button('Rename', icon='drive_file_rename_outline', on_click=dialog.open).props('flat dense no-caps size=sm')

    def delete_project_button(self, project: Project) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Delete {project.name}?').classes('font-medium')
            ui.label('This removes the project folder, all offers, and the comparison.').classes('text-sm text-gray-600')

            def confirm():
                if self.delete_project(project):
                    dialog.close()

            with ui.row().classes('justify-end w-full'):
                ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button('Delete', icon='delete', on_click=confirm).props('dense no-caps size=sm color=negative')

        ui.button('Delete', icon='delete', on_click=dialog.open).props('flat dense no-caps size=sm color=negative')

    def move_button(self, offer: Offer) -> None:
        project_options = [project.name for project in self.folder_handler.projects()]
        if UNASSIGNED_PROJECT not in project_options:
            project_options.insert(0, UNASSIGNED_PROJECT)
        current_project = offer.project_name

        with ui.dialog() as dialog, ui.card():
            ui.label(f'Move {offer.name}').classes('font-medium')
            project_select = ui.select(
                project_options,
                label='Project',
                value=current_project,
            ).classes('w-80')

            def save():
                if self.move_file(offer, project_select.value):
                    dialog.close()

            with ui.row().classes('justify-end w-full'):
                ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button('Move', on_click=save).props('dense no-caps size=sm')

        ui.button('Move', icon='drive_file_move', on_click=dialog.open).props('flat dense no-caps size=sm')

    def request_extract(self, offer: Offer) -> None:
        if self.extraction_job_service.is_running(offer):
            ui.notify(f'Extraction already running for {offer.name}')
            return

        started = self.extraction_job_service.start(offer)
        if not started:
            ui.notify(f'Extraction already running for {offer.name}')
            return

        self.state.extract_requested_offers.add(offer)
        ui.notify(f'Extraction started for {offer.name}')
        self.schedule_refresh()
