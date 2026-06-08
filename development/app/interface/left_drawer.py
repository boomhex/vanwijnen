from pathlib import Path
from typing import Callable

from nicegui import events, run, ui

from services.folder_handler import FolderHandler, UNASSIGNED_PROJECT
from services.offer import Offer
from services.project import Project
from interface.page_state import MainPageState


class LeftDrawer:
    def __init__(
        self,
        *,
        state: MainPageState,
        folder_handler: FolderHandler,
        projects_dir: Path,
        refresh_right_side: Callable[[], None],
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.projects_dir = projects_dir
        self.refresh_right_side = refresh_right_side
        self.file_list_container = None

    def render(self) -> None:
        ui.upload(on_upload=self.handle_upload, auto_upload=True).classes('w-full')
        self.file_list_container = ui.column().classes('w-full')
        with self.file_list_container:
            self.file_list()

    async def handle_upload(self, event: events.UploadEventArguments) -> None:
        await self.folder_handler.add_uploaded_file(event, self.state.upload_project)
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        ui.timer(0.05, self.refresh, once=True)

    def refresh(self) -> None:
        if self.file_list_container is None:
            return

        self.file_list_container.clear()
        with self.file_list_container:
            self.file_list()

    def refresh_safe(self) -> None:
        try:
            self.refresh()
        except RuntimeError:
            pass

    @staticmethod
    def notify_safe(message: str) -> None:
        try:
            ui.notify(message)
        except RuntimeError:
            print(message)

    def file_list(self) -> None:
        project_names = [project.name for project in self.folder_handler.projects()]
        upload_options = project_names if project_names else [UNASSIGNED_PROJECT]

        with ui.row().classes('items-end gap-2 w-full no-wrap'):
            project_input = ui.input('New project').classes('grow')
            ui.button('Add', on_click=lambda: self.create_project(project_input.value)).props('dense')

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
        with ui.expansion(project.name, icon='folder').classes('w-full'):
            with ui.row().classes('items-center justify-between gap-2 w-full no-wrap'):
                ui.label(project.name).classes('font-medium')
                ui.button(
                    'Compare',
                    icon='compare_arrows',
                    on_click=lambda selected_project=project: self.open_project_comparison(selected_project),
                ).props('flat dense no-caps size=sm')

            offers = project.offers()
            if not offers:
                ui.label('No offers in this project').classes('text-gray-500 pl-8')
                return

            for offer in offers:
                self.file_item(offer)

    def file_item(self, offer: Offer) -> None:
        result_exists = offer.extract_path.exists()
        extract_requested = offer in self.state.extract_requested_offers

        with ui.card().classes('w-full ml-6 my-1 p-2 gap-1'):
            with ui.row().classes('items-start gap-2 w-full no-wrap'):
                ui.icon('description').classes('text-gray-600 mt-1 text-sm')
                with ui.column().classes('gap-0 grow min-w-0'):
                    ui.label(offer.name).classes('font-medium text-sm break-all')
                    ui.label(offer.project_name).classes('text-xs text-gray-500')
                if result_exists:
                    ui.icon('check_circle').classes('text-green-700 mt-1 text-sm')
                elif extract_requested:
                    ui.icon('hourglass_empty').classes('text-orange-700 mt-1 text-sm')

            with ui.row().classes('items-center gap-1 w-full no-wrap'):
                ui.button('Open', icon='visibility', on_click=lambda selected_offer=offer: self.open_file(selected_offer)).props('flat dense no-caps size=sm')
                extract_button = ui.button(
                    'Requested' if extract_requested else 'Extract',
                    icon='task_alt' if result_exists else 'text_snippet',
                ).props('flat dense no-caps size=sm')
                if result_exists or extract_requested:
                    extract_button.props('disable')
                else:
                    async def request_extract(_event, selected_offer=offer, button=extract_button):
                        await self.extract_file(selected_offer, button)

                    extract_button.on('click', request_extract)
                self.rename_button(offer)
                self.move_button(offer)
                ui.button('Delete', icon='delete', on_click=lambda selected_offer=offer: self.delete_file(selected_offer)).props('flat dense no-caps size=sm color=negative')

    def open_file(self, offer: Offer) -> None:
        self.state.opened_offer = offer
        self.state.current_view = 'offer'
        self.refresh_right_side()
        ui.notify(f'Opened {offer.name}')

    def open_project_comparison(self, project: Project) -> None:
        self.state.comparison_project = project
        self.state.current_view = 'comparison'
        self.refresh_right_side()

    def delete_file(self, offer: Offer) -> None:
        try:
            offer.delete()
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
            new_offer = offer.rename(new_name)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            return False

        if self.state.opened_offer == offer:
            self.state.opened_offer = new_offer
            self.schedule_right_side_refresh()

        ui.notify(f'Renamed to {new_offer.name}')
        self.schedule_refresh()
        return True

    def move_file(self, offer: Offer, target_project: str | None) -> bool:
        try:
            new_offer = offer.move_to_project(target_project)
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
        ui.timer(0.05, self.refresh_right_side, once=True)

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

    async def extract_file(self, offer: Offer, button) -> None:
        self.state.extract_requested_offers.add(offer)
        button.set_text('Requested')
        button.props('loading disable')
        button.update()

        from services.extract_offer import extract_offer

        try:
            await run.io_bound(extract_offer, offer.document, self.folder_handler)
        except Exception as error:
            self.notify_safe(f'Could not extract {offer.name}: {error}')
        finally:
            self.state.extract_requested_offers.discard(offer)
            self.refresh_safe()
