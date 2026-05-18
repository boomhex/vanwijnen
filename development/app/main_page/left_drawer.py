from pathlib import Path
from typing import Callable

from nicegui import events, run, ui

from main_page.folder_handler import FolderHandler
from main_page.page_state import MainPageState


class LeftDrawer:
    def __init__(
        self,
        *,
        state: MainPageState,
        folder_handler: FolderHandler,
        pdf_dir: Path,
        refresh_right_side: Callable[[], None],
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.pdf_dir = pdf_dir
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

    def file_list(self) -> None:
        project_names = [project.name for project in self.folder_handler.projects()]

        with ui.row().classes('items-end gap-2 w-full no-wrap'):
            project_input = ui.input('New project').classes('grow')
            ui.button('Add', on_click=lambda: self.create_project(project_input.value)).props('dense')

        if project_names:
            ui.select(
                project_names,
                label='Upload to project',
                value=self.state.upload_project,
                on_change=lambda event: setattr(self.state, 'upload_project', event.value),
            ).classes('w-full')

        ui.label('Projects').classes('text-lg font-bold mb-2')
        projects = self.folder_handler.projects()
        root_files = self.folder_handler.files_in_folder(self.pdf_dir)

        if not projects and not root_files:
            ui.label('No projects or PDFs yet').classes('text-gray-500')
            return

        for project in projects:
            self.project_item(project)

        if root_files:
            with ui.expansion('Unassigned', icon='folder_open').classes('w-full'):
                for file in root_files:
                    self.file_item(file)

    def create_project(self, project_name: str | None) -> None:
        try:
            project = self.folder_handler.create_project(project_name)
        except ValueError as error:
            ui.notify(str(error))
            return

        self.state.upload_project = project.name
        self.schedule_refresh()

    def project_item(self, project: Path) -> None:
        with ui.expansion(project.name, icon='folder').classes('w-full'):
            with ui.row().classes('items-center justify-between gap-2 w-full no-wrap'):
                ui.label(project.name).classes('font-medium')
                ui.button(
                    'Compare',
                    icon='compare_arrows',
                    on_click=lambda selected_project=project: self.open_project_comparison(selected_project),
                ).props('flat dense no-caps size=sm')

            project_files = self.folder_handler.project_files(project)
            if not project_files:
                ui.label('No PDFs in this project').classes('text-gray-500 pl-8')
                return

            for file in project_files:
                self.file_item(file)

    def file_item(self, file: Path) -> None:
        result_exists = self.folder_handler.result_path_for_file(file).exists()
        extract_requested = file in self.state.extract_requested_files

        with ui.card().classes('w-full ml-6 my-1 p-2 gap-1'):
            with ui.row().classes('items-start gap-2 w-full no-wrap'):
                ui.icon('description').classes('text-gray-600 mt-1 text-sm')
                with ui.column().classes('gap-0 grow min-w-0'):
                    ui.label(file.name).classes('font-medium text-sm break-all')
                    ui.label(file.parent.name if file.parent != self.pdf_dir else 'Unassigned').classes('text-xs text-gray-500')
                if result_exists:
                    ui.icon('check_circle').classes('text-green-700 mt-1 text-sm')
                elif extract_requested:
                    ui.icon('hourglass_empty').classes('text-orange-700 mt-1 text-sm')

            with ui.row().classes('items-center gap-1 w-full no-wrap'):
                ui.button('Open', icon='visibility', on_click=lambda selected_file=file: self.open_file(selected_file)).props('flat dense no-caps size=sm')
                extract_button = ui.button(
                    'Requested' if extract_requested else 'Extract',
                    icon='task_alt' if result_exists else 'text_snippet',
                ).props('flat dense no-caps size=sm')
                if result_exists or extract_requested:
                    extract_button.props('disable')
                else:
                    async def request_extract(_event, selected_file=file, button=extract_button):
                        await self.extract_file(selected_file, button)

                    extract_button.on('click', request_extract)
                self.rename_button(file)
                self.move_button(file)
                ui.button('Delete', icon='delete', on_click=lambda selected_file=file: self.delete_file(selected_file)).props('flat dense no-caps size=sm color=negative')

    def open_file(self, file: Path) -> None:
        self.state.opened_file = file
        self.state.current_view = 'offer'
        self.refresh_right_side()
        ui.notify(f'Opened {file.name}')

    def open_project_comparison(self, project: Path) -> None:
        self.state.comparison_project = project
        self.state.current_view = 'comparison'
        self.refresh_right_side()

    def delete_file(self, file: Path) -> None:
        try:
            self.folder_handler.delete_file(file)
        except FileNotFoundError as error:
            ui.notify(str(error))
            self.schedule_refresh()
            return

        if self.state.opened_file == file:
            self.state.opened_file = None
            self.schedule_right_side_refresh()

        ui.notify(f'Deleted {file.name}')
        self.schedule_refresh()

    def rename_file(self, file: Path, new_name: str | None) -> bool:
        try:
            new_file = self.folder_handler.rename_file(file, new_name)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            return False

        if self.state.opened_file == file:
            self.state.opened_file = new_file
            self.schedule_right_side_refresh()

        ui.notify(f'Renamed to {new_file.name}')
        self.schedule_refresh()
        return True

    def move_file(self, file: Path, target_project: str | None) -> bool:
        try:
            new_file = self.folder_handler.move_file(file, target_project)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            ui.notify(str(error))
            return False

        if self.state.opened_file == file:
            self.state.opened_file = new_file
            self.schedule_right_side_refresh()

        if file in self.state.extract_requested_files:
            self.state.extract_requested_files.discard(file)
            self.state.extract_requested_files.add(new_file)

        ui.notify(f'Moved {new_file.name}')
        self.schedule_refresh()
        return True

    def schedule_right_side_refresh(self) -> None:
        ui.timer(0.05, self.refresh_right_side, once=True)

    def rename_button(self, file: Path) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Rename {file.name}').classes('font-medium')
            name_input = ui.input('Filename', value=file.stem).classes('w-80')

            def save():
                if self.rename_file(file, name_input.value):
                    dialog.close()

            with ui.row().classes('justify-end w-full'):
                ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button('Save', on_click=save).props('dense no-caps size=sm')

        ui.button('Rename', icon='edit', on_click=dialog.open).props('flat dense no-caps size=sm')

    def move_button(self, file: Path) -> None:
        project_options = ['Unassigned'] + [project.name for project in self.folder_handler.projects()]
        current_project = 'Unassigned' if file.parent == self.pdf_dir else file.parent.name

        with ui.dialog() as dialog, ui.card():
            ui.label(f'Move {file.name}').classes('font-medium')
            project_select = ui.select(
                project_options,
                label='Project',
                value=current_project,
            ).classes('w-80')

            def save():
                if self.move_file(file, project_select.value):
                    dialog.close()

            with ui.row().classes('justify-end w-full'):
                ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
                ui.button('Move', on_click=save).props('dense no-caps size=sm')

        ui.button('Move', icon='drive_file_move', on_click=dialog.open).props('flat dense no-caps size=sm')

    async def extract_file(self, file: Path, button) -> None:
        self.state.extract_requested_files.add(file)
        button.set_text('Requested')
        button.props('loading disable')
        button.update()

        from main_page.extract_offer import extract_offer

        try:
            await run.io_bound(extract_offer, file, self.folder_handler.result_dir_for_file(file))
        except Exception as error:
            ui.notify(f'Could not extract {file.name}: {error}')
            self.state.extract_requested_files.discard(file)
        finally:
            self.schedule_refresh()
