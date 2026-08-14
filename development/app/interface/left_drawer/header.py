from collections.abc import Callable

from nicegui import ui

from application.project_service import ProjectService
from interface.page_state import MainPageState
from services.folder_handler import UNASSIGNED_PROJECT


class DrawerHeader:
    def __init__(
        self,
        *,
        state: MainPageState,
        project_service: ProjectService,
        create_project: Callable[[str | None], None],
    ) -> None:
        self.state = state
        self.project_service = project_service
        self.create_project = create_project

    def render(self) -> None:
        project_names = [project.name for project in self.project_service.list_projects()]
        upload_options = project_names if project_names else [UNASSIGNED_PROJECT]

        with ui.row().classes('items-end gap-2 w-full no-wrap'):
            project_input = ui.input('New project').classes('grow')
            project_input.on('keydown.enter', lambda: self.create_project(project_input.value))
            ui.button(icon='eva-folder-add-outline', on_click=lambda: self.create_project(project_input.value))

        ui.select(
            upload_options,
            label='Upload to project',
            value=self.state.upload_project,
            on_change=lambda event: setattr(self.state, 'upload_project', event.value),
        ).classes('w-full')
