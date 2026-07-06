from typing import TYPE_CHECKING

from services.folder_handler import UNASSIGNED_PROJECT
from services.project import Project

if TYPE_CHECKING:
    from services.folder_handler import FolderHandler


class ProjectService:
    def __init__(self, folder_handler: 'FolderHandler') -> None:
        self.folder_handler = folder_handler

    def list_projects(self) -> list[Project]:
        return self.folder_handler.projects()

    def create(self, project_name: str | None) -> Project:
        return self.folder_handler.create_project(project_name)

    def rename(self, project: Project, new_name: str | None) -> Project:
        return project.rename(new_name)

    def delete(self, project: Project) -> None:
        project.delete()

    def first_project_name_or_unassigned(self) -> str:
        projects = self.list_projects()
        return projects[0].name if projects else UNASSIGNED_PROJECT

    def load_status(self, project: Project) -> dict | None:
        try:
            return self.folder_handler.load_comparison_status(project)
        except (FileNotFoundError, ValueError, OSError):
            return None
