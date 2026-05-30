from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Callable

from nicegui import ui

from interface.page_state import MainPageState
from services.folder_handler import FolderHandler

class SubPage(ABC):
    def __init__(
        self,
        state: MainPageState,
        folder_handler: FolderHandler,
        projects_dir: Path,
        refresh: Callable[[], None] | None = None,
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.projects_dir = projects_dir
        self.refresh_callback = refresh

        self.container = None

    @abstractmethod
    def render(self) -> None:
       pass

    @abstractmethod
    def show(self) -> None:
        pass

    def save_result(self, file: Path, result: dict) -> None:
        self.folder_handler.save_result(file, result)

    def refresh(self) -> None:
        if self.refresh_callback is not None:
            self.refresh_callback()

    @staticmethod
    def notify_safe(message: str) -> None:
        try:
            ui.notify(message)
        except RuntimeError:
            print(message)

