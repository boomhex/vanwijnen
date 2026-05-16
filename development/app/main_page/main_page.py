from nicegui import ui, events
from utils.eva_html import eva_html
from pathlib import Path
from dataclasses import dataclass

PRIMARY_RED = '#B00000'
SECONDARY_RED = "#F9BFBF"


class State:
    def __init__(self, file_path: Path) -> None:
        self._file_path: Path = file_path
        self._file_path.mkdir(exist_ok=True)
        self._opened_file = None

        self._file_list = self.file_list_from_folder()

    @property
    def opened_file(self):
        return self._opened_file

    @opened_file.setter
    def opened_file(self, filepath: Path=None):
        # TODO: Check validity of file before opening
        self._opened_file = filepath

    def file_list_from_folder(self):
        return list(self._file_path.glob('files/*.pdf'))

    async def add_file(self, event: events.UploadEventArguments):
        await event.file.save(self._file_path / "files" / event.file.name)

state = State(Path("./tmp"))

async def handle_upload(e: events.UploadEventArguments):
    await state.add_file(e)
    refresh_file_list()


def left_drawer():
    file_upload = ui.upload(on_upload=handle_upload, auto_upload=True).classes('w-full')


def open_file(file: Path):
    state.opened_file = file
    ui.notify(f'Opened {file.name}')


def refresh_file_list():
    global file_list_container
    file_list_container.clear()

    with file_list_container:
        file_list()


def file_list():
    ui.label('Files').classes('text-lg font-bold mb-2')
    files = state.file_list_from_folder()
    if not files:
        ui.label('No files uploaded yet').classes('text-gray-500')
        return
    for file in files:
        ui.button(
            file.name,
            on_click=lambda f=file: open_file(f),
        )


def opened_file():
    ui.label('Opened file placeholder')


def right_side():
    global file_list_container

    with ui.column().classes('w-full h-screen'):
        with ui.splitter(limits=(10, 90)).classes('w-full h-full') as splitter:
            with splitter.before:
                file_list_container = ui.column().classes('w-full')
                with file_list_container:
                    file_list()
            with splitter.after:
                opened_file()


ui.page('/')
def main_page():
    eva_html()
    ui.colors(primary=PRIMARY_RED)

    with ui.left_drawer().style(f'background-color: {SECONDARY_RED}'):
        left_drawer()

    right_side()
