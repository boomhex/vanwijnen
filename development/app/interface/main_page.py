from pathlib import Path

from nicegui import app, ui

from application.extraction_job_service import ExtractionJobService
from services.folder_handler import FolderHandler
from interface.left_drawer import LeftDrawer
from interface.page_state import MainPageState
from interface.right_side.right_side import RightSide
from utils.eva_html import eva_html


PRIMARY_RED = '#B00000'
SECONDARY_RED = "#F9BFBF"
APP_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = APP_DIR / 'storage'
PROJECTS_DIR = STORAGE_DIR

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

app.add_static_files('/storage', PROJECTS_DIR)

state = MainPageState()
folder_handler = FolderHandler(PROJECTS_DIR)
extraction_job_service = ExtractionJobService(folder_handler)
right_side_component = RightSide(state=state, folder_handler=folder_handler, projects_dir=PROJECTS_DIR)
left_drawer_component = LeftDrawer(
    state=state,
    folder_handler=folder_handler,
    extraction_job_service=extraction_job_service,
    projects_dir=PROJECTS_DIR,
    refresh_right_side=right_side_component.refresh,
)


@ui.page('/')
def main_page():
    eva_html()
    ui.colors(primary=PRIMARY_RED)

    with ui.left_drawer().style(f'background-color: {SECONDARY_RED}'):
        left_drawer_component.render()

    right_side_component.render()
