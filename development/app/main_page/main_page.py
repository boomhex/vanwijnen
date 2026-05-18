from pathlib import Path

from nicegui import app, ui

from main_page.folder_handler import FolderHandler
from main_page.left_drawer import LeftDrawer
from main_page.page_state import MainPageState
from main_page.right_side import RightSide
from utils.eva_html import eva_html


PRIMARY_RED = '#B00000'
SECONDARY_RED = "#F9BFBF"
APP_DIR = Path(__file__).resolve().parents[1]
TMP_DIR = APP_DIR / 'tmp'
PDF_DIR = TMP_DIR / 'pdfs'
RESULTS_DIR = TMP_DIR / 'results'
COMPARISON_DIR = TMP_DIR / 'comparison'

PDF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

app.add_static_files('/pdfs', PDF_DIR)

state = MainPageState()
folder_handler = FolderHandler(PDF_DIR, RESULTS_DIR, COMPARISON_DIR)
right_side_component = RightSide(state=state, folder_handler=folder_handler, pdf_dir=PDF_DIR)
left_drawer_component = LeftDrawer(
    state=state,
    folder_handler=folder_handler,
    pdf_dir=PDF_DIR,
    refresh_right_side=right_side_component.refresh,
)


@ui.page('/')
def main_page():
    eva_html()
    ui.colors(primary=PRIMARY_RED)

    with ui.left_drawer().style(f'background-color: {SECONDARY_RED}'):
        left_drawer_component.render()

    right_side_component.render()
