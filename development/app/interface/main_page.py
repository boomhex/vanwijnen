from pathlib import Path

from fastapi.responses import RedirectResponse
from nicegui import app, ui

from application.extraction_job_service import ExtractionJobService
from services import auth
from services.folder_handler import FolderHandler
from interface.left_drawer import LeftDrawer
from interface.page_state import MainPageState
from interface.right_side.right_side import RightSide
from interface.session_panel import render_session_panel
from interface.theme import PRIMARY_RED, SECONDARY_RED
from utils.app_logging import log_action
from utils.eva_html import eva_html

APP_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = APP_DIR / 'storage'
PROJECTS_DIR = STORAGE_DIR

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

app.add_static_files('/storage', PROJECTS_DIR)

# Shared between all clients: one extraction job queue. Its handler is rooted
# at the storage root but only performs offer-relative reads/writes, so it
# works for offers in any user's workspace.
extraction_job_service = ExtractionJobService(FolderHandler(PROJECTS_DIR))


@ui.page('/')
def main_page():
    # Users work inside a selected workspace at storage/<workspace>/;
    # AuthMiddleware blocks /storage requests outside granted workspaces.
    username = app.storage.user.get('username')
    if not auth.is_valid_username(username):
        app.storage.user.clear()
        return RedirectResponse('/login')

    workspace = app.storage.user.get('workspace')
    if not auth.workspace_authorized(username, workspace):
        return RedirectResponse('/workspaces')

    folder_handler = FolderHandler(PROJECTS_DIR / workspace)

    # Built per client so every user has their own selection state and UI
    # element tree. projects_dir stays the storage root so PDF links resolve
    # to /storage/<workspace>/... matching the static mount above.
    state = MainPageState()
    right_side_component = RightSide(
        state=state,
        folder_handler=folder_handler,
        projects_dir=PROJECTS_DIR,
    )
    left_drawer_component = LeftDrawer(
        state=state,
        folder_handler=folder_handler,
        extraction_job_service=extraction_job_service,
        projects_dir=PROJECTS_DIR,
        refresh_right_side=right_side_component.refresh,
    )

    log_action('page_loaded')
    eva_html()
    ui.colors(primary=PRIMARY_RED)

    with ui.left_drawer().style(f'background-color: {SECONDARY_RED}'):
        left_drawer_component.render()
        render_session_panel()

    right_side_component.render()
