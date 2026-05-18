from nicegui import ui, events, app, run
from utils.eva_html import eva_html
from pathlib import Path
from urllib.parse import quote
import json
from main_page.editable_table_helper import render_editable_rows, render_editable_table
from main_page.folder_handler import FolderHandler

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

class State:
    def __init__(self) -> None:
        self.opened_file: Path | None = None
        self.current_view = 'offer'
        self.comparison_project: Path | None = None
        self.upload_project: str | None = None
        self.extract_requested_files: set[Path] = set()


folder_handler = FolderHandler(PDF_DIR, RESULTS_DIR, COMPARISON_DIR)
state = State()
opened_file_container = None
file_list_container = None


async def handle_upload(e: events.UploadEventArguments):
    await folder_handler.add_uploaded_file(e, state.upload_project)
    schedule_file_list_refresh()


def left_drawer():
    global file_list_container

    ui.upload(on_upload=handle_upload, auto_upload=True).classes('w-full')
    file_list_container = ui.column().classes('w-full')
    with file_list_container:
        file_list()


def open_file(file: Path):
    state.opened_file = file
    state.current_view = 'offer'
    refresh_opened_file()
    ui.notify(f'Opened {file.name}')


def open_project_comparison(project: Path):
    state.comparison_project = project
    state.current_view = 'comparison'
    refresh_opened_file()


def delete_file(file: Path):
    try:
        folder_handler.delete_file(file)
    except FileNotFoundError as error:
        ui.notify(str(error))
        schedule_file_list_refresh()
        return

    if state.opened_file == file:
        state.opened_file = None
        schedule_opened_file_refresh()

    ui.notify(f'Deleted {file.name}')
    schedule_file_list_refresh()


def rename_file(file: Path, new_name: str | None):
    try:
        new_file = folder_handler.rename_file(file, new_name)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        ui.notify(str(error))
        return False

    if state.opened_file == file:
        state.opened_file = new_file
        schedule_opened_file_refresh()

    ui.notify(f'Renamed to {new_file.name}')
    schedule_file_list_refresh()
    return True


def move_file(file: Path, target_project: str | None):
    try:
        new_file = folder_handler.move_file(file, target_project)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        ui.notify(str(error))
        return False

    if state.opened_file == file:
        state.opened_file = new_file
        schedule_opened_file_refresh()

    if file in state.extract_requested_files:
        state.extract_requested_files.discard(file)
        state.extract_requested_files.add(new_file)

    ui.notify(f'Moved {new_file.name}')
    schedule_file_list_refresh()
    return True


def schedule_file_list_refresh():
    ui.timer(0.05, refresh_file_list, once=True)


def schedule_opened_file_refresh():
    ui.timer(0.05, refresh_opened_file, once=True)


def notify_safe(message: str):
    try:
        ui.notify(message)
    except RuntimeError:
        print(message)


def schedule_opened_file_refresh_safe():
    try:
        schedule_opened_file_refresh()
    except RuntimeError:
        pass


def refresh_file_list():
    global file_list_container
    file_list_container.clear()

    with file_list_container:
        file_list()


def create_project(project_name: str | None):
    try:
        project = folder_handler.create_project(project_name)
    except ValueError as error:
        ui.notify(str(error))
        return

    state.upload_project = project.name
    schedule_file_list_refresh()


def result_dir_for_file(file: Path) -> Path:
    return folder_handler.result_dir_for_file(file)


async def extract_file(file: Path, button):
    state.extract_requested_files.add(file)
    button.set_text('Requested')
    button.props('loading disable')
    button.update()

    from main_page.extract_offer import extract_offer

    try:
        await run.io_bound(extract_offer, file, result_dir_for_file(file))
    except Exception as error:
        ui.notify(f'Could not extract {file.name}: {error}')
        state.extract_requested_files.discard(file)
    finally:
        schedule_file_list_refresh()
    

def file_list():
    project_names = [project.name for project in folder_handler.projects()]

    with ui.row().classes('items-end gap-2 w-full no-wrap'):
        project_input = ui.input('New project').classes('grow')
        ui.button('Add', on_click=lambda: create_project(project_input.value)).props('dense')

    if project_names:
        ui.select(
            project_names,
            label='Upload to project',
            value=state.upload_project,
            on_change=lambda e: setattr(state, 'upload_project', e.value),
        ).classes('w-full')

    ui.label('Projects').classes('text-lg font-bold mb-2')
    projects = folder_handler.projects()
    root_files = folder_handler.files_in_folder(PDF_DIR)

    if not projects and not root_files:
        ui.label('No projects or PDFs yet').classes('text-gray-500')
        return

    for project in projects:
        project_item(project)

    if root_files:
        with ui.expansion('Unassigned', icon='folder_open').classes('w-full'):
            for file in root_files:
                file_item(file)


def project_item(project: Path):
    with ui.expansion(project.name, icon='folder').classes('w-full'):
        with ui.row().classes('items-center justify-between gap-2 w-full no-wrap'):
            ui.label(project.name).classes('font-medium')
            ui.button(
                'Compare',
                icon='compare_arrows',
                on_click=lambda p=project: open_project_comparison(p),
            ).props('flat dense no-caps size=sm')

        project_files = folder_handler.project_files(project)
        if not project_files:
            ui.label('No PDFs in this project').classes('text-gray-500 pl-8')
            return

        for file in project_files:
            file_item(file)


def file_item(file: Path):
    result_exists = folder_handler.result_path_for_file(file).exists()
    extract_requested = file in state.extract_requested_files

    with ui.card().classes('w-full ml-6 my-1 p-2 gap-1'):
        with ui.row().classes('items-start gap-2 w-full no-wrap'):
            ui.icon('description').classes('text-gray-600 mt-1 text-sm')
            with ui.column().classes('gap-0 grow min-w-0'):
                ui.label(file.name).classes('font-medium text-sm break-all')
                ui.label(file.parent.name if file.parent != PDF_DIR else 'Unassigned').classes('text-xs text-gray-500')
            if result_exists:
                ui.icon('check_circle').classes('text-green-700 mt-1 text-sm')
            elif extract_requested:
                ui.icon('hourglass_empty').classes('text-orange-700 mt-1 text-sm')

        with ui.row().classes('items-center gap-1 w-full no-wrap'):
            ui.button('Open', icon='visibility', on_click=lambda f=file: open_file(f)).props('flat dense no-caps size=sm')
            extract_button = ui.button(
                'Requested' if extract_requested else 'Extract',
                icon='task_alt' if result_exists else 'text_snippet',
            ).props('flat dense no-caps size=sm')
            if result_exists or extract_requested:
                extract_button.props('disable')
            else:
                async def request_extract(_event, f=file, button=extract_button):
                    await extract_file(f, button)

                extract_button.on('click', request_extract)
            rename_button(file)
            move_button(file)
            ui.button('Delete', icon='delete', on_click=lambda f=file: delete_file(f)).props('flat dense no-caps size=sm color=negative')


def rename_button(file: Path):
    with ui.dialog() as dialog, ui.card():
        ui.label(f'Rename {file.name}').classes('font-medium')
        name_input = ui.input('Filename', value=file.stem).classes('w-80')

        def save():
            if rename_file(file, name_input.value):
                dialog.close()

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Save', on_click=save).props('dense no-caps size=sm')

    ui.button('Rename', icon='edit', on_click=dialog.open).props('flat dense no-caps size=sm')


def move_button(file: Path):
    project_options = ['Unassigned'] + [project.name for project in folder_handler.projects()]
    current_project = 'Unassigned' if file.parent == PDF_DIR else file.parent.name

    with ui.dialog() as dialog, ui.card():
        ui.label(f'Move {file.name}').classes('font-medium')
        project_select = ui.select(
            project_options,
            label='Project',
            value=current_project,
        ).classes('w-80')

        def save():
            if move_file(file, project_select.value):
                dialog.close()

        with ui.row().classes('justify-end w-full'):
            ui.button('Cancel', on_click=dialog.close).props('flat dense no-caps size=sm')
            ui.button('Move', on_click=save).props('dense no-caps size=sm')

    ui.button('Move', icon='drive_file_move', on_click=dialog.open).props('flat dense no-caps size=sm')


def refresh_opened_file():
    opened_file_container.clear()
    with opened_file_container:
        show_opened_file()


def opened_file():
    if state.opened_file is None:
        ui.label('No file selected')
        return

    relative_pdf_path = state.opened_file.relative_to(PDF_DIR).as_posix()
    pdf_url = f'/pdfs/{quote(relative_pdf_path, safe="/")}'

    ui.html(f'''
        <iframe
            src="{pdf_url}"
            style="width: 100%; height: 100vh; border: none;"
        ></iframe>
    ''', sanitize=False).classes('w-full h-full')


def load_result(file: Path):
    return folder_handler.load_result(file)


def save_result(file: Path, result: dict):
    folder_handler.save_result(file, result)


def update_summary_value(file: Path, result: dict, field: str, value: str):
    result[field] = value
    save_result(file, result)
    refresh_opened_file()


def update_post_value(file: Path, result: dict, post_index: int, field: str, value: str):
    result['Posten'][post_index][field] = value
    save_result(file, result)
    refresh_opened_file()


def add_summary_field(file: Path, result: dict, field: str | None, value: str | None):
    if not field:
        ui.notify('Enter a field name')
        return

    clean_field = field.strip()
    if not clean_field or clean_field == 'Posten':
        ui.notify('Invalid field name')
        return

    if clean_field in result:
        ui.notify(f'{clean_field} already exists')
        return

    result[clean_field] = value or ''
    save_result(file, result)
    refresh_opened_file()


def add_post_row(file: Path, result: dict):
    result.setdefault('Posten', [])
    result['Posten'].append({
        'Omschrijving': '',
        'Aantal': '',
        'Eenheid': '',
        'Eenheidsprijs': '',
        'Totaalbedrag': '',
    })
    save_result(file, result)
    refresh_opened_file()


def delete_post_row(file: Path, result: dict, post_index: int):
    if 'Posten' not in result or post_index >= len(result['Posten']):
        ui.notify('Row no longer exists')
        refresh_opened_file()
        return

    result['Posten'].pop(post_index)
    save_result(file, result)
    refresh_opened_file()


def update_comparison_value(project: Path, comparison: dict, row_index: int, field: str, value: str):
    comparison['Posten'][row_index][field] = value
    comparison.pop('Matches', None)
    folder_handler.save_comparison(project, comparison)


def add_comparison_row(project: Path, comparison: dict):
    comparison.setdefault('Posten', [])
    comparison['Posten'].append({
        'Omschrijving': '',
        'Aantal': '',
        'Eenheid': '',
    })
    comparison.pop('Matches', None)
    folder_handler.save_comparison(project, comparison)
    schedule_opened_file_refresh()


def delete_comparison_row(project: Path, comparison: dict, row_index: int):
    if 'Posten' not in comparison or row_index >= len(comparison['Posten']):
        ui.notify('Row no longer exists')
        refresh_opened_file()
        return

    comparison['Posten'].pop(row_index)
    comparison.pop('Matches', None)
    folder_handler.save_comparison(project, comparison)
    schedule_opened_file_refresh()


def project_offer_results(project: Path) -> list[dict]:
    offer_results = []
    for file in folder_handler.project_files(project):
        result = folder_handler.load_result(file)
        if result is None:
            continue

        offer_results.append({
            'Bestand': file.name,
            'Posten': result.get('Posten', []),
        })

    return offer_results


def match_comparison_posts(project: Path, comparison: dict) -> dict:
    from main_page.extract_offer import ask_llm, parse_json_response

    prompt = f"""
        Je koppelt begrotings-/vergelijkingsregels aan offerteposten.

        Vergelijkingsregels:
        {json.dumps(comparison.get('Posten', []), ensure_ascii=False, indent=2)}

        Offerteposten per bestand:
        {json.dumps(project_offer_results(project), ensure_ascii=False, indent=2)}

        Maak per vergelijkingsregel en per offertebestand de beste match.
        Als er geen goede match is, vul dan "ONBEKEND" in voor de gematchte velden.
        Reageer ALLEEN met geldige JSON, zonder markdown, in exact dit formaat:
        {{
        "Matches": [
            {{
            "Vergelijking omschrijving": "...",
            "Vergelijking aantal": "...",
            "Vergelijking eenheid": "...",
            "Offerte": "...",
            "Gematchte omschrijving": "...",
            "Gematcht aantal": "...",
            "Gematchte eenheid": "...",
            "Totaalbedrag": "...",
            "Match toelichting": "..."
            }}
        ]
        }}
    """
    return parse_json_response(ask_llm(prompt))


async def match_project_posts(project: Path, comparison: dict, button):
    if not comparison.get('Posten'):
        ui.notify('Add comparison rows before matching')
        return

    if not project_offer_results(project):
        ui.notify('No extracted offer results available for this project')
        return

    button.set_text('Matching')
    button.props('loading disable')
    button.update()

    try:
        match_result = await run.io_bound(match_comparison_posts, project, comparison)
    except Exception as error:
        notify_safe(f'Could not match posts: {error}')
        return

    comparison['Matches'] = match_result.get('Matches', [])
    folder_handler.save_comparison(project, comparison)
    notify_safe('Matched posts')
    schedule_opened_file_refresh_safe()


def opened_file_result():
    if state.opened_file is None:
        return

    result = load_result(state.opened_file)
    if result is None:
        ui.label('No result found for this file').classes('text-gray-500')
        return

    from main_page.extract_offer import validate_offer_json

    ui.label('Result').classes('text-lg font-bold')
    validation_warnings = validate_offer_json(result)
    if validation_warnings:
        with ui.column().classes('w-full gap-1 bg-yellow-50 border border-yellow-300 p-2 rounded'):
            ui.label('Checks').classes('font-medium text-yellow-900')
            for warning in validation_warnings:
                ui.label(warning).classes('text-xs text-yellow-900')

    opened_file = state.opened_file
    render_editable_table(
        result,
        row_collection_key='Posten',
        row_fields=['Omschrijving', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag'],
        on_summary_update=lambda field, value: update_summary_value(opened_file, result, field, value),
        on_summary_add=lambda field, value: add_summary_field(opened_file, result, field, value),
        on_row_update=lambda index, field, value: update_post_value(opened_file, result, index, field, value),
        on_row_add=lambda: add_post_row(opened_file, result),
        on_row_delete=lambda index: delete_post_row(opened_file, result, index),
    )


def comparison_page():
    project = state.comparison_project
    if project is None:
        ui.label('No project selected').classes('text-gray-500')
        return

    ui.label(f'Comparison: {project.name}').classes('text-xl font-bold')
    # ui.label('Comparison page placeholder').classes('text-gray-500')

    result_files = [
        file
        for file in folder_handler.project_files(project)
        if folder_handler.result_path_for_file(file).exists()
    ]

    ui.label(f'{len(result_files)} extracted file(s) available for comparison').classes('text-sm text-gray-600')

    comparison = folder_handler.load_comparison(project)
    comparison_rows = [
        {'id': index, **row}
        for index, row in enumerate(comparison.get('Posten', []))
    ]

    ui.label('Comparison rows').classes('text-lg font-bold mt-4')
    render_editable_rows(
        comparison_rows,
        ['Omschrijving', 'Aantal', 'Eenheid'],
        on_update=lambda index, field, value: update_comparison_value(project, comparison, index, field, value),
        on_add=lambda: add_comparison_row(project, comparison),
        on_delete=lambda index: delete_comparison_row(project, comparison, index),
    )

    with ui.row().classes('items-center gap-2 mt-4'):
        match_button = ui.button('Match posten', icon='auto_fix_high').props('dense no-caps')

        async def request_match(_event, p=project, data=comparison, button=match_button):
            await match_project_posts(p, data, button)

        match_button.on('click', request_match)

    match_rows = [
        {'id': index, **row}
        for index, row in enumerate(comparison.get('Matches', []))
    ]
    if not match_rows:
        return

    ui.label('Matched posten').classes('text-lg font-bold mt-4')
    ui.table(
        columns=[
            {'name': 'Vergelijking omschrijving', 'label': 'Vergelijking', 'field': 'Vergelijking omschrijving'},
            {'name': 'Vergelijking aantal', 'label': 'Aantal', 'field': 'Vergelijking aantal'},
            {'name': 'Vergelijking eenheid', 'label': 'Eenheid', 'field': 'Vergelijking eenheid'},
            {'name': 'Offerte', 'label': 'Offerte', 'field': 'Offerte'},
            {'name': 'Gematchte omschrijving', 'label': 'Gematchte post', 'field': 'Gematchte omschrijving'},
            {'name': 'Gematcht aantal', 'label': 'Gem. aantal', 'field': 'Gematcht aantal'},
            {'name': 'Gematchte eenheid', 'label': 'Gem. eenheid', 'field': 'Gematchte eenheid'},
            {'name': 'Totaalbedrag', 'label': 'Totaal', 'field': 'Totaalbedrag'},
            {'name': 'Match toelichting', 'label': 'Toelichting', 'field': 'Match toelichting'},
        ],
        rows=match_rows,
        row_key='id',
    ).classes('w-full')


def show_opened_file():
    if state.current_view == 'comparison':
        with ui.column().classes('w-full h-full p-4'):
            comparison_page()
        return

    with ui.row().classes('w-full h-full flex-nowrap'):
        with ui.column().classes('w-1/2 h-full'):
            opened_file()
        with ui.column().classes('w-1/2 h-full p-4 overflow-auto'):
            opened_file_result()


def right_side():
    global opened_file_container

    opened_file_container = ui.column().classes('w-full h-full')
    with opened_file_container:
        show_opened_file()


@ui.page('/')
def main_page():
    eva_html()
    ui.colors(primary=PRIMARY_RED)

    with ui.left_drawer().style(f'background-color: {SECONDARY_RED}'):
        left_drawer()

    right_side()
