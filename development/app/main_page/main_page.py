from nicegui import ui, events, app, run
from utils.eva_html import eva_html
from pathlib import Path
from urllib.parse import quote
import json

PRIMARY_RED = '#B00000'
SECONDARY_RED = "#F9BFBF"
APP_DIR = Path(__file__).resolve().parents[1]
TMP_DIR = APP_DIR / 'tmp'
PDF_DIR = TMP_DIR / 'pdfs'
RESULTS_DIR = TMP_DIR / 'results'

PDF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app.add_static_files('/pdfs', PDF_DIR)


class State:
    def __init__(self, pdf_dir: Path) -> None:
        self.pdf_dir = pdf_dir
        self.opened_file: Path | None = None
        self.upload_project: str | None = None
        self.extract_requested_files: set[Path] = set()

    def files_in_folder(self, folder: Path) -> list[Path]:
        return sorted(folder.glob('*.pdf'), key=lambda file: file.name.lower())

    def projects(self) -> list[Path]:
        return sorted(
            [path for path in self.pdf_dir.iterdir() if path.is_dir()],
            key=lambda path: path.name.lower(),
        )

    def project_files(self, project: Path) -> list[Path]:
        return self.files_in_folder(project)

    def upload_folder(self) -> Path:
        if not self.upload_project:
            return self.pdf_dir

        project_folder = self.pdf_dir / self.upload_project
        project_folder.mkdir(parents=True, exist_ok=True)
        return project_folder

    async def add_file(self, event: events.UploadEventArguments):
        await event.file.save(self.upload_folder() / event.file.name)


state = State(PDF_DIR)
opened_file_container = None
file_list_container = None


async def handle_upload(e: events.UploadEventArguments):
    await state.add_file(e)
    refresh_file_list()


def left_drawer():
    global file_list_container

    ui.upload(on_upload=handle_upload, auto_upload=True).classes('w-full')
    file_list_container = ui.column().classes('w-full')
    with file_list_container:
        file_list()


def open_file(file: Path):
    state.opened_file = file
    refresh_opened_file()
    ui.notify(f'Opened {file.name}')


def delete_file(file: Path):
    if not file.exists():
        ui.notify(f'{file.name} does not exist')
        refresh_file_list()
        return

    file.unlink()

    if state.opened_file == file:
        state.opened_file = None
        refresh_opened_file()

    refresh_file_list()
    ui.notify(f'Deleted {file.name}')


def rename_file(file: Path, new_name: str | None):
    if not file.exists():
        ui.notify(f'{file.name} does not exist')
        refresh_file_list()
        return False

    if not new_name:
        ui.notify('Enter a new filename')
        return False

    clean_name = new_name.strip()
    if not clean_name or '/' in clean_name or '\\' in clean_name or clean_name in {'.', '..'}:
        ui.notify('Invalid filename')
        return False

    if Path(clean_name).suffix.lower() != '.pdf':
        clean_name = f'{clean_name}.pdf'

    new_file = file.with_name(clean_name)
    if new_file == file:
        ui.notify('Filename is unchanged')
        return False

    if new_file.exists():
        ui.notify(f'{clean_name} already exists')
        return False

    old_result_path = result_path_for_file(file)
    new_result_path = result_path_for_file(new_file)

    try:
        file.rename(new_file)
    except OSError as error:
        ui.notify(f'Could not rename file: {error}')
        return False

    if old_result_path.exists() and not new_result_path.exists():
        new_result_path.parent.mkdir(parents=True, exist_ok=True)
        old_result_path.rename(new_result_path)

    if state.opened_file == file:
        state.opened_file = new_file
        refresh_opened_file()

    refresh_file_list()
    ui.notify(f'Renamed to {new_file.name}')
    return True


def refresh_file_list():
    global file_list_container
    file_list_container.clear()

    with file_list_container:
        file_list()


def create_project(project_name: str | None):
    if not project_name:
        ui.notify('Enter a project name')
        return

    clean_name = project_name.strip()
    if not clean_name or '/' in clean_name or '\\' in clean_name or clean_name in {'.', '..'}:
        ui.notify('Invalid project name')
        return

    (PDF_DIR / clean_name).mkdir(exist_ok=True)
    state.upload_project = clean_name
    refresh_file_list()


def process_selected_projects(selected_projects):
    checked_projects = [
        item['project']
        for item in selected_projects
        if item['checkbox'].value
    ]

    from main_page.extract_offer import extract_offer

    for project in checked_projects:
        project_results_dir = RESULTS_DIR / project.name
        for file in state.project_files(project):
            extract_offer(file, project_results_dir)

    refresh_file_list()


def result_dir_for_file(file: Path) -> Path:
    relative_parent = file.relative_to(PDF_DIR).parent
    if relative_parent == Path('.'):
        return RESULTS_DIR

    return RESULTS_DIR / relative_parent


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
        refresh_file_list()
    

def file_list():
    selected_projects = []
    project_names = [project.name for project in state.projects()]

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

    ui.button(
        'Compare Selected Projects',
        on_click=lambda: process_selected_projects(selected_projects)
    )
    ui.label('Projects').classes('text-lg font-bold mb-2')
    projects = state.projects()
    root_files = state.files_in_folder(PDF_DIR)

    if not projects and not root_files:
        ui.label('No projects or PDFs yet').classes('text-gray-500')
        return

    for project in projects:
        project_item(project, selected_projects)

    if root_files:
        with ui.expansion('Unassigned', icon='folder_open').classes('w-full'):
            for file in root_files:
                file_item(file)


def project_item(project: Path, selected_projects: list[dict]):
    with ui.expansion(project.name, icon='folder').classes('w-full'):
        with ui.row().classes('items-center gap-2 w-full no-wrap'):
            checkbox = ui.checkbox()
            selected_projects.append({
                'project': project,
                'checkbox': checkbox,
            })
            ui.label(project.name).classes('font-medium')

        project_files = state.project_files(project)
        if not project_files:
            ui.label('No PDFs in this project').classes('text-gray-500 pl-8')
            return

        for file in project_files:
            file_item(file)


def file_item(file: Path):
    result_exists = result_path_for_file(file).exists()
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


def result_path_for_file(file: Path) -> Path:
    nested_result_path = result_dir_for_file(file) / f'{file.stem}.txt'
    flat_result_path = RESULTS_DIR / f'{file.stem}.txt'

    if nested_result_path.exists():
        return nested_result_path

    return flat_result_path


def load_result(file: Path):
    result_path = result_path_for_file(file)
    if not result_path.exists():
        return None

    with result_path.open('r') as result_file:
        result_text = result_file.read()

    from main_page.extract_offer import parse_json_response

    return parse_json_response(result_text)


def save_result(file: Path, result: dict):
    result_path = result_path_for_file(file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open('w') as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=4)


def update_summary_value(file: Path, result: dict, field: str, value: str):
    result[field] = value
    save_result(file, result)
    refresh_opened_file()


def update_post_value(file: Path, result: dict, post_index: int, field: str, value: str):
    result['Posten'][post_index][field] = value
    save_result(file, result)
    refresh_opened_file()


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

    editable_summary_table(state.opened_file, result)

    posten = [
        {'id': index, **row}
        for index, row in enumerate(result.get('Posten', []))
    ]
    if not posten:
        return

    ui.label('Posten').classes('text-lg font-bold mt-4')
    editable_posten_table(state.opened_file, result, posten)


def editable_summary_table(file: Path, result: dict):
    with ui.column().classes('w-full gap-1'):
        for field, value in result.items():
            if field == 'Posten':
                continue

            with ui.row().classes('items-center w-full gap-2 no-wrap'):
                ui.label(field).classes('w-40 text-xs font-medium')
                input_field = ui.input(value=str(value)).props('dense outlined').classes('grow')
                input_field.on(
                    'blur',
                    lambda _event, key=field, field_input=input_field: update_summary_value(
                        file,
                        result,
                        key,
                        field_input.value,
                    ),
                )
                input_field.on(
                    'keydown.enter',
                    lambda _event, key=field, field_input=input_field: update_summary_value(
                        file,
                        result,
                        key,
                        field_input.value,
                    ),
                )


def editable_posten_table(file: Path, result: dict, posten: list[dict]):
    fields = ['Omschrijving', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag']

    with ui.column().classes('w-full gap-2'):
        with ui.row().classes('w-full gap-1 no-wrap text-xs font-medium text-gray-600'):
            for field in fields:
                ui.label(field).classes('grow basis-0')

        for post in posten:
            post_index = post['id']
            with ui.row().classes('w-full gap-1 no-wrap'):
                for field in fields:
                    input_field = ui.input(value=str(post.get(field, ''))).props('dense outlined').classes('grow basis-0')
                    input_field.on(
                        'blur',
                        lambda _event, index=post_index, key=field, field_input=input_field: update_post_value(
                            file,
                            result,
                            index,
                            key,
                            field_input.value,
                        ),
                    )
                    input_field.on(
                        'keydown.enter',
                        lambda _event, index=post_index, key=field, field_input=input_field: update_post_value(
                            file,
                            result,
                            index,
                            key,
                            field_input.value,
                        ),
                    )


def show_opened_file():
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
