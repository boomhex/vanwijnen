from pathlib import Path
import pdfplumber
from nicegui import ui
import os
from extract_offer import main as extraction
import json

RED = '#B00000'


files = {
    "files" : []
}

async def handle_upload(e):
    await e.file.save(f"./files/{e.file.name}")
    files['files'].append(e.file.name)
    ui.notify(f'Upload file: {e.file.name}')
    reload()

def reload():
    update_file_folder(Path("./files"))
    file_grid.refresh()

def analyse_tab():
    ui.label('Upload').classes('text-2xl font-bold text-red-900')
    ui.upload(
        auto_upload=True,
        on_upload=handle_upload,
        label='Drop files here or click to upload',
    ).props(
        'flat'
    ).classes(
        'upload-dropzone w-96 h-40 border border-dashed border-red-900 '
        'bg-red-50 text-red-900 rounded-lg'
    )


def update_file_folder(folder: Path):
    for file in os.listdir(folder):
        if file not in files['files']:
            os.remove(folder / file)


def delete_file(file_name):
    files['files'].remove(file_name)
    ui.notify(f'Delete file: {file_name}')
    reload()


def process_file(file_name):
    ui.notify(f'Process file: {file_name}')


@ui.refreshable
def file_grid():
    ui.label("Files:").classes('text-2xl font-bold text-red-900')
    with ui.list().props('bordered separator'):
        for file_name in files["files"]:
            with ui.item():
                with ui.item_section():
                    ui.item_label(file_name)
                with ui.item_section().props('side'):
                    ui.button(
                        icon='eva-close-outline',
                        on_click=lambda file_name=file_name: delete_file(file_name),
                    )
                # with ui.item_section().props('side'):
                #     ui.button(
                #         icon='eva-checkmark-outline',
                #         on_click=lambda file_name=file_name: process_file(file_name),
                #     )

def extract_file_information(files):
    for file in files:
        input_path = Path(
            './files'
        ) / file
        result_path = Path(
            'res'
        )
        extraction(input_path, result_path)

def display_json(json):
    pass

@ui.page('/analyse')
def analyse_files_page():
    extract_file_information(files['files'])

    for file in os.listdir(Path('./res')):
        with open(Path('./res') / file, 'r') as f:
            text = f.read()
        ui.label(text)
    
    

@ui.page('/')
def main_page():
    ui.add_head_html(
        '<link ' + \
            'href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" ' +\
            'rel="stylesheet"' + \
        '/>'
    )
    ui.colors(primary=RED)
    ui.add_css('''
        .upload-dropzone .q-uploader__header {
            background: transparent;
            color: #7f1d1d;
            box-shadow: none;
        }

        .upload-dropzone .q-uploader__list {
            background: transparent;
        }
    ''')

    reload()

    with ui.header().classes('w-full bg-white text-red-900 shadow-sm px-6 py-3'):
        with ui.row().classes('w-full items-center'):
            ui.label('Project tool').classes('text-xl font-bold')

    with ui.row():
        with ui.column():
            analyse_tab()
        with ui.column():
            file_grid()
        with ui.column():
            ui.button('Vergelijk', on_click=lambda : ui.navigate.to(analyse_files_page))

ui.run()
