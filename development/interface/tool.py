from pathlib import Path
import pdfplumber
from nicegui import ui
import os

RED = '#B00000'
OVERVIEW_MARKDOWN = Path(__file__).with_name('overzicht.md')

files = {
    "files" : ""
}

async def handle_upload(e):
    await e.file.save(f"./files/{e.file.name}")
    reload()

def reload():
    get_files()

def analyse_tab():
    ui.label('Analyse').classes('text-2xl font-bold text-red-900')
    ui.label('Analyse project data and compare findings here.').classes(
        'text-red-800'
    )
    ui.upload(on_upload=handle_upload)

def get_files():
    files.update(files='\n'.join(os.listdir("./files/")))

def printhello():
    print('hello')

def file_grid():
    with ui.list().props('bordered separator'):
        with ui.item():
            with ui.item_section():
                ui.item_label("File 1")
            with ui.item_section().props('side'):
                ui.icon('home')


@ui.page('/')
def main_page():
    ui.colors(primary=RED)

    reload()

    with ui.header().classes('w-full bg-white text-red-900 shadow-sm px-6 py-3'):
        with ui.row().classes('w-full items-center'):
            ui.label('Project tool').classes('text-xl font-bold')

    with ui.row():
        with ui.column():
            analyse_tab()
        with ui.column():
            file_grid()


ui.run()
