from pathlib import Path

from nicegui import ui

RED = '#B00000'
OVERVIEW_MARKDOWN = Path(__file__).with_name('overzicht.md')

def analyse_tab():
    ui.label('Analyse').classes('text-2xl font-bold text-red-900')
    ui.label('Analyse project data and compare findings here.').classes(
        'text-red-800'
    )

def overview_tab():
    ui.markdown(OVERVIEW_MARKDOWN.read_text(encoding='utf-8')).classes(
        'max-w-4xl text-red-900'
    )

def report_tab():
    ui.label('Rapporteer').classes('text-2xl font-bold text-red-900')
    ui.label('Build or review the final report here.').classes(
        'text-red-800'
    )

@ui.page('/')
def main_page():
    ui.colors(primary=RED)

    with ui.header().classes('w-full bg-white text-red-900 shadow-sm px-6 py-3'):
        with ui.row().classes('w-full items-center'):
            ui.label('Project tool').classes('text-xl font-bold')

            with ui.tabs().classes('text-red-900') as tabs:
                overzicht = ui.tab('Overzicht')
                analyse = ui.tab('Analyse')
                rapport = ui.tab('Rapporteer')

    with ui.column().classes('w-full min-h-screen bg-red-50'):
        with ui.tab_panels(tabs, value=overzicht).classes(
            'w-full flex-grow bg-red-50'
        ):
            with ui.tab_panel(overzicht).classes('p-6'):
                overview_tab()

            with ui.tab_panel(analyse).classes('p-6'):
                analyse_tab()

            with ui.tab_panel(rapport).classes('p-6'):
                report_tab()


ui.run()
