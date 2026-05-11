from nicegui import ui

RED = '#B00000'
DARK_RED = '#7A0000'
LIGHT_RED = '#FDECEC'

projects = ['EEG-fMRI', 'Van Wijnen', 'Solar Racing', 'New Proposal']
active_project = {'name': projects[0]}


@ui.page('/')
def page_layout():
    ui.colors(primary=RED)

    with ui.left_drawer(top_corner=True, bottom_corner=True).classes(
        'bg-red-900 text-white p-4'
    ).style('width: 280px;'):

        ui.label('Projects').classes('text-2xl font-bold mb-4')

        def select_project(project_name: str):
            active_project['name'] = project_name
            project_title.set_text(project_name)
            chat_area.clear()
            with chat_area:
                ui.chat_message(
                    f'Project switched to {project_name}.',
                    name='System',
                    sent=False,
                )

        for project in projects:
            ui.button(
                project,
                on_click=lambda p=project: select_project(p),
            ).props('flat align=left').classes(
                'w-full justify-start text-white hover:bg-red-700 mb-1'
            )

        ui.separator().classes('my-4 bg-red-300')
        ui.button(
            'New project',
            icon='add',
            on_click=lambda: ui.notify('Create project clicked'),
        ).style(
            'background-color: #B00000; color: white;'
        ).classes(
            'w-full font-bold'
        )

    with ui.column().classes('w-full h-screen bg-red-50'):

        with ui.row().classes(
            'w-full items-center justify-between p-4 bg-white shadow'
        ):
            project_title = ui.label(active_project['name']).classes(
                'text-2xl font-bold text-red-900'
            )
            ui.button('Settings', icon='settings').props('flat color=red')

        chat_area = ui.column().classes(
            'w-full max-w-4xl mx-auto flex-grow p-6 gap-4'
        )

        with chat_area:
            ui.chat_message(
                'Hello. Select a project on the left or start chatting here.',
                name='Assistant',
                sent=False,
            )
            ui.chat_message(
                'Can you summarize the latest project status?',
                name='You',
                sent=True,
            )

        with ui.row().classes(
            'w-full max-w-4xl mx-auto p-4 bg-white rounded-t-2xl shadow-lg items-center'
        ):
            message_input = ui.input(
                placeholder='Message this project...'
            ).props('outlined').classes('flex-grow')

            def send_message():
                text = message_input.value
                if not text:
                    return

                with chat_area:
                    ui.chat_message(text, name='You', sent=True)
                    ui.chat_message(
                        f'Received in project: {active_project["name"]}',
                        name='Assistant',
                        sent=False,
                    )

                message_input.value = ''

            message_input.on('keydown.enter', send_message)

            ui.button(
                icon='send',
                on_click=send_message,
            ).classes('bg-red-700 text-white')


ui.run()