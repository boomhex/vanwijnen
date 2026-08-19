from nicegui import app, ui

from utils.app_logging import log_action, log_feedback


def render_session_panel() -> None:
    """Logged-in user, workspace switcher, logout, and a problem-report box."""
    with ui.column().classes('w-full mt-4 gap-1'):
        with ui.row().classes('w-full items-center gap-2'):
            ui.icon('person')
            username = app.storage.user.get('username', '?')
            workspace = app.storage.user.get('workspace', '?')
            ui.label(f'{username} · {workspace}').classes('grow')
            ui.button(icon='swap_horiz', on_click=lambda: ui.navigate.to('/workspaces')) \
                .props('flat dense round').tooltip('Wissel werkruimte')
            ui.button(icon='logout', on_click=logout).props('flat dense round').tooltip('Uitloggen')

        with ui.expansion('Probleem melden', icon='bug_report').classes('w-full'):
            message = ui.textarea(
                placeholder='Wat gebeurde er? Waar had u net op geklikt?'
            ).classes('w-full').props('outlined dense')

            def send() -> None:
                text = (message.value or '').strip()
                if not text:
                    ui.notify('Beschrijf eerst het probleem', color='warning')
                    return
                log_feedback(text)
                message.set_value('')
                ui.notify('Bedankt, uw melding is geregistreerd', color='positive')

            ui.button('Melding versturen', icon='send', on_click=send).props('no-caps dense')


def logout() -> None:
    log_action('logout')
    app.storage.user.clear()
    ui.navigate.to('/login')
