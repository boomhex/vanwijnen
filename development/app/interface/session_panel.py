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
                .props('flat dense round').tooltip('Switch workspace')
            ui.button(icon='logout', on_click=logout).props('flat dense round').tooltip('Log out')

        with ui.expansion('Report a problem', icon='bug_report').classes('w-full'):
            message = ui.textarea(
                placeholder='What happened? What did you click just before?'
            ).classes('w-full').props('outlined dense')

            def send() -> None:
                text = (message.value or '').strip()
                if not text:
                    ui.notify('Describe the problem first', color='warning')
                    return
                log_feedback(text)
                message.set_value('')
                ui.notify('Thanks, your report was logged', color='positive')

            ui.button('Send report', icon='send', on_click=send).props('no-caps dense')


def logout() -> None:
    log_action('logout')
    app.storage.user.clear()
    ui.navigate.to('/login')
