from nicegui import ui


def eva_html():
    ui.add_head_html(
        '<link ' + \
            'href="https://unpkg.com/eva-icons@1.1.3/style/eva-icons.css" ' +\
            'rel="stylesheet"' + \
        '/>'
    )