from nicegui import ui


def compact_name(name: str, max_length: int = 32, *, mode: str = 'middle') -> str:
    if len(name) <= max_length:
        return name

    if max_length <= 3:
        return name[:max_length]

    if mode == 'end':
        return f'{name[:max_length - 3]}...'

    keep = max_length - 3
    left = keep // 2
    right = keep - left
    return f'{name[:left]}...{name[-right:]}'


def add_name_tooltip(displayed_name: str, full_name: str) -> None:
    if displayed_name != full_name:
        ui.tooltip(full_name)


def element_is_live(element) -> bool:
    if element is None:
        return False
    if getattr(element, 'is_deleted', False):
        return False

    client = getattr(element, 'client', None)
    if client is None:
        return False

    return not getattr(client, '_deleted', False)
