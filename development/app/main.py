import logging
import sys

from nicegui import ui
from interface.main_page import main_page


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    stream=sys.stdout,
)

ui.run()
