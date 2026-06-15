from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from domain.offer import Offer

logger = logging.getLogger(__name__)


class ExtractionJobService:
    """Runs offer extraction jobs outside the UI event lifecycle."""

    def __init__(self, folder_handler) -> None:
        self.folder_handler = folder_handler
        self._tasks: dict[Path, asyncio.Task] = {}

    def is_running(self, offer: Offer) -> bool:
        task = self._tasks.get(offer.path)
        return task is not None and not task.done()

    def start(self, offer: Offer) -> bool:
        if self.is_running(offer):
            return False

        task = asyncio.create_task(self._run(offer))
        self._tasks[offer.path] = task
        task.add_done_callback(lambda completed_task, path=offer.path: self._finish(path, completed_task))
        return True

    async def _run(self, offer: Offer) -> None:
        from services.extract_offer import extract_offer

        await asyncio.to_thread(extract_offer, offer.document, self.folder_handler)

    def _finish(self, offer_path: Path, task: asyncio.Task) -> None:
        self._tasks.pop(offer_path, None)

        if task.cancelled():
            logger.info('Extraction job cancelled for %s', offer_path)
            return

        try:
            task.result()
        except Exception:
            logger.exception('Extraction job failed for %s', offer_path)
