from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from domain.offer import Offer
from utils.app_logging import log_action

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_EXTRACTIONS = int(os.environ.get('EXTRACT_MAX_CONCURRENT', '3'))


class ExtractionJobService:
    """Runs offer extraction jobs outside the UI event lifecycle.

    Concurrent jobs are capped by a semaphore (default
    ``DEFAULT_MAX_CONCURRENT_EXTRACTIONS``, override via
    ``EXTRACT_MAX_CONCURRENT``) so that bulk-extracting many offers at once
    doesn't fire off unbounded simultaneous Gemini calls and worker threads.
    A job still shows as "running" in the UI the moment it's requested, even
    while it's queued waiting for a slot.
    """

    def __init__(self, folder_handler, *, max_concurrent: int = DEFAULT_MAX_CONCURRENT_EXTRACTIONS) -> None:
        self.folder_handler = folder_handler
        self._tasks: dict[Path, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

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

    def cancel(self, offer: Offer) -> bool:
        """Best-effort cancel: stops the UI from waiting on this job.

        The extraction itself runs in a worker thread (``asyncio.to_thread``),
        which Python cannot forcibly interrupt, so any in-flight LLM call or
        file write finishes regardless. Once it does, its (now orphaned)
        status/extract.json writes may still land after the 'cancelled'
        status below — status.json writes are atomic, so this can only cause
        the UI to flip back to 'done'/'failed' shortly after a cancel, never
        a corrupted file.
        """
        task = self._tasks.get(offer.path)
        if task is None or task.done():
            return False

        task.cancel()
        return True

    async def _run(self, offer: Offer) -> None:
        from services.extract_offer import extract_offer

        async with self._semaphore:
            await asyncio.to_thread(extract_offer, offer.document, self.folder_handler)

    def _finish(self, offer_path: Path, task: asyncio.Task) -> None:
        self._tasks.pop(offer_path, None)
        offer_label = f'{offer_path.parent.name}/{offer_path.name}'

        if task.cancelled():
            logger.info('Extraction job cancelled for %s', offer_path)
            log_action('extraction_cancelled', offer=offer_label)
            self._mark_cancelled(offer_path)
            return

        try:
            task.result()
        except Exception:
            logger.exception('Extraction job failed for %s', offer_path)
            log_action('extraction_failed', offer=offer_label)
            return

        log_action('extraction_finished', offer=offer_label)

    def _mark_cancelled(self, offer_path: Path) -> None:
        from services.extract_offer import update_extraction_status

        try:
            update_extraction_status(
                self.folder_handler,
                offer_path / 'document.pdf',
                status='cancelled',
                step='cancelled',
                message='Extraction cancelled by user',
            )
        except Exception:
            logger.exception('Failed to record cancellation status for %s', offer_path)
