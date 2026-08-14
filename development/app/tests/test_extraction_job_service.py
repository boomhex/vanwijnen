import asyncio
import threading
import time

import services.extract_offer as extract_offer_module
from application.extraction_job_service import ExtractionJobService
from domain.offer import Offer


def test_extraction_job_service_limits_concurrency(monkeypatch, tmp_path):
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_extract_offer(document_path, folder_handler):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    monkeypatch.setattr(extract_offer_module, 'extract_offer', fake_extract_offer)

    service = ExtractionJobService(folder_handler=object(), max_concurrent=2)
    offers = [Offer(tmp_path / f'offer{index}') for index in range(5)]

    async def run_all():
        for offer in offers:
            assert service.start(offer)
        await asyncio.gather(*list(service._tasks.values()), return_exceptions=True)

    asyncio.run(run_all())

    assert max_active <= 2


def test_extraction_job_service_default_limit_is_used_when_not_specified():
    service = ExtractionJobService(folder_handler=object())
    assert service._semaphore._value >= 1
