from dataclasses import dataclass

from application.extraction_job_service import ExtractionJobService
from application.offer_service import OfferService
from application.project_service import ProjectService
from domain.status import is_active_running, is_stale_running
from interface.page_state import MainPageState, View
from services.folder_handler import FolderHandler, UNASSIGNED_PROJECT
from services.offer import Offer
from services.project import Project
from utils.app_logging import logged_action


@dataclass(frozen=True)
class DrawerActionResult:
    success: bool
    message: str | None = None
    refresh_drawer: bool = False
    refresh_right_side: bool = False


@dataclass(frozen=True)
class OfferItemStatus:
    result_exists: bool
    extraction_status: dict | None
    status_step: str | None
    status_message: str | None
    status_error: str | None
    status_stale: bool
    extract_requested: bool
    extraction_failed: bool


class DrawerActions:
    def __init__(
        self,
        *,
        state: MainPageState,
        folder_handler: FolderHandler,
        offer_service: OfferService,
        project_service: ProjectService,
        extraction_job_service: ExtractionJobService,
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.offer_service = offer_service
        self.project_service = project_service
        self.extraction_job_service = extraction_job_service

    @logged_action
    def create_project(self, project_name: str | None) -> DrawerActionResult:
        try:
            project = self.project_service.create(project_name)
        except ValueError as error:
            return self.failure(error)

        self.state.upload_project = project.name
        self.state.selected_project = project
        self.state.selected_offer = None
        return DrawerActionResult(True, refresh_drawer=True)

    def load_extraction_status(self, offer: Offer) -> dict | None:
        try:
            return self.folder_handler.load_extraction_status(offer.document)
        except (FileNotFoundError, ValueError, OSError):
            return None

    def offer_item_status(self, offer: Offer) -> OfferItemStatus:
        result_exists = offer.extract_path.exists()
        extraction_status = self.load_extraction_status(offer)
        status = extraction_status.get('status') if extraction_status else None
        status_step = extraction_status.get('step') if extraction_status else None
        status_message = extraction_status.get('message') if extraction_status else None
        status_error = extraction_status.get('error') if extraction_status else None
        job_running = self.extraction_job_service.is_running(offer)
        status_running = is_active_running(extraction_status)

        if not job_running and not status_running:
            self.state.extract_requested_offers.discard(offer)

        return OfferItemStatus(
            result_exists=result_exists,
            extraction_status=extraction_status,
            status_step=status_step,
            status_message=status_message,
            status_error=status_error,
            status_stale=is_stale_running(extraction_status),
            extract_requested=job_running or status_running,
            extraction_failed=status == 'failed',
        )

    @logged_action
    def open_offer(self, offer: Offer) -> DrawerActionResult:
        self.select_offer(offer)
        self.state.opened_offer = offer
        self.state.current_view = View.OFFER
        return DrawerActionResult(True, f'{offer.name} geopend', refresh_drawer=True, refresh_right_side=True)

    @logged_action
    def open_project_comparison(self, project: Project) -> DrawerActionResult:
        self.select_project(project)
        self.state.comparison_project = project
        self.state.current_view = View.COMPARISON
        return DrawerActionResult(True, refresh_drawer=True, refresh_right_side=True)

    @logged_action
    def select_offer(self, offer: Offer) -> DrawerActionResult:
        self.state.selected_offer = offer
        self.state.selected_project = None
        return DrawerActionResult(True, refresh_drawer=True)

    @logged_action
    def select_project(self, project: Project) -> DrawerActionResult:
        self.state.selected_project = project
        self.state.selected_offer = None
        return DrawerActionResult(True, refresh_drawer=True)

    @logged_action
    def delete_offer(self, offer: Offer) -> DrawerActionResult:
        try:
            self.offer_service.delete(offer)
        except FileNotFoundError as error:
            return DrawerActionResult(False, str(error), refresh_drawer=True)

        refresh_right_side = False
        if self.state.opened_offer == offer:
            self.state.opened_offer = None
            refresh_right_side = True

        self.state.extract_requested_offers.discard(offer)
        if self.state.selected_offer == offer:
            self.state.selected_offer = None
        return DrawerActionResult(
            True,
            f'{offer.name} verwijderd',
            refresh_drawer=True,
            refresh_right_side=refresh_right_side,
        )

    @logged_action
    def clear_extraction(self, offer: Offer) -> DrawerActionResult:
        if self.extraction_job_service.is_running(offer):
            return DrawerActionResult(False, f'Extractie loopt nog voor {offer.name}')

        try:
            removed = self.offer_service.clear_extraction(offer)
        except FileNotFoundError as error:
            return DrawerActionResult(False, str(error), refresh_drawer=True)

        return DrawerActionResult(
            True,
            f'Extractie gereset voor {offer.name}: {removed} bestanden verwijderd',
            refresh_drawer=True,
            refresh_right_side=self.state.opened_offer == offer,
        )

    @logged_action
    def rename_offer(self, offer: Offer, new_name: str | None) -> DrawerActionResult:
        try:
            new_offer = self.offer_service.rename(offer, new_name)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            return self.failure(error)

        refresh_right_side = False
        if self.state.opened_offer == offer:
            self.state.opened_offer = new_offer
            refresh_right_side = True

        if self.state.selected_offer == offer:
            self.state.selected_offer = new_offer

        if offer in self.state.extract_requested_offers:
            self.state.extract_requested_offers.discard(offer)
            self.state.extract_requested_offers.add(new_offer)

        return DrawerActionResult(
            True,
            f'Hernoemd naar {new_offer.name}',
            refresh_drawer=True,
            refresh_right_side=refresh_right_side,
        )

    @logged_action
    def rename_project(self, project: Project, new_name: str | None) -> DrawerActionResult:
        old_name = project.name

        try:
            new_project = self.project_service.rename(project, new_name)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            return self.failure(error)

        if self.state.upload_project == old_name:
            self.state.upload_project = new_project.name

        if old_name in self.state.expanded_project_names:
            self.state.expanded_project_names.discard(old_name)
            self.state.expanded_project_names.add(new_project.name)

        if self.state.selected_project == project:
            self.state.selected_project = new_project

        refresh_right_side = False
        if self.state.comparison_project == project:
            self.state.comparison_project = new_project
            refresh_right_side = True

        if self.state.opened_offer is not None and self.state.opened_offer.project_name == old_name:
            new_offer_path = new_project.path / self.state.opened_offer.name
            self.state.opened_offer = self.folder_handler.offer_from_path(new_offer_path)
            refresh_right_side = True

        if self.state.selected_offer is not None and self.state.selected_offer.project_name == old_name:
            new_offer_path = new_project.path / self.state.selected_offer.name
            self.state.selected_offer = self.folder_handler.offer_from_path(new_offer_path)

        self.state.extract_requested_offers = {
            self.folder_handler.offer_from_path(new_project.path / offer.name)
            if offer.project_name == old_name
            else offer
            for offer in self.state.extract_requested_offers
        }

        return DrawerActionResult(
            True,
            f'{old_name} hernoemd naar {new_project.name}',
            refresh_drawer=True,
            refresh_right_side=refresh_right_side,
        )

    @logged_action
    def delete_project(self, project: Project) -> DrawerActionResult:
        project_name = project.name

        try:
            self.project_service.delete(project)
        except (FileNotFoundError, OSError, ValueError) as error:
            return DrawerActionResult(False, str(error), refresh_drawer=True)

        if self.state.upload_project == project_name:
            self.state.upload_project = self.project_service.first_project_name_or_unassigned()

        self.state.expanded_project_names.discard(project_name)
        if self.state.selected_project == project:
            self.state.selected_project = None

        refresh_right_side = False
        if self.state.comparison_project == project:
            self.state.comparison_project = None
            if self.state.current_view == View.COMPARISON:
                self.state.current_view = View.OFFER
            refresh_right_side = True

        if self.state.opened_offer is not None and self.state.opened_offer.project_name == project_name:
            self.state.opened_offer = None
            refresh_right_side = True

        if self.state.selected_offer is not None and self.state.selected_offer.project_name == project_name:
            self.state.selected_offer = None

        self.state.extract_requested_offers = {
            offer
            for offer in self.state.extract_requested_offers
            if offer.project_name != project_name
        }

        if self.state.upload_project is None:
            self.state.upload_project = UNASSIGNED_PROJECT

        return DrawerActionResult(
            True,
            f'{project_name} verwijderd',
            refresh_drawer=True,
            refresh_right_side=refresh_right_side,
        )

    @logged_action
    def move_offer(self, offer: Offer, target_project: str | None) -> DrawerActionResult:
        try:
            new_offer = self.offer_service.move_to_project(offer, target_project)
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
            return self.failure(error)

        refresh_right_side = False
        if self.state.opened_offer == offer:
            self.state.opened_offer = new_offer
            refresh_right_side = True

        if self.state.selected_offer == offer:
            self.state.selected_offer = new_offer

        if offer in self.state.extract_requested_offers:
            self.state.extract_requested_offers.discard(offer)
            self.state.extract_requested_offers.add(new_offer)

        return DrawerActionResult(
            True,
            f'{new_offer.name} verplaatst',
            refresh_drawer=True,
            refresh_right_side=refresh_right_side,
        )

    @logged_action
    def bulk_delete_offers(self, offers: list[Offer]) -> DrawerActionResult:
        deleted = 0
        failed = []
        for offer in offers:
            if self.delete_offer(offer).success:
                deleted += 1
            else:
                failed.append(offer.name)

        message = f'{deleted} offerte(s) verwijderd'
        if failed:
            message += f"; mislukt: {', '.join(failed)}"

        return DrawerActionResult(deleted > 0, message, refresh_drawer=True, refresh_right_side=True)

    @logged_action
    def bulk_extract_offers(self, offers: list[Offer]) -> DrawerActionResult:
        started = 0
        skipped = 0
        for offer in offers:
            if self.request_extract(offer).success:
                started += 1
            else:
                skipped += 1

        message = f'Extractie gestart voor {started} offerte(s)'
        if skipped:
            message += f'; {skipped} al actief of al aangevraagd'

        return DrawerActionResult(started > 0, message, refresh_drawer=True)

    @logged_action
    def bulk_move_offers(self, offers: list[Offer], target_project: str | None) -> DrawerActionResult:
        moved = 0
        failed = []
        for offer in offers:
            if self.move_offer(offer, target_project).success:
                moved += 1
            else:
                failed.append(offer.name)

        message = f'{moved} offerte(s) verplaatst'
        if failed:
            message += f"; mislukt: {', '.join(failed)}"

        return DrawerActionResult(moved > 0, message, refresh_drawer=True, refresh_right_side=True)

    @logged_action
    def request_extract(self, offer: Offer) -> DrawerActionResult:
        if self.extraction_job_service.is_running(offer):
            return DrawerActionResult(False, f'Extractie loopt al voor {offer.name}')

        started = self.extraction_job_service.start(offer)
        if not started:
            return DrawerActionResult(False, f'Extractie loopt al voor {offer.name}')

        self.state.extract_requested_offers.add(offer)
        return DrawerActionResult(
            True,
            f'Extractie gestart voor {offer.name}',
            refresh_drawer=True,
        )

    @logged_action
    def cancel_extraction(self, offer: Offer) -> DrawerActionResult:
        if not self.extraction_job_service.cancel(offer):
            return DrawerActionResult(False, f'Geen actieve extractie om te annuleren voor {offer.name}')

        self.state.extract_requested_offers.discard(offer)
        return DrawerActionResult(True, f'Extractie geannuleerd voor {offer.name}', refresh_drawer=True)

    @staticmethod
    def failure(error: Exception) -> DrawerActionResult:
        return DrawerActionResult(False, str(error))
