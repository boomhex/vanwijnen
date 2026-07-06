from pathlib import Path
from typing import TYPE_CHECKING, Any

from domain.fields import FIELD_TO_ATTR, OFFER_FIELDS
from domain.offer import Offer, Posten
from utils.app_logging import logged_action

if TYPE_CHECKING:
    from services.folder_handler import FolderHandler


class OfferService:
    """Application actions for offers.

    Offer stays a light domain object; this service performs persistence and
    filesystem mutations through FolderHandler.
    """

    fields = set(OFFER_FIELDS)

    def __init__(self, folder_handler: 'FolderHandler') -> None:
        self.folder_handler = folder_handler

    def load_data(self, offer: Offer) -> dict[str, Any] | None:
        return self.folder_handler.load_result(offer.document)

    def save_data(self, offer: Offer, data: dict[str, Any]) -> None:
        self.folder_handler.save_result(offer.document, data)

    @logged_action
    def update_summary_value(self, offer: Offer, result: dict[str, Any], field: str, value: str) -> None:
        result[field] = value
        self.save_data(offer, result)

    @logged_action
    def add_summary_field(self, offer: Offer, result: dict[str, Any], field: str, value: str | None) -> None:
        result[field] = value or ''
        self.save_data(offer, result)

    @logged_action
    def add_post_row(self, offer: Offer, result: dict[str, Any]) -> Posten:
        row = Posten()
        posten = self.posten_list(result)
        posten.append(row)
        self.save_posten_list(offer, result, posten)
        return row

    @logged_action
    def update_post_row(
        self,
        offer: Offer,
        result: dict[str, Any],
        row_id: int | None,
        field: str | None,
        value: str,
    ) -> None:
        if field not in self.fields or row_id is None:
            return

        posten = self.posten_list(result)
        if row_id < 0 or row_id >= len(posten):
            return

        setattr(posten[row_id], self.field_name(field), value)
        self.save_posten_list(offer, result, posten)

    @logged_action
    def delete_post_row(self, offer: Offer, result: dict[str, Any], row_id: int | None) -> None:
        if row_id is None:
            return

        posten = self.posten_list(result)
        if row_id < 0 or row_id >= len(posten):
            return

        posten.pop(row_id)
        self.save_posten_list(offer, result, posten)

    @staticmethod
    def posten_list(result: dict[str, Any]) -> list[Posten]:
        posten = result.get('Posten', [])
        if not isinstance(posten, list):
            return []
        return [Posten.from_dict(post) for post in posten if isinstance(post, dict)]

    def save_posten_list(self, offer: Offer, result: dict[str, Any], posten: list[Posten]) -> None:
        result['Posten'] = [post.to_dict() for post in posten]
        self.save_data(offer, result)

    def save_raw_text(self, offer: Offer, text: str) -> None:
        self.folder_handler.save_raw_pdf_text(offer.document, text)

    def rename(self, offer: Offer, new_name: str | None) -> Offer:
        new_path = self.folder_handler.rename_file(offer.document, new_name).parent
        return self.folder_handler.offer_from_path(new_path)

    def move_to_project(self, offer: Offer, target_project: str | None) -> Offer:
        new_path = self.folder_handler.move_file(offer.document, target_project).parent
        return self.folder_handler.offer_from_path(new_path)

    def delete(self, offer: Offer) -> None:
        self.folder_handler.delete_file(offer.document)

    def clear_extraction(self, offer: Offer, *, include_raw_text: bool = False) -> int:
        return self.folder_handler.clear_extraction_artifacts(
            offer.document,
            include_raw_text=include_raw_text,
        )

    def offer_from_path(self, path: Path) -> Offer:
        return self.folder_handler.offer_from_path(path)

    @staticmethod
    def field_name(field: str) -> str:
        return FIELD_TO_ATTR[field]
