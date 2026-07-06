from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from domain.offer import Offer
    from services.project import Project


UNASSIGNED_PROJECT = 'Unassigned'


class FolderHandler:
    """Manages file storage with structure: storage/<project>/<offer>/{document.pdf, extract.json, raw.txt}"""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def projects(self) -> list['Project']:
        """Return all projects."""
        from services.project import Project

        return [Project(path, self) for path in self.project_paths()]

    def project_paths(self) -> list[Path]:
        """Return all project directories."""
        project_dirs = [path for path in self.storage_dir.iterdir() if path.is_dir()]
        return sorted(project_dirs, key=lambda path: path.name.lower())

    def project_path(self, project_name: str | None) -> Path:
        """Get the path for a project by name."""
        clean_name = self.clean_name(project_name, kind='project')
        return self.storage_dir / clean_name

    def project_offer_paths(self, project: Path | 'Project') -> list[Path]:
        """Return all offer folders within a project."""
        project = self._project_path(project)
        if not project.exists():
            return []
        offer_dirs = [path for path in project.iterdir() if path.is_dir() and path.name != 'offers']
        return sorted(offer_dirs, key=lambda path: path.name.lower())

    def project_offers(self, project: Path | 'Project') -> list['Offer']:
        """Return all offers within a project."""
        return [self.offer_from_path(offer_dir) for offer_dir in self.project_offer_paths(project)]

    def project_files(self, project: Path | 'Project') -> list[Path]:
        """Return list of PDFs from all offers in project (for backward compatibility with sidebar)."""
        pdfs = []
        for offer_dir in self.project_offer_paths(project):
            pdf = offer_dir / 'document.pdf'
            if pdf.exists():
                pdfs.append(pdf)
        return sorted(pdfs, key=lambda f: f.name.lower())

    def offer_from_file(self, file: Path) -> 'Offer':
        """Create an Offer instance from a file path."""
        from domain.offer import Offer
        offer_dir = self.offer_dir_for_file(file)
        return Offer(offer_dir)

    def offer_from_path(self, offer_dir: Path) -> 'Offer':
        """Create an Offer instance from an offer folder path."""
        from domain.offer import Offer
        return Offer(offer_dir)

    def offer_dir_for_file(self, file: Path) -> Path:
        """Get the offer folder containing a file."""
        if file.name == 'document.pdf':
            return file.parent
        # For any file in an offer folder, return the parent (offer folder)
        relative = file.relative_to(self.storage_dir)
        if len(relative.parts) < 2:
            raise ValueError(f'Cannot resolve offer for file: {file}')
        return self.storage_dir / relative.parts[0] / relative.parts[1]

    def project_dir_for_file(self, file: Path) -> Path:
        """Get the project folder containing a file."""
        relative = file.relative_to(self.storage_dir)
        if not relative.parts:
            raise ValueError(f'Cannot resolve project for file outside storage: {file}')
        return self.storage_dir / relative.parts[0]

    def project_name_for_file(self, file: Path) -> str:
        """Get the project name for a file."""
        return self.project_dir_for_file(file).name

    def offer_name_for_file(self, file: Path) -> str:
        """Get the offer name for a file."""
        return self.offer_dir_for_file(file).name

    def offer_document_path(self, offer: Path) -> Path:
        """Get the PDF path for an offer."""
        return offer / 'document.pdf'

    def offer_extract_path(self, offer: Path) -> Path:
        """Get the extract.json path for an offer."""
        return offer / 'extract.json'

    def offer_raw_path(self, offer: Path) -> Path:
        """Get the raw.txt path for an offer."""
        return offer / 'raw.txt'

    def offer_llm_response_path(self, offer: Path) -> Path:
        """Get the llm_response.txt path for an offer."""
        return offer / 'llm_response.txt'

    def offer_status_path(self, offer: Path) -> Path:
        """Get the extraction status path for an offer."""
        return offer / 'status.json'

    def project_comparison_path(self, project: Path) -> Path:
        """Get the comparison.json path for a project."""
        return project / 'comparison.json'

    def project_comparison_status_path(self, project: Path) -> Path:
        """Get the comparison_status.json path for a project."""
        return project / 'comparison_status.json'

    def project_comparison_llm_response_path(self, project: Path) -> Path:
        """Get the raw comparison matching LLM response path for a project."""
        return project / 'comparison_llm_response.txt'

    def comparison_path_for_file(self, file: Path) -> Path:
        """Get the comparison.json for the project containing a file."""
        return self.project_comparison_path(self.project_dir_for_file(file))

    def result_path_for_file(self, file: Path) -> Path:
        """Get the extract.json path for a file (PDF or extract)."""
        offer = self.offer_dir_for_file(file)
        return self.offer_extract_path(offer)

    def raw_path_for_file(self, file: Path) -> Path:
        """Get the raw.txt path for a file."""
        offer = self.offer_dir_for_file(file)
        return self.offer_raw_path(offer)

    def llm_response_path_for_file(self, file: Path) -> Path:
        """Get the llm_response.txt path for a file."""
        offer = self.offer_dir_for_file(file)
        return self.offer_llm_response_path(offer)

    def named_llm_response_path_for_file(self, file: Path, name: str) -> Path:
        """Get a named LLM response path for a file."""
        safe_name = self.clean_name(name, kind='response name')
        if not safe_name.endswith('.txt'):
            safe_name = f'{safe_name}.txt'
        offer = self.offer_dir_for_file(file)
        return offer / safe_name

    def status_path_for_file(self, file: Path) -> Path:
        """Get the status.json path for a file."""
        offer = self.offer_dir_for_file(file)
        return self.offer_status_path(offer)

    def upload_folder(self, project_name: str | None) -> Path:
        """Get the upload destination folder for a project."""
        project = self.project_path(project_name or UNASSIGNED_PROJECT)
        project.mkdir(parents=True, exist_ok=True)
        return project

    async def add_uploaded_file(self, event, project_name: str | None) -> Path:
        """Save an uploaded PDF file to a new offer folder."""
        project = self.upload_folder(project_name)
        
        # Create offer folder named after the file
        offer_name = Path(event.file.name).stem
        offer_dir = project / offer_name
        offer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PDF as document.pdf
        destination = offer_dir / 'document.pdf'
        # If a file with that name already exists, we might need to handle it
        if destination.exists():
            raise FileExistsError(f'Offer "{offer_name}" already exists in {project_name or "Unassigned"}')
        
        await event.file.save(destination)
        return destination

    def create_project(self, project_name: str | None) -> 'Project':
        """Create a new project."""
        project = self.project_path(project_name)
        project.mkdir(parents=True, exist_ok=True)
        from services.project import Project
        return Project(project, self)

    def rename_project(self, project: Path | 'Project', new_name: str | None) -> 'Project':
        """Rename a project folder."""
        project_path = self._project_path(project)
        if not project_path.exists():
            raise FileNotFoundError(f'Project "{project_path.name}" does not exist')

        clean_name = self.clean_name(new_name, kind='project')
        new_project_path = self.storage_dir / clean_name

        if new_project_path == project_path:
            raise ValueError('Project name is unchanged')
        if new_project_path.exists():
            raise FileExistsError(f'Project "{clean_name}" already exists')

        project_path.rename(new_project_path)
        from services.project import Project
        return Project(new_project_path, self)

    def delete_project(self, project: Path | 'Project') -> None:
        """Delete a project folder and all offers inside it."""
        project_path = self._project_path(project)
        if not project_path.exists():
            raise FileNotFoundError(f'Project "{project_path.name}" does not exist')
        if project_path.parent != self.storage_dir:
            raise ValueError('Can only delete project folders inside storage')

        import shutil
        shutil.rmtree(project_path)

    def rename_file(self, file: Path, new_name: str | None) -> Path:
        """Rename a PDF file and its associated data."""
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')

        if file.name != 'document.pdf':
            raise ValueError('Can only rename files in offer folders')

        offer_dir = file.parent
        clean_name = self.clean_name(new_name, kind='filename')
        new_offer_name = clean_name
        new_offer_dir = offer_dir.parent / new_offer_name

        if new_offer_dir == offer_dir:
            raise ValueError('Offer name is unchanged')
        if new_offer_dir.exists():
            raise FileExistsError(f'Offer "{new_offer_name}" already exists')

        # Rename the entire offer folder
        offer_dir.rename(new_offer_dir)
        return new_offer_dir / 'document.pdf'

    def move_file(self, file: Path, target_project: str | None) -> Path:
        """Move a file to a different project."""
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')

        if file.name != 'document.pdf':
            raise ValueError('Can only move files in offer folders')

        offer_dir = file.parent
        offer_name = offer_dir.name
        target_project_dir = self.upload_folder(None if target_project == 'Unassigned' else target_project)
        new_offer_dir = target_project_dir / offer_name

        if new_offer_dir == offer_dir:
            raise ValueError('File is already in this location')
        if new_offer_dir.exists():
            raise FileExistsError(f'Offer "{offer_name}" already exists in {target_project or "Unassigned"}')

        # Move the entire offer folder
        import shutil
        shutil.move(str(offer_dir), str(new_offer_dir))
        return new_offer_dir / 'document.pdf'

    def delete_file(self, file: Path) -> None:
        """Delete a file and its entire offer folder."""
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')

        if file.name != 'document.pdf':
            raise ValueError('Can only delete files in offer folders')

        offer_dir = file.parent
        import shutil
        shutil.rmtree(offer_dir)

    def clear_extraction_artifacts(self, file: Path, *, include_raw_text: bool = False) -> int:
        """Remove saved extraction output for an offer while keeping the PDF."""
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')
        if file.name != 'document.pdf':
            raise ValueError('Can only clear extraction artifacts for files in offer folders')

        artifact_paths = [
            self.result_path_for_file(file),
            self.llm_response_path_for_file(file),
            self.status_path_for_file(file),
        ]
        artifact_paths.extend(self.offer_dir_for_file(file).glob('llm_*_response.txt'))
        if include_raw_text:
            artifact_paths.append(self.raw_path_for_file(file))

        removed = 0
        for artifact_path in dict.fromkeys(artifact_paths):
            if artifact_path.exists() and artifact_path.is_file():
                artifact_path.unlink()
                removed += 1

        return removed

    def load_result(self, file: Path) -> dict[str, Any] | None:
        """Load the extract.json for a file."""
        extract_path = self.result_path_for_file(file)
        if not extract_path.exists():
            return None

        with extract_path.open('r') as f:
            return json.load(f)

    def save_result(self, file: Path, result: dict[str, Any]) -> None:
        """Save the extract.json for a file."""
        extract_path = self.result_path_for_file(file)
        extract_path.parent.mkdir(parents=True, exist_ok=True)
        with extract_path.open('w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def save_raw_pdf_text(self, file: Path, text: str) -> None:
        """Save the raw plain-text extraction from PDF."""
        raw_path = self.raw_path_for_file(file)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text)

    def save_llm_response(self, file: Path, text: str) -> None:
        """Save the raw LLM response for debugging."""
        response_path = self.llm_response_path_for_file(file)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(text)

    def save_named_llm_response(self, file: Path, name: str, text: str) -> None:
        """Save a named raw LLM response for debugging."""
        response_path = self.named_llm_response_path_for_file(file, name)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(text)

    def load_extraction_status(self, file: Path) -> dict[str, Any] | None:
        """Load the extraction status for a file."""
        status_path = self.status_path_for_file(file)
        if not status_path.exists():
            return None

        try:
            with status_path.open('r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def save_extraction_status(self, file: Path, status: dict[str, Any]) -> None:
        """Save the extraction status for a file."""
        status_path = self.status_path_for_file(file)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = status_path.with_suffix('.json.tmp')
        with temporary_path.open('w') as f:
            json.dump(status, f, ensure_ascii=False, indent=4)
        temporary_path.replace(status_path)

    def load_comparison(self, project_or_file: Path | 'Project') -> dict[str, Any]:
        """Load comparison.json for a project or file."""
        comparison_path = self._comparison_path(project_or_file)
        if not comparison_path.exists():
            return {'Posten': []}

        with comparison_path.open('r') as f:
            return json.load(f)

    def save_comparison(self, project_or_file: Path | 'Project', comparison: dict[str, Any]) -> None:
        """Save comparison.json for a project or file."""
        comparison_path = self._comparison_path(project_or_file)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with comparison_path.open('w') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=4)

    def load_comparison_status(self, project: Path | 'Project') -> dict[str, Any] | None:
        """Load the comparison/matching status for a project."""
        status_path = self.project_comparison_status_path(self._project_path(project))
        if not status_path.exists():
            return None

        try:
            with status_path.open('r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def save_comparison_status(self, project: Path | 'Project', status: dict[str, Any]) -> None:
        """Save the comparison/matching status for a project."""
        status_path = self.project_comparison_status_path(self._project_path(project))
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = status_path.with_suffix('.json.tmp')
        with temporary_path.open('w') as f:
            json.dump(status, f, ensure_ascii=False, indent=4)
        temporary_path.replace(status_path)

    def save_comparison_llm_response(self, project: Path | 'Project', text: str) -> None:
        """Save the raw LLM response for comparison matching."""
        response_path = self.project_comparison_llm_response_path(self._project_path(project))
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(text)

    @staticmethod
    def clean_name(name: str | None, *, kind: str) -> str:
        """Clean and validate a name (project, file, etc)."""
        if not name:
            raise ValueError(f'Enter a {kind}')

        clean_name = name.strip()
        if not clean_name or '/' in clean_name or '\\' in clean_name or clean_name in {'.', '..'}:
            raise ValueError(f'Invalid {kind}')

        return clean_name

    def _comparison_path(self, project_or_file: Path | 'Project') -> Path:
        """Get the comparison.json path for a project or file."""
        project_or_file = self._project_path(project_or_file)
        if project_or_file.is_dir():
            return self.project_comparison_path(project_or_file)
        return self.comparison_path_for_file(project_or_file)

    @staticmethod
    def _project_path(project_or_path: Path | 'Project') -> Path:
        if isinstance(project_or_path, Path):
            return project_or_path

        return project_or_path.path
