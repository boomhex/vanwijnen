from pathlib import Path
from typing import Any

import json


class FolderHandler:

    def __init__(self, pdf_dir: Path, results_dir: Path, comparison_dir: Path | None = None) -> None:
        self.pdf_dir = pdf_dir
        self.results_dir = results_dir
        self.comparison_dir = comparison_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.comparison_dir is not None:
            self.comparison_dir.mkdir(parents=True, exist_ok=True)

    def files_in_folder(self, folder: Path) -> list[Path]:
        return sorted(folder.glob('*.pdf'), key=lambda file: file.name.lower())

    def projects(self) -> list[Path]:
        return sorted(
            [path for path in self.pdf_dir.iterdir() if path.is_dir()],
            key=lambda path: path.name.lower(),
        )

    def project_files(self, project: Path) -> list[Path]:
        return self.files_in_folder(project)

    def upload_folder(self, project_name: str | None) -> Path:
        if not project_name:
            return self.pdf_dir

        project_folder = self.pdf_dir / project_name
        project_folder.mkdir(parents=True, exist_ok=True)
        return project_folder

    async def add_uploaded_file(self, event, project_name: str | None) -> Path:
        destination = self.upload_folder(project_name) / event.file.name
        await event.file.save(destination)
        return destination

    def create_project(self, project_name: str | None) -> Path:
        clean_name = self.clean_name(project_name, kind='project')
        project = self.pdf_dir / clean_name
        project.mkdir(exist_ok=True)
        return project

    def rename_file(self, file: Path, new_name: str | None) -> Path:
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')

        clean_name = self.clean_name(new_name, kind='filename')
        if Path(clean_name).suffix.lower() != '.pdf':
            clean_name = f'{clean_name}.pdf'

        new_file = file.with_name(clean_name)
        if new_file == file:
            raise ValueError('Filename is unchanged')
        if new_file.exists():
            raise FileExistsError(f'{clean_name} already exists')

        old_result_path = self.result_path_for_file(file)
        new_result_path = self.result_path_for_file(new_file)

        file.rename(new_file)
        self.move_result_if_possible(old_result_path, new_result_path)
        return new_file

    def move_file(self, file: Path, target_project: str | None) -> Path:
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')

        target_folder = self.upload_folder(None if target_project == 'Unassigned' else target_project)
        new_file = target_folder / file.name
        if new_file == file:
            raise ValueError('File is already in this location')
        if new_file.exists():
            raise FileExistsError(f'{file.name} already exists in {target_project or "Unassigned"}')

        old_result_path = self.result_path_for_file(file)
        new_result_path = self.result_dir_for_file(new_file) / f'{new_file.stem}.txt'

        file.rename(new_file)
        self.move_result_if_possible(old_result_path, new_result_path)
        return new_file

    def delete_file(self, file: Path) -> None:
        if not file.exists():
            raise FileNotFoundError(f'{file.name} does not exist')

        file.unlink()

    def result_dir_for_file(self, file: Path) -> Path:
        relative_parent = file.relative_to(self.pdf_dir).parent
        if relative_parent == Path('.'):
            return self.results_dir

        return self.results_dir / relative_parent

    def result_path_for_file(self, file: Path) -> Path:
        canonical_result_path = self.result_dir_for_file(file) / f'{file.stem}.txt'
        legacy_result_path = self.results_dir / f'{file.stem}.txt'

        if canonical_result_path.exists() or canonical_result_path == legacy_result_path:
            return canonical_result_path

        if legacy_result_path.exists():
            canonical_result_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_result_path.rename(canonical_result_path)

        return canonical_result_path

    def load_result(self, file: Path) -> dict[str, Any] | None:
        result_path = self.result_path_for_file(file)
        if not result_path.exists():
            return None

        from main_page.extract_offer import parse_json_response

        return parse_json_response(result_path.read_text())

    def save_result(self, file: Path, result: dict[str, Any]) -> None:
        result_path = self.result_path_for_file(file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open('w') as result_file:
            json.dump(result, result_file, ensure_ascii=False, indent=4)

    def load_comparison(self, project: Path) -> dict[str, Any]:
        comparison_path = self.comparison_path_for_project(project)
        if not comparison_path.exists():
            return {'Posten': []}

        with comparison_path.open('r') as comparison_file:
            return json.load(comparison_file)

    def save_comparison(self, project: Path, comparison: dict[str, Any]) -> None:
        comparison_path = self.comparison_path_for_project(project)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        with comparison_path.open('w') as comparison_file:
            json.dump(comparison, comparison_file, ensure_ascii=False, indent=4)

    def comparison_path_for_project(self, project: Path) -> Path:
        if self.comparison_dir is None:
            raise ValueError('Comparison directory is not configured')

        return self.comparison_dir / f'{project.name}.json'

    @staticmethod
    def clean_name(name: str | None, *, kind: str) -> str:
        if not name:
            raise ValueError(f'Enter a {kind}')

        clean_name = name.strip()
        if not clean_name or '/' in clean_name or '\\' in clean_name or clean_name in {'.', '..'}:
            raise ValueError(f'Invalid {kind}')

        return clean_name

    @staticmethod
    def move_result_if_possible(old_result_path: Path, new_result_path: Path) -> None:
        if not old_result_path.exists() or old_result_path == new_result_path:
            return

        new_result_path.parent.mkdir(parents=True, exist_ok=True)
        if new_result_path.exists():
            return

        old_result_path.rename(new_result_path)
