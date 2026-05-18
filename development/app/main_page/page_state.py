from pathlib import Path


class MainPageState:
    def __init__(self) -> None:
        self.opened_file: Path | None = None
        self.current_view = 'offer'
        self.comparison_project: Path | None = None
        self.upload_project: str | None = None
        self.extract_requested_files: set[Path] = set()
