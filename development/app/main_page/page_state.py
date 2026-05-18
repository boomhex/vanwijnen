from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class MainPageState:
    opened_file: Path | None = None
    current_view: str = 'offer'
    comparison_project: Path | None = None
    upload_project: str | None = None
    extract_requested_files: set[Path] = field(default_factory=set)
