from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    input_dir: Path
    output_dir: Path
    failed_dir: Path
    database_path: Path


def get_project_paths(project_root: Path | None = None) -> ProjectPaths:
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    data_dir = root / "data"
    return ProjectPaths(
        project_root=root,
        data_dir=data_dir,
        input_dir=data_dir / "input",
        output_dir=data_dir / "output",
        failed_dir=data_dir / "failed",
        database_path=data_dir / "documents.sqlite3",
    )


def ensure_data_directories(paths: ProjectPaths) -> None:
    for directory in (paths.input_dir, paths.output_dir, paths.failed_dir):
        directory.mkdir(parents=True, exist_ok=True)


def output_path_for(source_path: Path, paths: ProjectPaths, suffix: str = ".txt") -> Path:
    relative_path = source_path.resolve().relative_to(paths.input_dir.resolve())
    return (paths.output_dir / relative_path).with_suffix(suffix)


def failed_path_for(source_path: Path, paths: ProjectPaths) -> Path:
    relative_path = source_path.resolve().relative_to(paths.input_dir.resolve())
    return paths.failed_dir / relative_path
