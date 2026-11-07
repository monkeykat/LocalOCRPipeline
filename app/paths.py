import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_INPUT_DIR = Path(
    "/home/gulfaero/Documents/Code/AviationRegs/AviationRegs/documents"
)


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
    input_name = Path(os.getenv("OCR_INPUT_DIR", str(_DEFAULT_INPUT_DIR))).expanduser()
    input_dir = input_name if input_name.is_absolute() else data_dir / input_name
    output_name = Path(os.getenv("OCR_OUTPUT_DIR", "output_PaddleOCR"))
    output_dir = output_name if output_name.is_absolute() else data_dir / output_name
    return ProjectPaths(
        project_root=root,
        data_dir=data_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        failed_dir=data_dir / "failed",
        database_path=data_dir / "documents.sqlite3",
    )


def ensure_data_directories(paths: ProjectPaths) -> None:
    if not paths.input_dir.exists():
        paths.input_dir.mkdir(parents=True)
    elif not paths.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {paths.input_dir}")
    for directory in (paths.output_dir, paths.failed_dir):
        directory.mkdir(parents=True, exist_ok=True)


def output_path_for(source_path: Path, paths: ProjectPaths, suffix: str = ".txt") -> Path:
    relative_path = source_path.resolve().relative_to(paths.input_dir.resolve())
    return (paths.output_dir / relative_path).with_suffix(suffix)


def failed_path_for(source_path: Path, paths: ProjectPaths) -> Path:
    relative_path = source_path.resolve().relative_to(paths.input_dir.resolve())
    return paths.failed_dir / relative_path
