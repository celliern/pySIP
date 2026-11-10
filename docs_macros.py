"""Render stored notebook content for the Zensical cookbook without execution."""

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote

from nbconvert import MarkdownExporter

DOCS_ROOT = Path(__file__).resolve().parent / "docs"


def notebook(path: str) -> str:
    """Render stored notebook content without executing code cells.

    Parameters
    ----------
    path : str
        Path to an existing ``.ipynb`` file, relative to ``docs/``.

    Returns
    -------
    str
        Markdown cells, code, and stored outputs, with embedded image resources.

    Raises
    ------
    ValueError
        The path is absolute, escapes ``docs/``, or has another extension.
    FileNotFoundError
        The requested notebook does not exist.
    """
    source = (DOCS_ROOT / path).resolve()
    if (
        Path(path).is_absolute()
        or not source.is_relative_to(DOCS_ROOT.resolve())
        or source.suffix != ".ipynb"
    ):
        raise ValueError("Notebook paths must name a .ipynb file inside docs/")

    markdown, resources = MarkdownExporter().from_filename(
        str(source), resources={"output_files_dir": "", "unique_key": source.stem}
    )
    for filename, data in resources.get("outputs", {}).items():
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        encoded = base64.b64encode(data).decode("ascii")
        uri = f"data:{mime_type};base64,{encoded}"
        markdown = markdown.replace(f"({quote(filename)})", f"({uri})")
    return markdown


def define_env(env: Any) -> None:
    """Register the notebook renderer with Zensical's dynamic macro environment."""
    env.macro(notebook)
