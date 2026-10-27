"""PDF text extraction with source filename and page metadata."""

from io import BytesIO
from typing import Any

from pypdf import PdfReader


def extract_pages(uploaded_files: list[Any]) -> list[dict[str, Any]]:
    """Read uploaded PDFs. Blank pages are skipped; page numbers start at 1."""
    pages: list[dict[str, Any]] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            if reader.is_encrypted:
                errors.append(f"{uploaded_file.name}: encrypted PDF")
                continue

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append(
                        {"source": uploaded_file.name, "page": page_number, "text": text}
                    )
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    if errors and not pages:
        raise ValueError("Could not extract text. " + "; ".join(errors))
    return pages
