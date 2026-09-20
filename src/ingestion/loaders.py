import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class Document:
    content: str
    source: str
    file_type: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def load_file(self, file_path: str | Path) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}")

        if ext in {".md", ".markdown"}:
            return self._load_markdown(path)
        elif ext in {".html", ".htm"}:
            return self._load_html(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        else:
            return self._load_text(path)

    def _load_text(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return [
            Document(
                content=self._clean_text(raw),
                source=path.name,
                file_type="text",
                metadata={"file_path": str(path.resolve())},
            )
        ]

    def _load_markdown(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        heading_match = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
        doc_title = heading_match.group(1).strip() if heading_match else path.stem

        return [
            Document(
                content=self._clean_text(raw),
                source=path.name,
                file_type="markdown",
                metadata={
                    "file_path": str(path.resolve()),
                    "document_title": doc_title,
                },
            )
        ]

    def _load_html(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
        text = soup.get_text(separator="\n")

        return [
            Document(
                content=self._clean_text(text),
                source=path.name,
                file_type="html",
                metadata={
                    "file_path": str(path.resolve()),
                    "document_title": title,
                },
            )
        ]

    def _load_pdf(self, path: Path) -> List[Document]:
        reader = PdfReader(str(path))
        docs: List[Document] = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned = self._clean_text(page_text)
            if cleaned:
                docs.append(
                    Document(
                        content=cleaned,
                        source=path.name,
                        file_type="pdf",
                        metadata={
                            "file_path": str(path.resolve()),
                            "page_number": page_idx + 1,
                            "total_pages": len(reader.pages),
                        },
                    )
                )
        return docs