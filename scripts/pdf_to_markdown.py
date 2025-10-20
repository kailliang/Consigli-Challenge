from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_LOCATIONS: Sequence[Path] = (ROOT / "backend" / ".env", ROOT / ".env")
API_URL = "https://api.va.landing.ai/v1/tools/agentic-document-analysis"
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF and DOCX files to Markdown via Landing AI's agentic document analysis API."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ingest/data"),
        help="Directory to scan for documents (default: ingest/data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write Markdown files (default: mirror input directory).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the agentic document analysis endpoint "
        "(default: read from .env files or AGENTIC_DOCUMENT_ANALYSIS_API_KEY / agentic_document_analysis_api_key).",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata in the generated Markdown.",
    )
    parser.add_argument(
        "--include-marginalia",
        action="store_true",
        help="Include marginalia in the generated Markdown.",
    )
    parser.add_argument(
        "--disable-rotation-detection",
        action="store_true",
        help="Disable automatic page rotation detection.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=None,
        help="Additional .env file(s) to load before resolving the API key.",
    )
    return parser.parse_args()


def _load_env_files(locations: Iterable[Path]) -> None:
    for location in locations:
        if not location or not location.exists():
            continue
        try:
            lines = location.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_api_key(explicit: str | None) -> str:
    api_key = (
        explicit
        or os.getenv("AGENTIC_DOCUMENT_ANALYSIS_API_KEY")
        or os.getenv("agentic_document_analysis_api_key")
    )
    if not api_key:
        message = (
            "Agentic document analysis API key not provided. "
            "Set --api-key or define AGENTIC_DOCUMENT_ANALYSIS_API_KEY / agentic_document_analysis_api_key."
        )
        raise SystemExit(message)
    return api_key


def gather_documents(root: Path) -> list[Path]:
    if not root.exists():
        return []

    documents: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(path)
    documents.sort()
    return documents


def ensure_output_path(input_root: Path, output_root: Path | None, document: Path) -> Path:
    root = output_root or input_root
    try:
        relative = document.relative_to(input_root)
    except ValueError:
        relative = document.name
    target = root / Path(relative).with_suffix(".md")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _guess_mime_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "application/pdf"
    if path.suffix.lower() == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def convert_document(
    document: Path,
    *,
    output_path: Path,
    api_key: str,
    include_metadata: bool,
    include_marginalia: bool,
    enable_rotation_detection: bool,
    timeout: float,
) -> bool:
    with document.open("rb") as stream:
        files = {
            "pdf": (document.name, stream, _guess_mime_type(document)),
        }
        data = {
            "include_metadata_in_markdown": str(include_metadata).lower(),
            "include_marginalia": str(include_marginalia).lower(),
            "enable_rotation_detection": str(enable_rotation_detection).lower(),
        }
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            response = requests.post(API_URL, files=files, data=data, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            print(f"[error] Failed to process {document}: {exc}", file=sys.stderr)
            return False

    if response.status_code >= 400:
        print(
            f"[error] API returned status {response.status_code} for {document}: {response.text}",
            file=sys.stderr,
        )
        return False

    try:
        payload = response.json()
    except ValueError:
        print(f"[error] Non-JSON response for {document}: {response.text[:200]}", file=sys.stderr)
        return False

    markdown = payload.get("markdown")
    if markdown is None and isinstance(payload.get("data"), dict):
        markdown = payload["data"].get("markdown")

    if not isinstance(markdown, str):
        print(f"[error] Missing 'markdown' field in response for {document}: {payload}", file=sys.stderr)
        return False

    output_path.write_text(markdown, encoding="utf-8")
    print(f"[ok] Wrote {output_path}")
    return True


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else None

    env_locations = list(DEFAULT_ENV_LOCATIONS)
    if args.env_file:
        env_locations.extend(args.env_file)
    _load_env_files(env_locations)

    documents = gather_documents(input_dir)
    if not documents:
        print(f"No PDF or DOCX files found under {input_dir}.")
        return 0

    api_key = resolve_api_key(args.api_key)
    success_count = 0

    for document in documents:
        output_path = ensure_output_path(input_dir, output_dir, document)
        success = convert_document(
            document,
            output_path=output_path,
            api_key=api_key,
            include_metadata=args.include_metadata,
            include_marginalia=args.include_marginalia,
            enable_rotation_detection=not args.disable_rotation_detection,
            timeout=args.timeout,
        )
        if success:
            success_count += 1

    print(f"Converted {success_count} of {len(documents)} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
