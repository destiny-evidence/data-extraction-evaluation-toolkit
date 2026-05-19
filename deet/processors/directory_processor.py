"""Tools to create Documents directly from pdf or md files."""

from collections.abc import Sequence
from pathlib import Path

from destiny_sdk.references import ReferenceFileInput

from deet.data_models.documents import Document
from deet.processors.linker import DocumentReferenceLinker, DocumentReferenceMapping
from deet.utils.identifier_utils import hash_n_strings_to_document_id


def create_documents_from_directory(directory_path: Path) -> Sequence[Document]:
    """Find PDF and Markdown files turns them into linked documents."""
    target_files = list(directory_path.glob("*.md"))
    md_stems = {p.stem for p in target_files}

    target_files.extend(
        p for p in directory_path.glob("*.pdf") if p.stem not in md_stems
    )

    mock_references = []
    mock_mappings = []

    for file_path in target_files:
        document_id = hash_n_strings_to_document_id([file_path.stem])

        mock_ref = Document(
            name=file_path.name,
            document_id=document_id,
            is_linked=False,
            is_final=False,
            citation=ReferenceFileInput(),
        )
        mock_ref.init_document_identity()

        mock_references.append(mock_ref)
        mock_mappings.append(
            DocumentReferenceMapping(document_id=document_id, file_path=file_path)
        )

    if not mock_references:
        return []

    linker = DocumentReferenceLinker(
        references=mock_references,
        document_reference_mapping=mock_mappings,
        document_base_dir=directory_path,
    )
    return linker.link_many_references_parsed_documents()
