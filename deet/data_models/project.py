"""Data models for working with projects."""

import itertools
import json
import random
from collections.abc import Generator
from pathlib import Path

from pydantic import BaseModel, computed_field

from deet.data_models.base import Attribute, GoldStandardAnnotatedDocument


class DeetProject(BaseModel):
    """Data structure for a deet project."""

    path: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def p(self) -> Path:
        """Return path where the folder lives."""
        return Path(self.path)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_data(self) -> Path:
        """Return path to raw data folder."""
        return self.p / "raw_data"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def proc_data(self) -> Path:
        """Return path to processed data folder."""
        return self.p / "processed_data"

    def read_annotated_documents(
        self,
        sample: int | None = None,
        exclude: list[str] | None = None,
        batch_ids: list[str] | None = None,
    ) -> Generator[GoldStandardAnnotatedDocument]:
        """Read annotated documents."""
        path = self.p / "processed_data" / "annotated_documents.json"
        with path.open() as f:
            if batch_ids is not None:
                for doc in json.load(f):
                    document = GoldStandardAnnotatedDocument.model_validate(doc)
                    if document.document_id in batch_ids:
                        yield document
            else:
                if exclude is None:
                    exclude = []
                documents = [
                    GoldStandardAnnotatedDocument.model_validate(document)
                    for document in json.load(f)
                    if GoldStandardAnnotatedDocument.model_validate(
                        document
                    ).document_id
                    not in exclude
                ]
                if sample is None:
                    sample = len(documents)
                sample = min(sample, len(documents))
                sample_ids = random.choices(range(len(documents)), k=sample)  # noqa: S311
                for i, document in enumerate(documents):
                    if i in sample_ids:
                        yield document

    def read_attributes(self) -> list[Attribute]:
        """Read the attributes of a project."""
        with self.proc_data.joinpath("attributes.json").open() as f:
            return [Attribute.model_validate(att) for att in json.load(f)]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prompt_folder(self) -> Path:
        """Return path to prompt folder."""
        return self.p / "prompts"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def batch_folder(self) -> Path:
        """Return path to batch folder."""
        return self.p / "batches"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def folders(self) -> list:
        """Return list of mandatory folders for project."""
        return [self.raw_data, self.proc_data, self.prompt_folder, self.batch_folder]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def documents_in_batches(self) -> list[str]:
        """Return ids of documents in all batches."""
        return list(
            itertools.chain.from_iterable(
                [json.load(f.joinpath("batch_ids.json").open()) for f in self.batches]
            )
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def batches(self) -> list[Path]:
        """Return list of batches."""
        return list(self.batch_folder.iterdir())

    def folders_exist(self) -> bool:
        """Check all required folders exist."""
        return all(f.exists() for f in self.folders)
