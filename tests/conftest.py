import json
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from vcr.request import Request

from deet.processors.converter_register import SupportedImportFormat
from deet.processors.parser import ParsedOutput
from deet.settings import get_settings


@pytest.fixture
def valid_parsed_pdf():
    with Path.open("tests/test_files/output/test_file_for_parser.md") as infile:
        return infile.read().lower()


@pytest.fixture
def valid_parsed_epub():
    with Path.open("tests/test_files/output/conrad-epub-parsed.md") as infile:
        return infile.read()


@pytest.fixture
def valid_parsed_html():
    with Path.open("tests/test_files/output/conrad-html-parsed.md") as infile:
        return infile.read()


@pytest.fixture
def mock_check_language(monkeypatch):
    """Stub the language checker."""
    monkeypatch.setattr(
        "deet.processors.parser.check_language",
        lambda txt, lang=None, threshold=0.2: txt.strip() != "not english",  # noqa: ARG005
    )


@pytest.fixture
def mock_pdfminerparser_parse(monkeypatch):
    """Stub PdfminerParser.parse to avoid actual PDF parsing."""

    def _stub_parse(
        cls,
        input_,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        return ParsedOutput(text="dummy pdfminer text", parser_library="pdfminer")

    monkeypatch.setattr(
        "deet.processors.parser.PdfminerParser.parse",
        classmethod(_stub_parse),
    )


@pytest.fixture
def sample_eppi_data() -> dict:
    """Sample EPPI-style data structure as a dict."""
    return {
        "CodeSets": [
            {
                "SetName": "Arms",
                "SetId": 105797,
                "Attributes": {
                    "AttributesList": [
                        {
                            "AttributeId": 5730447,
                            "AttributeName": "Arm name",
                            "AttributeType": "Selectable (show checkbox)",
                        }
                    ]
                },
            },
            {
                "SetName": "New Prioritised Codeset",
                "SetId": 111925,
                "Attributes": {
                    "AttributesList": [
                        {
                            "AttributeId": 6080465,
                            "AttributeName": "Population",
                            "AttributeType": "Selectable (show checkbox)",
                            "Attributes": {
                                "AttributesList": [
                                    {
                                        "AttributeId": 6080480,
                                        "AttributeName": "Aggregate age",
                                        "AttributeType": "Selectable (show checkbox)",
                                    },
                                    {
                                        "AttributeId": 6080481,
                                        "AttributeName": "Mean age",
                                        "AttributeType": "Selectable (show checkbox)",
                                    },
                                ]
                            },
                        },
                        {
                            "AttributeId": 6080466,
                            "AttributeName": "Setting",
                            "AttributeType": "Selectable (show checkbox)",
                        },
                    ]
                },
            },
        ],
        "References": [
            {
                "ItemId": 28856292,
                "Title": "A title",
                "ShortTitle": "Smith (2014)",
                "Year": "2014",
                "Abstract": "Lorem ipsum",
                "Authors": "Smith;",
                "Codes": [
                    {
                        "AttributeId": 5730447,
                        "AdditionalText": "Dolor si amet...",
                        "ItemAttributeFullTextDetails": [
                            {
                                "ItemDocumentId": 423106,
                                "TextFrom": 0,
                                "TextTo": 0,
                                "Text": 'Page 1:\n[¬s]"Dolor si amet...[¬e]"',
                                "IsFromPDF": True,
                                "DocTitle": "Smith (2014).pdf",
                                "ItemArm": "",
                            }
                        ],
                        "ArmId": 3,
                        "ArmTitle": "Lorem ipsum",
                    },
                    {
                        "AttributeId": 6080466,
                        "AdditionalText": "1",
                        "ItemAttributeFullTextDetails": [],
                        "ArmId": 0,
                        "ArmTitle": "",
                    },
                    {
                        "AttributeId": 123,
                        "AdditionalText": "1",
                        "ItemAttributeFullTextDetails": [],
                        "ArmId": 0,
                        "ArmTitle": "",
                    },
                ],
            }
        ],
    }


@pytest.fixture
def sample_eppi_data_duplicated_annotations(sample_eppi_data):
    duplicated = deepcopy(sample_eppi_data)
    for ref in duplicated["References"]:
        ref["Codes"] += ref["Codes"]

    return duplicated


@pytest.fixture
def valid_project_data(tmp_path, sample_eppi_data):
    # Create a real dummy file so Pydantic's FilePath is happy
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(sample_eppi_data))

    return {
        "name": "TestProject",
        "gold_standard_data_path": data_file,
        "gold_standard_data_format": SupportedImportFormat.EPPI_JSON,
        "environment_file": "project",
        "pdf_dir": tmp_path,  # A real directory
    }


def normalise_nested_json(data: str):
    """Recursiveley resolve and normalise nested JSON strings."""
    if isinstance(data, str):
        normalised_str = unicodedata.normalize("NFC", data)

        stripped = normalised_str.strip()
        if stripped.startswith(("{", "[")):
            try:
                return normalise_nested_json(json.loads(stripped))
            except (json.JSONDecodeError, TypeError):
                pass
        return normalised_str

    if isinstance(data, dict):
        return {k: normalise_nested_json(v) for k, v in data.items()}

    if isinstance(data, list):
        return [normalise_nested_json(item) for item in data]

    return data


def json_body_match(r1, r2) -> bool:
    """Parse two vcr request bodies as JSON for strict comparison."""
    try:
        body1 = json.loads(
            r1.body.decode("utf-8") if isinstance(r1.body, bytes) else r1.body
        )
        body2 = json.loads(
            r2.body.decode("utf-8") if isinstance(r2.body, bytes) else r2.body
        )

        cleaned_body1 = normalise_nested_json(body1)
        cleaned_body2 = normalise_nested_json(body2)

        return bool(cleaned_body1 == cleaned_body2)

    except (json.JSONDecodeError, AttributeError, TypeError):
        return bool(r1.body == r2.body)


def scrub_response_secrets(response: dict[str, Any]):
    """Scrub secrets from the response body before VCR saves it."""
    settings = get_settings()

    clean_secrets = [
        val.get_secret_value()
        for _, val in settings
        if isinstance(val, SecretStr)
        and val.get_secret_value()
        and len(val.get_secret_value()) > 4
    ]

    body_data = response["body"]["string"]

    # If VCR has decoded it to a string, decode it to modify it
    is_bytes = isinstance(body_data, bytes)
    body_str = body_data.decode("utf-8") if is_bytes else body_data

    # Scrub every secret found in the text
    for secret in clean_secrets:
        body_str = body_str.replace(secret, "DUMMY_SECRET")

    # Re-encode back to the original type so VCR doesn't break
    response["body"]["string"] = body_str.encode("utf-8") if is_bytes else body_str

    # 2. Fix the Content-Length to match the newly scrubbed size perfectly
    if "headers" in response:
        headers = response["headers"]
        # Handle case-insensitive headers
        for k in list(headers.keys()):
            if k.lower() == "content-length":
                actual_len = len(response["body"]["string"])
                headers[k] = [str(actual_len)]

    return response


def scrub_request_uri(request: Request) -> Request:
    """Remove secrets from uri."""
    settings = get_settings()

    clean_secrets = [
        val.get_secret_value()
        for _, val in settings
        if isinstance(val, SecretStr)
        and val.get_secret_value()
        and len(val.get_secret_value()) > 4
    ]

    for secret in clean_secrets:
        if secret.lower() in request.uri.lower():
            request.uri = "https://dummy.secret/"

    return request


def pytest_recording_configure(config, vcr):
    """Use plugin hook to configure VCR instance to scrub secrets from casettes."""
    vcr.register_matcher("json_body", json_body_match)


@pytest.fixture(scope="module")
def vcr_config():
    """Instruct pytest-recording to use the custom registered serializer."""
    base_cassette_path = Path(__file__).parent / "integration" / "cassettes"

    return {
        "ignore_hosts": ["raw.githubusercontent.com"],
        "decode_compressed_response": True,
        "match_on": ["method", "uri", "json_body"],
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "before_record_request": scrub_request_uri,
        "before_record_response": scrub_response_secrets,
        "cassette_dir": str(base_cassette_path),
    }
