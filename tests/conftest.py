import json
from pathlib import Path

import pytest

from deet.processors.converter_register import SupportedImportFormat


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
