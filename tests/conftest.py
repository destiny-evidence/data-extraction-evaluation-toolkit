from copy import deepcopy
from pathlib import Path

import pytest


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
