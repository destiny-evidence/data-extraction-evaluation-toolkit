from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deet.data_models.processed_gold_standard_annotations import ProcessedAnnotationData
from deet.processors.base_converter import AnnotationConverter
from deet.processors.converter_register import SupportedImportFormat


@pytest.mark.parametrize("import_format", list(SupportedImportFormat))
def test_supported_import_converters_returns_processed_data_mock(import_format):
    """
    Test that each member of the SupportedImportFormat returns a working
        converter.
    """
    converter = import_format.get_annotation_converter()

    # Ensure each converter returns a subtype of AnnotationConverter
    assert isinstance(converter, AnnotationConverter)

    test_path = Path("dummy_path.json")

    with patch.object(
        type(converter), "process_annotation_file", autospec=True
    ) as mock_process:
        mock_result = MagicMock(spec=ProcessedAnnotationData)
        mock_process.return_value = mock_result

        processed = converter.process_annotation_file(test_path)

        mock_process.assert_called_once_with(converter, test_path)
        assert processed is mock_result
        assert isinstance(processed, ProcessedAnnotationData)
