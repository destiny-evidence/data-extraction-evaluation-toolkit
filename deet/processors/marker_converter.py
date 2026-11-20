"""Disk-cached marker PDF converter."""

from typing import TYPE_CHECKING

from diskcache import Cache
from loguru import logger

from deet.settings import get_settings

if TYPE_CHECKING:
    from marker.converters.pdf import PdfConverter

CACHE_DIR = get_settings().base_disk_cache_dir / "marker_parser_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
parser_cache = Cache(str(CACHE_DIR))


@parser_cache.memoize(typed=True, expire=None, tag="marker-converter")
def get_converter() -> "PdfConverter":
    """Build the marker converter, cached on disk."""
    logger.debug("Initializing marker converter...")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    artifact_dict = create_model_dict()
    return PdfConverter(artifact_dict=artifact_dict)
