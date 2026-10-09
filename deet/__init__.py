import warnings

from authlib.deprecate import AuthlibDeprecationWarning

warnings.filterwarnings(
    action="ignore",
    message=r"^The httpx module is deprecated; please use httpx2 instead\.$",
    category=AuthlibDeprecationWarning,
)
