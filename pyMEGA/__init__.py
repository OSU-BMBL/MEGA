"""
pyMEGA initialization module.
"""
import logging

from .log import create_logging_handler

LOGGER = logging.getLogger(__name__)
# Set logging level.
LOGGING_DEBUG_OPT = False
LOGGER.addHandler(create_logging_handler(LOGGING_DEBUG_OPT))
LOGGER.setLevel(logging.DEBUG)

from . import _version

__version__ = _version.get_versions()["version"]
