import tempfile
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temp_file(suffix: str = ""):
    """Yield a temp file path and guarantee deletion on exit, even on exception."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield Path(path)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
