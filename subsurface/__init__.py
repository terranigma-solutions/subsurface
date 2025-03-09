import subsurface.modules.reader
import subsurface.api.interfaces
import subsurface.modules.writer
from . import core
from .modules import visualization
from subsurface.core.structs import *
from datetime import datetime
import dotenv

dotenv.load_dotenv()

try:
    from subsurface import visualization
except ImportError:
    pass
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # For Python <3.8, fallback
    from importlib_metadata import version, PackageNotFoundError
    
# Version.
try:
    __version__ = version("subsurface")  # Use package name
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. subsurface should be
    # installed properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')

if __name__ == '__main__':
    pass
