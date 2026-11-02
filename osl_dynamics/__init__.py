import logging
from importlib.metadata import PackageNotFoundError, version


def run_pipeline(*args, **kwargs):
    """See osl_dynamics.config_api.pipeline.run_pipeline (imported on use,
    as it pulls in the whole analysis and plotting stack)."""
    from osl_dynamics.config_api.pipeline import run_pipeline as _run_pipeline
    return _run_pipeline(*args, **kwargs)


# Setup the version
try:
    __version__ = version("osl-dynamics")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d:%(funcName)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("osl-dynamics")
logger.debug("Version %s", __version__)
del logger, logging
