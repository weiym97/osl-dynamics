import logging

import numpy as np

_logger = logging.getLogger("VRAD")


def override_dict_defaults(default_dict: dict, override_dict: dict = None) -> dict:
    if override_dict is None:
        override_dict = {}

    return {**default_dict, **override_dict}


def listify(obj: object):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def time_axis_first(input_array: np.ndarray) -> np.ndarray:
    if input_array.ndim != 2:
        _logger.info("Non-2D array not transposed.")
        return input_array
    if input_array.shape[1] > input_array.shape[0]:
        input_array = np.transpose(input_array)
        _logger.warning(
            "More channels than time points detected. Time series has been transposed."
        )
    return input_array


class LoggingContext:
    def __init__(self, logger, suppress_level="warning", handler=None, close=True):
        self.logger = logging.getLogger(logger)
        if isinstance(suppress_level, str):
            suppress_level = logging.getLevelName(suppress_level.upper())
        self.level = suppress_level + 10
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
