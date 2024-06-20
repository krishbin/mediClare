import logging
from datetime import datetime
from . import yaml_parser
import os
import inspect

class Logger:
    def __init__(self, log_dir: str = "logs", **kwargs):
        self.caller = self._get_calling_module()
        self.caller = kwargs["caller"] if "caller" in kwargs else self.caller
        self.log_dir = log_dir
        self.run_mode = kwargs["run_mode"] if "run_mode" in kwargs else "debug"
        self.logger = logging.getLogger(self.caller)
        if self.run_mode == "debug":
            self.logger.setLevel(logging.DEBUG)
            self.file_handler = logging.FileHandler(os.path.join(log_dir,f"{datetime.today().strftime('%Y-%m-%d')}__{self.caller}.log"))
            self.file_handler.setLevel(logging.DEBUG)

    def info(self, message):
        if self.run_mode == "debug":
            self.logger.addHandler(self.file_handler)
        self.logger.info(datetime.today().strftime('%Y-%m-%d %H:%M:%S') + " - " + message)
        if self.run_mode == "debug":
            self.logger.removeHandler(self.file_handler)

    def close(self):
        if self.run_mode == "debug":
            self.file_handler.close()
        logging.shutdown()
    
    def _get_calling_module(self):
        frame = inspect.stack()[2]
        calling_module = inspect.getmodule(frame[0]).__name__
        return calling_module