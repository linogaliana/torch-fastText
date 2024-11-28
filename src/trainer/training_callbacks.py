"""Training Callbacks for training monitoring integrated in `pythae` (inspired from
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

import logging


class TrainingLogger:
    """
    A lightweight logging class to replace TensorBoard
    """

    def __init__(self, log_file=None, console_log=True):
        """
        Initialize logger

        Args:
            log_file (str, optional): Path to log file. Defaults to None.
            console_log (bool): Whether to print logs to console
        """
        self.log_file = log_file
        self.console_log = console_log

        # Configure logging
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(message)s",
                handlers=[logging.StreamHandler()],
            )
            self.logger = logging.getLogger(__name__)

    def log(self, message, level="info"):
        """
        Log a message

        Args:
            message (str): Message to log
            level (str): Logging level
        """
        if self.log_file:
            if level == "info":
                self.logger.info(message)
            elif level == "error":
                self.logger.error(message)

        if self.console_log:
            print(message)
