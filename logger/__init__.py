# logger/__init__.py

import logging
from datetime import datetime
import os

class CustomLogger:
    """Custom logger class to handle logging with a timestamped log file."""
    
    def __init__(self):
        # Define the directory for log files
        self.log_dir = "logs"
        
        # Create LOG_DIR if it does not exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a timestamp for the log file name
        current_time_stamp = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        # Define the log file name with timestamp
        self.log_file_name = f"log_{current_time_stamp}.log"
        
        # Define the complete path for the log file
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name)

        # Set up basic logging configuration
        logging.basicConfig(
            filename=self.log_file_path,
            filemode="w",  # Use "a" for appending to existing logs instead of overwriting
            format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)  # Create a logger instance

    def get_logger(self):
        """Return the configured logger instance."""
        return self.logger
