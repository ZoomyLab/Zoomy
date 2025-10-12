
import os
import sys
from loguru import logger

# Remove any default handlers
logger.remove()
# Check the ZoomyLog setting
zoomy_log_mode = os.getenv("ZoomyLog", "Default")

zoomy_log_level = os.getenv("ZoomyLogLevel", "INFO")

main_dir = os.getenv("ZOOMY_DIR", os.getcwd())

if zoomy_log_mode == "Default":
    logger.add(sys.stderr, level=zoomy_log_level) 
else:
    logger.add(os.path.join(main_dir, "logs/log.log"), rotation="1 MB", retention="10 days", compression="zip")
