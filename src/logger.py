import logging
import os
from datetime import datetime


# create log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# define log Path
logs_path = os.path.join(os.getcwd(),'logs',LOG_FILE)

# create log dir
os.makedirs(logs_path,exist_ok = True)



LOG_FILE_PATH =os.path.join(logs_path,LOG_FILE)


# log
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)