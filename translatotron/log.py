import logging
from pathlib import Path


def setup_logging(log_file="logs/app.log", level="INFO"):

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[console, file_handler],
        force=True  # Override existing setup
    )
    

def get_logger(name):
    return logging.getLogger(name)
