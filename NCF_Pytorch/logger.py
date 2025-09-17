import os 
import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(model_name, log_type, config=None, log_dir="logs"):
    
    
    ## timestamp
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    
    run_dir = Path(log_dir) / model_name / f"run_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok = True)
    
    if config is not None:
        
        serializable_config = {
            k:str(v) if isinstance(v,Path) else v for k,v in config.items()
        }
        config_path = run_dir / "config.json"
        
        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=4)
            
    log_path = run_dir / f"{log_type}.log"
    
    logger = logging.getLogger(f"{model_name}_{log_type}_{timestamp}")
    
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # ch = logging.StreamHandler()                   ## IF USE THIS WE CAN SEE THE THINGS IN TERMINAL ALSO
    # ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -  %(message)s")
    
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(fh)
        # logger.addHandler(ch)
        
    return logger, log_path


if __name__ == "__main__":
    print("calling from logger file")