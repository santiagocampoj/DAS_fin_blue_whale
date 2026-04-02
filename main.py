from config import DAS_FOLDER
from logging_config import setup_logging
import os
import pandas as pd



def main() -> None:
    logger =setup_logging()
    logger.info("Starting the application.")

    
    #kmddkmdwd
    current_directory = os.getcwd()
    logger.info(f"Current working directory: {current_directory}")


    #with walk
    for dirpath, dirnames, filenames in os.walk(DAS_FOLDER):
        logger.info(f"Directory: {dirpath}")
        logger.info(f"Subdirectories: {dirnames}")
        logger.info(f"Files: {filenames}")

        if "annotated_calls_north_2021-11-04_02_00_02.csv" in filenames:
            file_path = os.path.join(dirpath, "annotated_calls_north_2021-11-04_02_00_02.csv")
            logger.info(f"Found the file at: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns}")
            logger.info(f"Dataframe {df.head()}")
            break





if __name__ == "__main__":
    main()