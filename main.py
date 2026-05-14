from config import DAS_FOLDER, DAS_D4W_FOLDER_BACKUP, SUFFIX, BBOX_SUFIX, DAS_H5_FOLDER_BACKUP
from logging_config import setup_logging
from points_to_bbox import points_to_bbox
from rename_csv_h5 import rename_annot_h5
import os




def main() -> None:
    logger = setup_logging()
    logger.info("Starting the application.")


    logger.info("")
    current_directory = os.getcwd()
    logger.info(f"Current working directory: {current_directory}")

    if DAS_D4W_FOLDER_BACKUP is None or not os.path.exists(DAS_D4W_FOLDER_BACKUP):
        logger.error(f"DAS_D4W_FOLDER_BACKUP is not set or does not exist: {DAS_D4W_FOLDER_BACKUP}")
        return
    
    if DAS_H5_FOLDER_BACKUP is None or not os.path.exists(DAS_H5_FOLDER_BACKUP):
        logger.error(f"DAS_H5_FOLDER_BACKUP is not set or does not exist: {DAS_H5_FOLDER_BACKUP}")
        return
    
    h5_files = [f for f in os.listdir(DAS_H5_FOLDER_BACKUP) if f.endswith('.h5')]
    logger.info(f"Found {len(h5_files)} .h5 files in {DAS_H5_FOLDER_BACKUP}:")



    logger.info("")
    logger.info(f"Walking through directory: {DAS_D4W_FOLDER_BACKUP}")
    for dirpath, dirnames, filenames in os.walk(DAS_D4W_FOLDER_BACKUP):
        logger.info(f"Directory: {dirpath}")
        logger.info(f"Subdirectories: {dirnames}")
        logger.info(f"Files: {filenames}")



        for filename in filenames:
            if filename.endswith('.csv') and 'rename' not in dirpath:
                csv_path = os.path.join(dirpath, filename)
                logger.info("")
                logger.info(f"Processing file: {csv_path}")


                # Step 1: new name and output dir
                try:
                    csv_path_renamed = rename_annot_h5(csv_path, logger=logger)
                    out_dir = os.path.dirname(csv_path_renamed)
                    out_name = os.path.splitext(os.path.basename(csv_path_renamed))[0] + BBOX_SUFIX
                except Exception as e:
                    logger.error(f"Error renaming {csv_path}: {e}")
                    continue



                # Step 2: process original, save with new name into renamed/
                try:
                    points_to_bbox(csv_path, output_path=out_dir, output_name=out_name, logger=logger)
                except Exception as e:
                    logger.error(f"Error processing {csv_path}: {e}")




if __name__ == "__main__":
    main()