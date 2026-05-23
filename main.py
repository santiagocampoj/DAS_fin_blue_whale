from config import DAS_FOLDER, DAS_D4W_FOLDER_BACKUP, SUFFIX, BBOX_SUFIX, DAS_H5_FOLDER_BACKUP
from logging_config import setup_logging
from points_to_bbox import points_to_bbox
from rename_csv_h5 import rename_annot_h5
from fix_bbox_shape import fix_bbox_with_h5_metadata
from plot_bbox import plot_bbox_overlay
from build_rgb_image import h5_to_rgb_png
from crop_range import compute_crop_channels
from yolo_export import draw_boxes_on_rgb


import os
from tqdm import tqdm
import das4whales as dw




def main() -> None:
    logger = setup_logging()
    logger.info("Starting the application.")
    logger.info(f"Current working directory: {os.getcwd()}")

    if DAS_D4W_FOLDER_BACKUP is None or not os.path.exists(DAS_D4W_FOLDER_BACKUP):
        logger.error(f"DAS_D4W_FOLDER_BACKUP is not set or does not exist: {DAS_D4W_FOLDER_BACKUP}")
        return

    if DAS_H5_FOLDER_BACKUP is None or not os.path.exists(DAS_H5_FOLDER_BACKUP):
        logger.error(f"DAS_H5_FOLDER_BACKUP is not set or does not exist: {DAS_H5_FOLDER_BACKUP}")
        return



    h5_files = [f for f in os.listdir(DAS_H5_FOLDER_BACKUP) if f.endswith('.h5')]
    logger.info(f"Found {len(h5_files)} .h5 files in {DAS_H5_FOLDER_BACKUP}: {h5_files}")
    if not h5_files:
        logger.error("No .h5 files found.")
        return



    # Single-file testing phase: just use the first H5 we find.
    # TODO: when there are multiple H5 files, match each CSV to its H5 by timestamp.
    h5_path = os.path.join(DAS_H5_FOLDER_BACKUP, h5_files[0])
    logger.info(f"Using H5 file: {h5_path}")



    logger.info(f"Walking through directory: {DAS_D4W_FOLDER_BACKUP}")
    for dirpath, dirnames, filenames in os.walk(DAS_D4W_FOLDER_BACKUP):
        logger.info(f"Directory: {dirpath}")
        #adding the tqdm progress bar to the file processing loop
        for filename in tqdm(filenames, desc=f"Processing files", unit="file"):
            if filename.endswith('.csv') and 'rename' not in dirpath:
                csv_path = os.path.join(dirpath, filename)
                logger.info(f"Processing file: {csv_path}")



                # Step 1: rename + prep output paths
                try:
                    logger.info(f"Renaming {csv_path} to match H5 filename")
                    csv_path_renamed = rename_annot_h5(csv_path, logger=logger)
                    out_dir = os.path.dirname(csv_path_renamed)
                    out_name = os.path.splitext(os.path.basename(csv_path_renamed))[0] + BBOX_SUFIX
                except Exception as e:
                    logger.error(f"Error renaming {csv_path}: {e}")
                    continue



                # Step 2: convert points to bboxes (returns df, also writes CSV)
                try:
                    logger.info(f"Converting points to bbox for {csv_path_renamed}")
                    df_bbox = points_to_bbox(csv_path, output_path=out_dir, output_name=out_name, logger=logger)
                except Exception as e:
                    logger.error(f"Error processing {csv_path}: {e}")
                    continue



                # Step 3: fix nx/nt/ti/di in memory and overwrite the bbox CSV
                try:
                    logger.info(f"Fixing bbox shape for {csv_path} using H5 metadata")
                    bbox_csv_path = os.path.join(out_dir, out_name)
                    df_bbox = fix_bbox_with_h5_metadata(
                        df_bbox, h5_path,
                        output_csv_path=bbox_csv_path,
                        logger=logger,
                    )
                except Exception as e:
                    logger.error(f"Error fixing bbox shape for {bbox_csv_path}: {e}")
                    continue



                # Step 3.5: calcular el rango de recorte (canales) desde las cajas, una vez
                try:
                    logger.info("Computing crop range from bbox")
                    nx_full = int(df_bbox['nx'].iloc[0])
                    md = dw.data_handle.get_acquisition_parameters(h5_path, interrogator='optasense')
                    dx = md['dx']
                    ch_start, ch_end = compute_crop_channels(df_bbox, nx_full, dx, logger=logger)
                except Exception as e:
                    logger.error(f"Error computing crop range: {e}")
                    continue



                # Step 4: plot the bboxes overlaid on the H5 envelope (PNG next to CSV)
                try:
                    logger.info(f"Plotting bbox overlay for {bbox_csv_path}")
                    plot_path = os.path.splitext(bbox_csv_path)[0] + '.png'
                    plot_bbox_overlay(h5_path, bbox_csv_path, save_path=plot_path,
                                      ch_start=ch_start, ch_end=ch_end, logger=logger)
                    # plot_bbox_overlay(h5_path, bbox_csv_path, save_path=plot_path, logger=logger)
                except Exception as e:
                    logger.error(f"Error plotting {bbox_csv_path}: {e}")
                


                # Step 5: build the multispectral RGB PNG from the H5 (for YOLO training)
                try:
                    logger.info(f"Building RGB image for {h5_path}")
                    h5_basename = os.path.splitext(os.path.basename(h5_path))[0]
                    rgb_path = os.path.join(out_dir, h5_basename + '_rgb.png')
                    # h5_to_rgb_png(h5_path, rgb_path, logger=logger)
                    h5_to_rgb_png(h5_path, rgb_path, ch_start=ch_start, ch_end=ch_end, logger=logger)
                except Exception as e:
                    logger.error(f"Error building RGB image for {h5_path}: {e}")



                try:
                    # Step 6: drawing the bboxes on the RGB image (for YOLO training)
                    nt = int(df_bbox['nt'].iloc[0])
                    check_path = os.path.join(out_dir, h5_basename + '_rgb_check.png')
                    draw_boxes_on_rgb(rgb_path, df_bbox, ch_start, ch_end, nt, check_path, logger=logger)
                except Exception as e:
                    logger.error(f"Error drawing boxes on RGB image: {e}")



                exit()



if __name__ == "__main__":
    main()