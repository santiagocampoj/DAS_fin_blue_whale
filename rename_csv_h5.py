import os
import re
from config import SUFFIX




def rename_annot_h5(csv_path: str, logger=None) -> str:
    pattern = re.compile(r'^annotated_calls_(north|south)_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})\.csv$', re.IGNORECASE)

    filename = os.path.basename(csv_path)
    match = pattern.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    ################
    cable = match.group(1).capitalize() # north -> North
    date = match.group(2) # 2021-11-04
    hh, mm, ss = match.group(3), match.group(4), match.group(5)
    datetime_tag = f"{date}T{hh}{mm}{ss}Z" # 2021-11-04T020002Z

    new_name = f"{cable}-{SUFFIX}_{datetime_tag}.csv"
    logger.info(f"Renaming {filename} to {new_name}")


    #####################
    folder = os.path.dirname(csv_path)
    renamed_folder = os.path.join(folder, "renamed")
    os.makedirs(renamed_folder, exist_ok=True)
    new_path = os.path.join(renamed_folder, new_name)


    if logger:
        logger.info(f"Ensured renamed folder exists: {renamed_folder}")
        logger.info(f"New path will be: {new_path}")


    return new_path