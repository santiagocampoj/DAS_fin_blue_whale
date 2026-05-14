import numpy as np
import pandas as pd
import das4whales as dw


def fix_bbox_with_h5_metadata(df_bbox, h5_path, output_csv_path=None, logger=None):
    """
    Overwrite nx, nt, ti0, ti1, di0, di1 in a bbox DataFrame using the real
    DAS grid from the H5 metadata (instead of the sparse axes derived from
    the annotation CSV).

    Parameters
    ----------
    df_bbox : pd.DataFrame
        The bbox table as returned by points_to_bbox().
    h5_path : str
        Path to the matching H5 file (used for nx, ns, fs, dx).
    output_csv_path : str, optional
        If given, the corrected DataFrame is written here (overwriting any
        existing file). Otherwise nothing is written and the caller is
        responsible for saving.
    logger : logging.Logger, optional
    """
    if logger:
        logger.info(f"Fixing bbox shape using H5 metadata: {h5_path}")

    # 1. Read DAS grid info from the H5 file
    metadata = dw.data_handle.get_acquisition_parameters(h5_path, interrogator='optasense')
    fs = metadata['fs']
    dx = metadata['dx']
    nx = int(metadata['nx'])
    ns = int(metadata['ns'])
    if logger:
        logger.info(f"H5 grid: nx={nx}, ns={ns}, fs={fs} Hz, dx={dx} m")

    # 2. Build the FULL time and distance axes
    time_s = np.arange(ns) / fs       # length ns (e.g. 12000)
    dist_m = np.arange(nx) * dx       # length nx (e.g. 32600)

    # 3. Work on a copy so we don't mutate the caller's DataFrame in place
    df = df_bbox.copy()

    # 4. Recompute ti, di against the real axes
    df['ti0'] = np.searchsorted(time_s, df['t0'].values).astype(int)
    df['ti1'] = np.searchsorted(time_s, df['t1'].values).astype(int)
    df['di0'] = np.searchsorted(dist_m, df['d0'].values).astype(int)
    df['di1'] = np.searchsorted(dist_m, df['d1'].values).astype(int)

    # 5. Overwrite nx and nt with the H5 values
    df['nx'] = nx
    df['nt'] = ns

    # 6. Save if requested
    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
        if logger:
            logger.info(f"Saved fixed bbox CSV to: {output_csv_path}")

    return df