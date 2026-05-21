import numpy as np
import das4whales as dw


def fix_bbox_with_h5_metadata(df_bbox, h5_path, output_csv_path=None, logger=None):
    if logger:
        logger.info(f"Fixing bbox shape using H5 metadata: {h5_path}")


    metadata = dw.data_handle.get_acquisition_parameters(h5_path, interrogator='optasense')
    fs = metadata['fs']
    dx = metadata['dx']
    nx = int(metadata['nx'])
    ns = int(metadata['ns'])
    if logger:
        logger.info(f"H5 grid: nx={nx}, ns={ns}, fs={fs} Hz, dx={dx} m")
    # full time and distance axes
    time_s = np.arange(ns) / fs # length ns (e.g. 12000)
    dist_m = np.arange(nx) * dx # length nx (e.g. 32600)



    df = df_bbox.copy()
    df['ti0'] = np.searchsorted(time_s, df['t0'].values).astype(int)
    df['ti1'] = np.searchsorted(time_s, df['t1'].values).astype(int)
    df['di0'] = np.searchsorted(dist_m, df['d0'].values).astype(int)
    df['di1'] = np.searchsorted(dist_m, df['d1'].values).astype(int)
    df['nx'] = nx
    df['nt'] = ns

    

    if output_csv_path is not None:
        df.to_csv(output_csv_path, index=False)
        if logger:
            logger.info(f"Saved fixed bbox CSV to: {output_csv_path}")


    return df