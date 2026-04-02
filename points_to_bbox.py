import os
import numpy as np
import pandas as pd
 


def points_to_bbox(csv_path, output_path=None, group_col='call_id',
                     time_s=None, dist_m=None, start_datetime_utc=None, logger=None):
    """
    Convert point annotations to bounding boxes in events CSV format.
 
    Parameters
    ----------
    csv_path : str
        Path to the annotation CSV file.
    output_path : str, optional
        Directory where the output CSV will be saved.
        Defaults to the same directory as the input CSV.
    group_col : str
        Column name to group by (default: 'call_id').
    time_s : np.ndarray, optional
        Time axis in seconds. Required to compute ti0, ti1, nt and start_datetime_utc.
    dist_m : np.ndarray, optional
        Distance axis in metres. Required to compute di0, di1, nx.
    start_datetime_utc : datetime or pd.Timestamp, optional
        UTC start time of the file. Required to compute start_datetime_utc field.
 
    Returns
    -------
    pd.DataFrame
        One row per group with columns:
        [ID, t0, t1, d0, d1, ti0, ti1, di0, di1, nx, nt, downsample, start_datetime_utc, comment]
    """
    df = pd.read_csv(csv_path)
    records = []
 
    for group_id, grp in df.groupby(group_col):
 
        t0 = grp['time'].min()
        t1 = grp['time'].max()
        d0 = grp['dist'].min()
        d1 = grp['dist'].max()
 
        if time_s is not None:
            ti0 = int(np.searchsorted(time_s, t0))
            ti1 = int(np.searchsorted(time_s, t1))
            nt  = len(time_s)
        else:
            ti0 = ti1 = nt = None
 
        if dist_m is not None:
            di0 = int(np.searchsorted(dist_m, d0))
            di1 = int(np.searchsorted(dist_m, d1))
            nx  = len(dist_m)
        else:
            di0 = di1 = nx = None
 
        if start_datetime_utc is not None:
            bbox_timestamp = (pd.Timestamp(start_datetime_utc) + pd.Timedelta(seconds=t0)).isoformat()
        else:
            bbox_timestamp = None
 
        records.append({
            'ID':                 grp['call_type'].iloc[0],  # class id = call type
            't0':                 t0,
            't1':                 t1,
            'd0':                 d0,
            'd1':                 d1,
            'ti0':                ti0,
            'ti1':                ti1,
            'di0':                di0,
            'di1':                di1,
            'nx':                 nx,
            'nt':                 nt,
            'downsample':         None,
            'start_datetime_utc': bbox_timestamp,
            'comment':            None,
        })
 
    df_bbox = pd.DataFrame(records)
 
    base_name = os.path.splitext(os.path.basename(csv_path))[0] + '_bbox.csv'
    out_dir   = output_path if output_path else os.path.dirname(csv_path) or '.'
    out_file  = os.path.join(out_dir, base_name)
    df_bbox.to_csv(out_file, index=False)
    print(f'Saved → {out_file}')
 
    return df_bbox