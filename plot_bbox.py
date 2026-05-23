import os
import numpy as np
import pandas as pd
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import das4whales as dw
from config import COLOR



def plot_bbox_overlay(h5_path, bbox_csv_path, save_path=None,
                      ch_start=None, ch_end=None, show_peak=None, logger=None):
    metadata = dw.data_handle.get_acquisition_parameters(h5_path, interrogator='optasense')
    fs, dx, nx = metadata['fs'], metadata['dx'], int(metadata['nx'])

    if ch_start is None: ch_start = 0
    if ch_end is None: ch_end = nx
    selected_channels = [ch_start, ch_end, 1]    

    # selected_channels = [0, int(nx*dx // dx), 1]
    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(
        h5_path, selected_channels, metadata
    )


    # band-pass + f-k filter
    sos_bpfilter = dw.dsp.butterworth_filter([5, [15, 25], 'bp'], fs)
    trf = sp.sosfiltfilt(sos_bpfilter, tr, axis=1)

    fk_params_s = {'c_min': 1400., 'c_max': 3500., 'fmin': 10., 'fmax': 30.}
    fk_filter = dw.dsp.hybrid_ninf_gs_filter_design(
        (tr.shape[0], tr.shape[1]), selected_channels, dx, fs, fk_params_s,
        display_filter=False,
    )
    trf_fk = dw.dsp.fk_filter_sparsefilt(trf, fk_filter, tapering=False)



    # bbox csv
    df_bbox = pd.read_csv(bbox_csv_path)
    if logger:
        logger.info(f"Loaded {len(df_bbox)} bboxes from {bbox_csv_path}")
        logger.info(df_bbox.head())



    # downsample envelope
    stride_t, stride_x = 4, 4
    envelope = np.abs(sp.hilbert(trf_fk, axis=1))
    envelope_ds = envelope[::stride_x, ::stride_t]
    time_ds = time[::stride_t]
    dist_ds = dist[::stride_x]




    # plots
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        envelope_ds,
        aspect='auto', origin='lower',
        extent=[time_ds[0], time_ds[-1], dist_ds[0]/1e3, dist_ds[-1]/1e3],
        vmin=0, vmax=0.4e-9, cmap='RdYlBu_r',
    )
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance (km)')
    ax.set_title(str(fileBeginTimeUTC))
    plt.colorbar(im, ax=ax, label='Strain Envelope (x1e-9)')



    # Overlay bounding boxes
    for _, row in df_bbox.iterrows():
        t0, t1 = row['t0'], row['t1']
        d0, d1 = row['d0'] / 1e3, row['d1'] / 1e3
        label = row['ID']
        color = COLOR.get(label, 'white')
        rect = patches.Rectangle(
            (t0, d0), t1 - t0, d1 - d0,
            linewidth=1.5, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)
        ax.text(t0, d1, str(label), color=color, fontsize=7, va='bottom')

        if show_peak:
            _, d_peak = call_distance(envelope, dist, ch_start, row)
            # ax.hlines(d_centroid / 1e3, t0, t1, color=color, lw=1.0, ls='--') # centroide
            ax.hlines(d_peak     / 1e3, t0, t1, color=color, lw=1.0, ls=':') # pico
            if logger:
                # logger.info(f"  {label}: centroid={d_centroid/1e3:.1f} km, peak={d_peak/1e3:.1f} km")
                logger.info(f"  {label}: peak={d_peak/1e3:.1f} km")



    # actual plotting
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        if logger:
            logger.info(f"Saved figure: {save_path}")
    else:
        plt.show()

    logger.info("Finished plotting bbox overlay.")
    exit()



def call_distance(envelope, dist, ch_start, row):
    n = envelope.shape[0]
    # channel; envelope row
    r0 = max(0, int(row['di0']) - ch_start)
    r1 = min(n, int(row['di1']) - ch_start)
    c0, c1 = int(row['ti0']), int(row['ti1'])


    # sum in time, energy per channel
    profile = envelope[r0:r1, c0:c1].sum(axis=1)
    d = dist[r0:r1]
    if profile.sum() == 0:
        return float(d.mean()), float(d.mean())


    # center or peak
    centroid = float(np.sum(d * profile) / profile.sum())
    peak = float(d[int(np.argmax(profile))])
    return centroid, peak