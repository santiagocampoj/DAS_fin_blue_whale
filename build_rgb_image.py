import os
import numpy as np
import scipy.signal as sp
import das4whales as dw
from PIL import Image


def h5_to_rgb_png(h5_path, output_png_path, bands=((16, 28), (30, 40), (40, 60)), perc=90, max_size=1024, logger=None):
    if logger:
        logger.info(f"Loading H5: {h5_path}")
    md = dw.data_handle.get_acquisition_parameters(h5_path, interrogator='optasense')
    fs, dx, nx = md['fs'], md['dx'], int(md['nx'])
    tr, _, _, _ = dw.data_handle.load_das_data(h5_path, [0, nx, 1], md)
    if logger:
        logger.info(f"  tr.shape={tr.shape}, fs={fs} Hz, dx={dx} m")

    
    
    # 2. Spectral decomposition: one band-pass filter per RGB channel
    if logger:
        logger.info(f"  decomposing into bands {list(bands)} Hz")
    layers = []
    for b in bands:
        # Using a 5th-order Butterworth band-pass filter for each band
        sos = dw.dsp.butterworth_filter([5, list(b), 'bp'], fs)
        layers.append(sp.sosfiltfilt(sos, tr, axis=-1))



    # 3. Normalize each band by the perc-th percentile of |x_k|
    if logger:
        logger.info(f"  normalizing each band with P{perc}")
    normed = []
    for x in layers:
        v = np.percentile(np.abs(x), perc)
        if v == 0:
            normed.append(np.zeros_like(x, dtype=np.float32))
        else:
            normed.append(np.clip(np.abs(x) / v, 0, 1).astype(np.float32))
    rgb_float = np.stack(normed, axis=-1)   
    # shape (nx, ns, 3), values in [0, 1]



    # 4. Rescale so the longest axis = max_size pixels
    h, w, _ = rgb_float.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))
        img_u8 = (rgb_float * 255).astype(np.uint8)
        rgb_u8 = np.array(Image.fromarray(img_u8).resize((new_w, new_h), Image.LANCZOS))
        if logger:
            logger.info(f"  rescaled from ({h}, {w}) → ({new_h}, {new_w})")
    else:
        rgb_u8 = (rgb_float * 255).astype(np.uint8)



    # 5. Save as 8-bit RGB PNG
    Image.fromarray(rgb_u8, mode='RGB').save(output_png_path)
    if logger:
        logger.info(f"Saved RGB PNG: {output_png_path}")
    return rgb_u8
