
import os
import numpy as np
import das4whales as dw
import scipy.signal as sp
import matplotlib.image as mpimg
from PIL import Image

def spectral_decomposition(
    tr: np.ndarray,
    fs: float,
    bands: list[list[float]],
    order: int = 5,
    stack: bool = False,
) -> list[np.ndarray] | np.ndarray:
    """
    Perform spectral decomposition into frequency bands.

    Filter DAS strain data into multiple frequency bands. Uses Butterworth bandpass 
    filters with zero-phase forward-backward filtering (sosfiltfilt).

    Parameters
    ----------
    tr : np.ndarray
        2D DAS strain array of shape (n_channels, n_samples).
    fs : float
        Sampling frequency [Hz].
    bands : list of [f_low, f_high]
        Frequency bands [Hz]. Each band must satisfy 0 < f_low < f_high < fs / 2.
    order : int, optional
        Butterworth filter order (default: 5).
    stack : bool, optional
        If True, returns a single 3D array of shape (n_bands, n_channels, n_samples)
        instead of a list of 2D arrays (default: False).

    Returns
    -------
    list of np.ndarray or np.ndarray
        If stack=False: list of n_bands arrays, each of shape (n_channels, n_samples).
        If stack=True:  single array of shape (n_bands, n_channels, n_samples).
    """

    if tr.ndim != 2:
        raise ValueError(f"tr must be a 2D array, got shape {tr.shape}.")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")
    for f_low, f_high in bands:
        if not (0 < f_low < f_high < fs / 2):
            raise ValueError(
                f"Invalid band [{f_low}, {f_high}] Hz for fs={fs} Hz. "
                f"Each band must satisfy 0 < f_low < f_high < fs/2 ({fs/2} Hz)."
            )

    filtered = [
        sp.sosfiltfilt(dw.dsp.butterworth_filter([order, band, 'bp'], fs), tr, axis=-1)
        for band in bands
    ]

    return np.stack(filtered, axis=0) if stack else filtered


def normalize_band(data: np.ndarray, perc: float = 90) -> np.ndarray:
    """
    Normalize a 2D array (or stack of bands) to [0, 1] by percentile-based
    absolute clipping.

    Computes the given percentile of the absolute values and clips the result
    to [0, 1]. When a 3D stack is provided, each band is normalized independently.
    Useful for robust normalization in the presence of outliers.

    Parameters
    ----------
    data : np.ndarray
        Input array. Either:
        - 2D of shape (n_channels, n_samples), or
        - 3D of shape (n_bands, n_channels, n_samples) as returned by
          spectral_decomposition(..., stack=True).
    perc : float, optional
        Percentile used as normalization reference, in the range (0, 100]
        (default: 90).

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1], same shape as input.
    """

    if not (0 < perc <= 100):
        raise ValueError(f"perc must be in (0, 100], got {perc}.")

    if data.ndim == 2:
        return _normalize_single(data, perc)

    elif data.ndim == 3:
        return np.stack([_normalize_single(band, perc) for band in data], axis=0)

    else:
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}.")


def _normalize_single(data: np.ndarray, perc: float) -> np.ndarray:
    """Normalize a single 2D band to [0, 1]. Internal helper for normalize_band."""
    v = np.percentile(np.abs(data), perc)
    if v == 0.0:
        raise ValueError(
            f"Normalization reference is zero at perc={perc}. "
            "The input may be silent or all-zero."
        )
    return np.clip(np.abs(data) / v, 0, 1)


def get_clim(data: np.ndarray, perc: float = 90) -> tuple[float, float]:
    """
    Compute symmetric color limits (vmin, vmax) for a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (n_channels, n_samples).
    perc : float, optional
        Percentile used to define the amplitude limit, in the range (0, 100]
        (default: 90).

    Returns
    -------
    tuple[float, float]
        (vmin, vmax) = (-v, +v), where v is the percentile of the absolute values.
    """
    if not (0 < perc <= 100):
        raise ValueError(f"perc must be in (0, 100], got {perc}.")

    v = float(np.percentile(np.abs(data), perc))

    if v == 0.0:
        raise ValueError(
            f"Color limit is zero at perc={perc}. "
            "The input may be silent or all-zero."
        )

    return -v, v

def crop_data(
    data: np.ndarray,
    dist: np.ndarray,
    time: np.ndarray,
    d_min: float = 0.0,
    d_max: float | None = None,
    t_min: float = 0.0,
    t_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop a 2D DAS array along the distance and time axes.

    Parameters
    ----------
    data : np.ndarray
        2D strain array of shape (n_channels, n_samples).
    dist : np.ndarray
        1D distance axis of shape (n_channels,) [m].
    time : np.ndarray
        1D time axis of shape (n_samples,) [s].
    d_min : float, optional
        Minimum distance [m] (default: 0.0).
    d_max : float or None, optional
        Maximum distance [m]. If None, uses the last value of dist (default: None).
    t_min : float, optional
        Minimum time [s] (default: 0.0).
    t_max : float or None, optional
        Maximum time [s]. If None, uses the last value of time (default: None).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - data_crop : np.ndarray of shape (n_channels_crop, n_samples_crop).
        - dist_crop : np.ndarray of shape (n_channels_crop,).
        - time_crop : np.ndarray of shape (n_samples_crop,).
    """

    if data.ndim != 2:
        raise ValueError(f"data must be a 2D array, got shape {data.shape}.")
    if dist.shape[0] != data.shape[0]:
        raise ValueError(
            f"dist length ({dist.shape[0]}) must match data.shape[0] ({data.shape[0]})."
        )
    if time.shape[0] != data.shape[1]:
        raise ValueError(
            f"time length ({time.shape[0]}) must match data.shape[1] ({data.shape[1]})."
        )

    d_max = dist[-1] if d_max is None else d_max
    t_max = time[-1] if t_max is None else t_max

    if d_min >= d_max:
        raise ValueError(f"d_min ({d_min}) must be less than d_max ({d_max}).")
    if t_min >= t_max:
        raise ValueError(f"t_min ({t_min}) must be less than t_max ({t_max}).")

    d_mask = (dist >= d_min) & (dist <= d_max)
    t_mask = (time >= t_min) & (time <= t_max)

    if not d_mask.any():
        raise ValueError(
            f"Empty crop: no channels found in distance range [{d_min}, {d_max}] m."
        )
    if not t_mask.any():
        raise ValueError(
            f"Empty crop: no samples found in time range [{t_min}, {t_max}] s."
        )

    return data[np.ix_(d_mask, t_mask)], dist[d_mask], time[t_mask]


def rescale_image(
    data: np.ndarray,
    max_size: int,
) -> np.ndarray:
    """
    Rescale a multichannel array (H, W, C) so that its longest axis is
    max_size pixels, preserving the aspect ratio.

    Supports any number of channels C (1, 3, 4, N, ...). For C <= 4 uses
    PIL directly; for C > 4 rescales channel by channel.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (H, W, C) with values in [0, 1].
    max_size : int
        Maximum size in pixels for the longest axis.

    Returns
    -------
    np.ndarray
        Rescaled array of shape (H', W', C) with values in [0, 1].
        Returns the original array unchanged if max(H, W) <= max_size.
    """

    if data.ndim != 3:
        raise ValueError(f"data must be a 3D array (H, W, C), got shape {data.shape}.")
    if max_size <= 0:
        raise ValueError(f"max_size must be a positive integer, got {max_size}.")

    h, w, c = data.shape
    if max(h, w) <= max_size:
        return data

    scale = max_size / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    img_uint8 = (data * 255).astype(np.uint8)

    if c <= 4:
        rescaled = np.array(
            Image.fromarray(img_uint8).resize((new_w, new_h), Image.LANCZOS)
        ).astype(np.float32) / 255.0
    else:
        rescaled = np.stack(
            [
                np.array(
                    Image.fromarray(img_uint8[:, :, i]).resize((new_w, new_h), Image.LANCZOS)
                ).astype(np.float32) / 255.0
                for i in range(c)
            ],
            axis=-1,
        )

    return rescaled

def save_raw_image(
    data: np.ndarray,
    output_folder: str,
    source_filepath: str,
    output_format: str = "png",
    max_size: int | None = None,
) -> str:
    """
    Save a multichannel array as a raw image file with no axes, borders,
    or decorations. Intended for CNN dataset generation.

    Supports PNG (C <= 4) and NPY (any number of channels). If output_format
    is 'png' but the array has more than 4 channels, the format is automatically
    forced to 'npy' with a warning.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (H, W, C) with values in [0, 1].
        C can be any number of bands (1, 3, 4, N, ...).
    output_folder : str
        Destination folder. Must exist prior to calling this function.
    source_filepath : str
        Path to the source .h5 file. Its basename (without extension) is used
        to name the output file.
    output_format : {'png', 'npy'}, optional
        Output format (default: 'png').
        - 'png': saves as PNG image. Only valid for C <= 4.
        - 'npy': saves as raw NumPy array encoded as uint8.
    max_size : int or None, optional
        If provided, rescales the image so that its longest axis is max_size
        pixels before saving, preserving aspect ratio (default: None).

    Returns
    -------
    str
        Absolute path to the saved file.
    """

    if data.ndim != 3:
        raise ValueError(f"data must be a 3D array (H, W, C), got shape {data.shape}.")
    if output_format.lower() not in {"png", "npy"}:
        raise ValueError(f"output_format must be 'png' or 'npy', got '{output_format}'.")
    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"output_folder does not exist: '{output_folder}'.")

    if max_size is not None:
        data = rescale_image(data, max_size)

    base_name = os.path.splitext(os.path.basename(source_filepath))[0]
    n_channels = data.shape[2]

    if output_format.lower() != "npy" and n_channels > 4:
        log(f"WARNING: {n_channels} channels detected — PNG does not support >4 channels. Forcing npy.")
        output_format = "npy"

    if output_format.lower() == "npy":
        out_path = os.path.join(output_folder, base_name + ".npy")
        np.save(out_path, (data * 255).astype(np.uint8))
    else:
        out_path = os.path.join(output_folder, base_name + ".png")
        mpimg.imsave(out_path, data)

    return out_path