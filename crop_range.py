def compute_crop_channels(df_bbox, nx, margin_frac=1, logger=None):
    if df_bbox.empty:
        return 0, int(nx)

    # min and max channels
    di_lo = int(df_bbox['di0'].min())
    di_hi = int(df_bbox['di1'].max())

    # margin
    margin = int(round(margin_frac * (di_hi - di_lo)))


    ch_start = max(0, di_lo - margin)
    ch_end   = min(int(nx), di_hi + margin)

    if logger:
        logger.info(f"  crop: channels box [{di_lo}, {di_hi}] "
                    f"(+{margin} magring) -> crop [{ch_start}, {ch_end})")
    return ch_start, ch_end