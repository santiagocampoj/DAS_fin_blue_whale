def compute_crop_channels(df_bbox, nx, dx, margin_m=300.0, logger=None):
    if df_bbox.empty:
        return 0, int(nx)

    di_lo = int(df_bbox['di0'].min())
    di_hi = int(df_bbox['di1'].max())
    margin = int(round(margin_m / dx))
    ch_start = max(0, di_lo - margin)
    ch_end   = min(int(nx), di_hi + margin)

    if logger:
        logger.info(f"  crop: box channels [{di_lo}, {di_hi}] "
                    f"(+{margin} ch ≈ {margin_m:.0f} m margin) -> crop [{ch_start}, {ch_end})")
    return ch_start, ch_end