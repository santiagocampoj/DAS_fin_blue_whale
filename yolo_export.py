import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from config import COLOR




def yolo_coords(row, ch_start, ch_end, nt):
    H = ch_end - ch_start
    di0 = max(ch_start, int(row['di0']))
    di1 = min(ch_end,   int(row['di1']))
    ti0, ti1 = float(row['ti0']), float(row['ti1'])


    # channel inside the crop
    c0, c1 = di0 - ch_start, di1 - ch_start

    x_center = (ti0 + ti1) / 2 / nt
    # -1 is the center
    y_center = 1 - ((c0 + c1) / 2) / H
    w = (ti1 - ti0) / nt
    h = (c1 - c0) / H
    return x_center, y_center, w, h



def draw_boxes_on_rgb(rgb_path, df_bbox, ch_start, ch_end, nt, save_path=None, logger=None):
    img = Image.open(rgb_path)
    # size
    W, Hpx = img.size

    fig, ax = plt.subplots(figsize=(5, 12))
    ax.imshow(img)
    for _, row in df_bbox.iterrows():
        cx, cy, w, h = yolo_coords(row, ch_start, ch_end, nt)
        # normalize
        x_px = (cx - w / 2) * W
        # img y=0 is top, bbox y=0 is bottom, so invert y and shift by h/2
        y_px = (cy - h / 2) * Hpx
        color = COLOR.get(row['ID'], 'white')
        ax.add_patch(patches.Rectangle((x_px, y_px), w * W, h * Hpx,
                                       linewidth=1.2, edgecolor=color, facecolor='none'))
        ax.text(x_px, y_px, str(row['ID']), color=color, fontsize=7, va='bottom')
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if logger:
            logger.info(f"Saved YOLO-check overlay: {save_path}")
    else:
        plt.show()