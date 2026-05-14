
# YOLO4DAS annotation tool (versión 2.0)
# 
# A simple tool for annotating events from the YOLO4DAS project. The tool allows you to 
# annotate events in bounding-box format and export them as a CSV file.
# 
# As a result of the annotation:
# 
# A CSV file is exported with the following columns: ID (ID or event class), t0 (start time 
# of the event), t1 (end time of the event), d0 (start distance of the event), d1 (end distance 
# of the event), ti0 (start index of the event on the time axis), ti1 (final index on the time 
# axis), di0 (initial index on the distance axis), di1 (final index on the distance axis).
#
# A NPZ file is exported with the following variables: An NPZ file is exported with the 
# following variables: ['tr'] 2D matrix with dimensions (distance, time), ['dist_m'] with the 
# distance array, ['time_s'] with the time array, ['fs_hz'] with the sampling frequency in Hz.

import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from scipy import signal

from PyQt5 import QtWidgets, QtCore
from matplotlib.widgets import Button, RectangleSelector, Slider

# ----------------------------------------------------------------------------
# INTERFACE CONFIGURATION
# ----------------------------------------------------------------------------
# We use the Qt5Agg backend to enable high-performance rendering and 
# compatibility with PyQt5 system dialogs (file explorers, input boxes).

import matplotlib; matplotlib.use("Qt5Agg")

def annotate(
    tr: np.ndarray, 
    time_s: np.ndarray, 
    dist_m: np.ndarray, 
    fs_hz: Union[float, int],
    start_datetime_utc: datetime.datetime,
    filename: Optional[str] = None,
    save_dir: Optional[str] = None,
    save_directly: Optional[bool] = False,
    vmin: Union[float, int] = None, 
    vmax: Union[float, int] = None, 
    d_min_lim: Union[float, int] = None,
    d_max_lim: Union[float, int] = None,
    ymin: Union[float, int] = None,
    ymax: Union[float, int] = None,
    downsample_step: int = None,
    annotations_output_format: Optional[str] = 'csv',
    data_output_format: Optional[str] = None
) -> None:
    """
    YOLO4DAS annotation tool

    A simple tool for annotating events from the YOLO4DAS project. The tool allows
    to annotate events as bounding-box and export them as a CSV file. Additionally you
    can export the DAS processed files.
    
    Parameters:
    -----------
    tr : np.ndarray
        A 2D NumPy array of floats containing the strain/acoustic data.
        Shape must be (n_distances, n_times), where:
        - Rows (dim 0): Spatial dimension (fiber distance).
        - Cols (dim 1): Temporal dimension (time).
        
    time_s : np.ndarray
        A 1D NumPy array of floats representing the time axis in seconds [s].
        Length must be equal to tr.shape[1].
        
    dist_m : np.ndarray
        A 1D NumPy array of floats representing the spatial axis in meters [m].
        Length must be equal to tr.shape[0].
        
    fs_hz : float or int
        The sampling frequency of the data in Hertz [Hz]. Used for 
        signal processing calculations and metadata export.
    
    start_datetime_utc : datetime.datetime
        The start timestamp UTC of the file in datetime format.
        
    filename : str, optional
        The path or name of the source file. Used to generate default 
        filenames during export.
    
    save_dir : str, optional
        The path where the results will be exported.
    
    vmin, vmax : float or int, optional
        Initial dynamic range for the colormap. If None, uses array min/max.
        These values control the amplitude visualization range.

    d_min_lim, d_max_lim : float or int, optional
        Initial distance limits in meters to visualize water entry/exit points.
        White horizontal lines will be drawn at these distances.
        If the provided values are outside the actual distance range (dist_m),
        they will be clamped to the valid range or set to None.

    ymin, ymax : float or int, optional
        Initial y-axis (distance) limits in meters to display.
        If None, uses the full distance range (dist_m[0], dist_m[-1]).
        These values control the visible vertical range of the plot.

    annotations_output_format : str, optional
        The format used to save the annotations. Default: "csv".
        Currently only CSV format is supported.
    
    data_output_format : str, optional
        The format used to save the processed data. Default: None. 
        Supported formats: 'npz' (NumPy compressed format).

    Keyboard Shortcuts:
    -------------------
    - 'l' : Trigger LOAD button
    - 's' : Activate SELECT mode for annotation
    - 'SPACE' : Toggle crosshair on/off (when SELECT mode is active)
    - 'e' : Export annotations
    - 'Ctrl++' : Zoom in (25% reduction)
    - 'Ctrl+-' : Zoom out (25% expansion)
    - '5' : Set 5 second time window
    - 'Ctrl+1' : Set 10 second time window
    - 'Ctrl+2' : Set 20 second time window
    - 'Ctrl+6' : Set 60 second time window
    - 'f' : Full time view
    - 'Left Arrow' : Navigate backward
    - 'Right Arrow' : Navigate forward

    Mouse Interactions:
    -------------------
    - Left-click + drag : Draw annotation bounding box (when SELECT mode is active)
    - Right-click : Context menu to edit ID/comment, remove annotation, show spectrogram, or spectral analysis
    - Mouse move (with SPACE active): Crosshair lines follow cursor for precise annotation

    """

    # ------------------------------------------------------------------------
    # INITIAL PARAMETERS SETUP
    # ------------------------------------------------------------------------
    
    # Initial Contrast: DAS data often has very low amplitudes (e.g., 1e-10).
    # Set default values from the data if not provided by the user.
    v_min = vmin if vmin is not None else np.min(tr)
    v_max = vmax if vmax is not None else np.max(tr)
    
    # Initial Distance Limits: Adjust values if they fall outside the valid range.
    # This prevents visualization artifacts when limits exceed actual data boundaries.
    if d_min_lim is not None and d_min_lim < np.min(dist_m):
        d_min_lim = None
    if d_max_lim is not None and d_max_lim > np.max(dist_m):
        d_max_lim = None
    
    # Set the actual distance limits, defaulting to data extremes if not specified
    d_min = d_min_lim if d_min_lim is not None else dist_m[0]
    d_max = d_max_lim if d_max_lim is not None else dist_m[-1]

    # Initial Y-axis Limits (distance range)
    if ymin is not None:
        ymin_val = max(ymin, dist_m[0])
    else:
        ymin_val = dist_m[0]
    if ymax is not None:
        ymax_val = min(ymax, dist_m[-1])
    else:
        ymax_val = dist_m[-1]

    # Downsample correction if downsampling was applied
    ds = downsample_step if downsample_step is not None else 1
    nt = tr.shape[1]
    nx_original = tr.shape[0] * ds

    # ------------------------------------------------------------------------
    # MATPLOTLIB FIGURE SETUP
    # ------------------------------------------------------------------------
    
    # Create figure and maximize to full screen for better visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("YOLO4DAS annotation tool")
    
    # Disable default matplotlib keyboard shortcuts that conflict with our custom ones
    # 'l' is used for log scale by default, we use it for LOAD
    # 's' is used for save by default, we use it for SELECT
    if 'l' in plt.rcParams['keymap.yscale']:
        plt.rcParams['keymap.yscale'].remove('l')
    if 's' in plt.rcParams['keymap.save']:
        plt.rcParams['keymap.save'].remove('s')
    
    # Attempt to maximize the window (platform-dependent)
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized()
    except Exception:
        pass

    # Layout: Top margin (0.90) is reserved for the custom toolbar axes.
    # This ensures buttons and controls don't overlap with the plot.
    plt.subplots_adjust(left=0.045, right=0.955, top=0.90, bottom=0.08)

    # Show loading message while processing large data
    loading_text = ax.text(0.5, 0.5, 
        'Loading data...', 
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
        fontsize=32, fontweight='bold')
    plt.draw()
    plt.pause(0.001)

    # ------------------------------------------------------------------------
    # DATA VISUALIZATION
    # ------------------------------------------------------------------------
    
    # Plotting: 'extent' maps matrix indices to physical units [seconds, meters].
    # This allows us to work with real-world coordinates instead of pixel indices.
    im = ax.imshow(
        tr,
        aspect="auto",
        origin="lower",
        extent=[time_s[0], time_s[-1], dist_m[0], dist_m[-1]],
        cmap="seismic",
        vmin=v_min,
        vmax=v_max
    )

    # Remove loading message
    loading_text.remove()

    # Apply ylim
    ax.set_ylim(ymin_val, ymax_val)

    # Axis styling for professional appearance
    ax.set_xlabel("Time (s)", fontsize=24)
    ax.set_ylabel("Distance (m)", fontsize=24)
    ax.tick_params(axis="both", labelsize=22)
    ax.tick_params(axis="y", labelrotation=90)

    # Colorbar to show amplitude scale
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cbar.set_label("Amplitude (a.u.)", fontsize=24)
    cbar.ax.tick_params(labelsize=22)

    # ------------------------------------------------------------------------
    # DISTANCE LIMIT LINES
    # ------------------------------------------------------------------------
    
    # Draw initial white horizontal lines for distance limits.
    # These lines help visualize where the fiber enters/exits the water,
    # marking the region of interest for analysis.
    line_dmin = ax.axhline(y=d_min, color='white', linewidth=2, linestyle='--')
    line_dmax = ax.axhline(y=d_max, color='white', linewidth=2, linestyle='--')

    # ------------------------------------------------------------------------
    # CROSSHAIR LINES FOR SELECT MODE (WITH TOGGLE)
    # ------------------------------------------------------------------------
    
    # Create crosshair lines (initially invisible)
    crosshair_h = ax.axhline(y=0, color='yellow', linewidth=1.5, linestyle='--', alpha=0.7, visible=False)
    crosshair_v = ax.axvline(x=0, color='yellow', linewidth=1.5, linestyle='--', alpha=0.7, visible=False)
    
    # Crosshair state control
    crosshair_state = {
        'enabled': False,  # Toggle state (controlled by SPACE key)
        'background': None  # For blitting optimization
    }

    # ------------------------------------------------------------------------
    # EVENT TRACKING DATA STRUCTURES
    # ------------------------------------------------------------------------
    
    # List to store all annotated events with their metadata
    events = []
    
    # Navigation state for time window controls
    current_window_size = None  # None means Full view, otherwise 5, 10, 20, or 60 seconds
    current_window_start = time_s[0]  # Current start position of the window
    
    # Dictionary to track the currently drawn rectangle during annotation
    current_rect = {"coords": None, "patch": None} 
    
    # Color cycle for events with IDs > 19
    color_cycle = plt.cm.tab20.colors
    id_color_map = {}

    # Fixed colors for the first 20 ID values [0-19]
    # This ensures consistent colors across different annotation sessions
    fixed_colors = {
        0: '#d62728',  # Red
        1: '#ff7f0e',  # Orange
        2: '#2ca02c',  # Green
        3: '#1f77b4',  # Blue
        4: '#9467bd',  # Purple
        5: '#8c564b',  # Brown
        6: '#e377c2',  # Pink
        7: '#7f7f7f',  # Gray
        8: '#bcbd22',  # Olive green
        9: '#17becf',  # Cyan
        10: '#aec7e8', # Light blue
        11: '#ffbb78', # Light orange
        12: '#98df8a', # Light green
        13: '#ff9896', # Light red
        14: '#c5b0d5', # Light purple
        15: '#c49c94', # Light brown
        16: '#f7b6d2', # Light pink
        17: '#c7c7c7', # Light gray
        18: '#dbdb8d', # Light olive green
        19: '#9edae5', # Light cyan
    }

    def get_color_for_id(event_id):
        """
        Get the color associated with a given event ID.
        IDs 0-19 use fixed colors for consistency.
        IDs >= 20 use colors from the tab20 colormap cycle.
        
        Parameters:
        -----------
        event_id : int
            The event identifier
            
        Returns:
        --------
        color : str or tuple
            The color to use for this event ID
        """
        if event_id in fixed_colors:
            return fixed_colors[event_id]
        if event_id not in id_color_map:
            id_color_map[event_id] = color_cycle[len(id_color_map) % len(color_cycle)]
        return id_color_map[event_id]

    def disable_toolbar_modes():
        """
        Disable zoom and pan modes from the matplotlib toolbar.
        This is called before activating SELECT mode to prevent conflicts
        between toolbar interactions and annotation drawing.
        """
        tb = fig.canvas.toolbar
        if tb is None: return
        for name in ("zoom", "pan"):
            if name in tb._actions:
                act = tb._actions[name]
                if act.isChecked():
                    act.setChecked(False)
                    act.trigger()

    def ask_event_id_and_comment():
        """
        Display a custom dialog asking the user to input an event ID and
        an optional free-text comment describing the event.

        The dialog contains:
        - A spin box for the integer event ID (default: 0)
        - A multi-line text area for an optional comment

        Returns:
        --------
        tuple (event_id, comment) or (None, None) if cancelled
            event_id : int   – The event ID entered by the user
            comment  : str   – The comment text (may be empty string)
        """
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Event ID")
        dialog.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- Event ID row ---
        id_layout = QtWidgets.QHBoxLayout()
        label_id = QtWidgets.QLabel("Event ID:")
        label_id.setStyleSheet("font-weight: bold; font-size: 10pt;")
        spin_id = QtWidgets.QSpinBox()
        spin_id.setRange(0, 99999)
        spin_id.setValue(0)
        spin_id.setFixedWidth(100)
        spin_id.setStyleSheet("font-size: 10pt; padding: 4px;")
        id_layout.addWidget(label_id)
        id_layout.addWidget(spin_id)
        id_layout.addStretch()
        layout.addLayout(id_layout)

        # --- Comment row ---
        label_comment = QtWidgets.QLabel("Comment (optional):")
        label_comment.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(label_comment)

        txt_comment = QtWidgets.QTextEdit()
        txt_comment.setPlaceholderText("Enter a description for this event…")
        txt_comment.setFixedHeight(90)
        txt_comment.setStyleSheet("font-size: 10pt; padding: 4px;")
        layout.addWidget(txt_comment)

        # --- OK / Cancel buttons ---
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        spin_id.setFocus()
        spin_id.selectAll()
        layout.addWidget(btn_box)

        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            event_id = spin_id.value()
            comment  = txt_comment.toPlainText().strip()
            return event_id, comment

        return None, None

    def save_event_with_id(event_id, comment=""):
        """
        Save the current annotation with the given event ID and comment.
        This function:
        1. Converts physical coordinates to array indices
        2. Calculates the event's absolute timestamp
        3. Applies the appropriate color to the annotation
        4. Stores all metadata (including comment) in the events list
        
        Parameters:
        -----------
        event_id : int
            The ID to assign to this event
        comment : str, optional
            A free-text description of the event (default: "")
        """
        # Extract coordinates from the current rectangle
        t0, t1, d0, d1 = current_rect["coords"]
        
        # Convert physical coordinates to array indices
        ti0 = int(np.searchsorted(time_s, t0))
        ti1 = int(np.searchsorted(time_s, t1))
        di0 = int(np.searchsorted(dist_m, d0))
        di1 = int(np.searchsorted(dist_m, d1))

        # Correct distance indexes to the original space, with no subsampling
        di0_orig = di0 * ds
        di1_orig = di1 * ds

        # Calculate the absolute UTC timestamp for this event
        start_datetime_utc_event = (start_datetime_utc + datetime.timedelta(seconds=t0))

        # Get color for this event ID and apply it to the rectangle
        color = get_color_for_id(event_id)
        current_rect["patch"].set_edgecolor(color)
        current_rect["patch"].set_linewidth(3)

        # Add text label showing the event ID
        txt = ax.text(
            t1, d1, f" ID: {event_id} ", 
            color='white', fontsize=10, fontweight='bold',
            va='bottom', ha='right',
            bbox=dict(facecolor=color, edgecolor=color, boxstyle='round,pad=0.2', alpha=0.8)
        )

        # Store all event metadata including the comment
        events.append({
            "ID": event_id,
            "t0": t0, "t1": t1,                                             # Time boundaries (seconds)
            "d0": d0, "d1": d1,                                             # Distance boundaries (meters)
            "ti0": ti0, "ti1": ti1,                                         # Time indices in array
            "di0": di0_orig, "di1": di1_orig,                               # Distance indices in array
            "nx": nx_original,                                              # Number of samples in the distance axix: shape[0] -> Channels
            "nt": nt,                                                       # Number of samples in the time axis: shape[1] -> Samples
            "start_datetime_utc": start_datetime_utc_event.isoformat(),     # Start timestamp UTC of the event. Example: 2021-11-04T02:00:08+00:00
            "comment": comment,                                             # Free-text annotation comment
            "artist_patch": current_rect["patch"],                          # Rectangle artist for removal
            "artist_text": txt                                              # Text artist for removal
        })

        # Clear current rectangle state and deactivate selector
        current_rect["coords"] = None
        current_rect["patch"] = None
        selector.set_active(False)
        
        # Hide crosshair when selection is complete and reset background
        crosshair_h.set_visible(False)
        crosshair_v.set_visible(False)
        crosshair_state['background'] = None
        
        fig.canvas.draw_idle()

    # ========================================================================
    # LOAD EXISTING ANNOTATIONS FROM CSV
    # ========================================================================
    
    def load_annotations_from_csv(csv_path):
        """
        Load annotations from an existing CSV file and display them on the plot.
        Supports CSV files both with and without a 'comment' column so that
        older annotation files remain fully compatible.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing annotations
            
        Returns:
        --------
        int : Number of annotations loaded
        """
        if not os.path.exists(csv_path):
            return 0
        
        loaded_count = 0
        
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Parse event data from CSV
                    event_id = int(row['ID'])
                    t0 = float(row['t0'])
                    t1 = float(row['t1'])
                    d0 = float(row['d0'])
                    d1 = float(row['d1'])
                    ti0 = int(row['ti0'])
                    ti1 = int(row['ti1'])
                    di0 = int(row['di0'])
                    di1 = int(row['di1'])
                    start_datetime_utc_str = row['start_datetime_utc']
                    # Graceful fallback: older CSV files may not have 'comment'
                    comment = row.get('comment', '')
                    
                    nx  = int(row.get('nx', nx_original))
                    nt_ = int(row.get('nt', nt)) # _ to insert
                    ds_ = int(row.get('downsample', 1)) # _ to insert
                    
                    # Get color for this event ID
                    color = get_color_for_id(event_id)
                    
                    # Draw rectangle on the plot
                    rect = plt.Rectangle(
                        (t0, d0), 
                        t1 - t0, 
                        d1 - d0,
                        fill=False, 
                        edgecolor=color, 
                        linewidth=3
                    )
                    ax.add_patch(rect)
                    
                    # Add text label
                    txt = ax.text(
                        t1, d1, f" ID: {event_id} ", 
                        color='white', fontsize=10, fontweight='bold',
                        va='bottom', ha='right',
                        bbox=dict(facecolor=color, edgecolor=color, boxstyle='round,pad=0.2', alpha=0.8)
                    )
                    
                    # Store event metadata (including comment)
                    events.append({
                        "ID": event_id,
                        "t0": t0, "t1": t1,
                        "d0": d0, "d1": d1,
                        "ti0": ti0, "ti1": ti1,
                        "di0": di0, "di1": di1,                             
                        "nx": nx,
                        "nt": nt_,
                        "downsample": ds_,
                        "start_datetime_utc": start_datetime_utc_str,
                        "comment": comment,
                        "artist_patch": rect,
                        "artist_text": txt
                    })
                    
                    loaded_count += 1
            
            # Redraw the figure with loaded annotations
            fig.canvas.draw_idle()
            
            print(f"Loaded {loaded_count} annotations from {csv_path}")
            
        except Exception as e:
            print(f"Error loading annotations from {csv_path}: {e}")
            return 0
        
        return loaded_count

    # Auto-load annotations if CSV file exists
    if filename and save_dir and annotations_output_format == 'csv':
        base = os.path.splitext(os.path.basename(filename))[0]
        auto_csv_path = os.path.join(save_dir, f"{base}_events.csv")
        
        if os.path.exists(auto_csv_path):
            # Show loading message
            loading_msg = ax.text(
                0.5, 0.5, 
                f'Loading existing annotations...', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes,
                fontsize=20, 
                fontweight='bold',
                color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )
            plt.draw()
            plt.pause(0.1)
            
            # Load annotations
            num_loaded = load_annotations_from_csv(auto_csv_path)
            
            # Remove loading message
            loading_msg.remove()
            
            # Show confirmation message briefly
            if num_loaded > 0:
                confirm_msg = ax.text(
                    0.5, 0.5, 
                    f'Loaded {num_loaded} annotation(s)', 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=ax.transAxes,
                    fontsize=18, 
                    fontweight='bold',
                    color='green',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
                )
                plt.draw()
                plt.pause(1.5)
                confirm_msg.remove()
            
            fig.canvas.draw_idle()

    # ========================================================================
    # NEW: SPECTRAL ANALYSIS WINDOW FUNCTION
    # ========================================================================
    
    def show_spectral_analysis_window(event_data):
        """
        Open a new window showing the frequency spectrum of randomly selected
        channels within the bounding box, along with their average spectrum.
        
        This function performs the following steps:
        1. Extracts signal data from channels within the bounding box
        2. Randomly selects up to MAX_SPECTRUMS channels
        3. Computes FFT for each selected channel
        4. Displays individual spectra with transparency
        5. Computes and displays the average spectrum in bold
        6. Provides Qt widgets for interactive FFT parameter control
        
        Parameters:
        -----------
        event_data : dict
            Event dictionary containing bounding box coordinates and indices
        """
        # Maximum number of spectra to display (to avoid overcrowding)
        MAX_SPECTRUMS = 30
        
        # Extract bounding box information
        di0, di1 = event_data["di0"], event_data["di1"]
        ti0, ti1 = event_data["ti0"], event_data["ti1"]
        t0, t1 = event_data["t0"], event_data["t1"]
        
        # Ensure we have valid channel range
        if di1 <= di0:
            di1 = di0 + 1
        
        # Calculate number of channels available in the bounding box
        num_channels = di1 - di0
        
        # Determine how many channels to analyze (up to MAX_SPECTRUMS)
        num_to_analyze = min(num_channels, MAX_SPECTRUMS)
        
        # Randomly select channel indices within the bounding box
        # This provides a representative sample of the event's spectral characteristics
        if num_channels <= MAX_SPECTRUMS:
            # Use all channels if we have fewer than MAX_SPECTRUMS
            selected_channels = list(range(di0, di1))
        else:
            # Randomly sample MAX_SPECTRUMS channels without replacement
            selected_channels = sorted(
                np.random.choice(range(di0, di1), size=num_to_analyze, replace=False)
            )
        
        # ------------------------------------------------------------------------
        # VISUALIZATION SETUP
        # ------------------------------------------------------------------------
        
        # Create new figure for spectral analysis with optimized layout
        spec_fig = plt.figure(figsize=(16, 9))
        spec_fig.canvas.manager.set_window_title(f"Spectral analysis - Event ID: {event_data['ID']}")
        
        # Create main axes for spectrum plot with space for legend on the right
        # Leave space at top for Qt widgets toolbar
        ax_spec = spec_fig.add_axes([0.06, 0.08, 0.70, 0.84])
        
        # ------------------------------------------------------------------------
        # Qt WIDGETS FOR FFT PARAMETERS
        # ------------------------------------------------------------------------
        
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        
        # Create container widget for FFT parameters
        param_widget = QtWidgets.QWidget()
        param_layout = QtWidgets.QHBoxLayout(param_widget)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(10)
        
        # NFFT parameter
        label_nfft = QtWidgets.QLabel("NFFT:")
        label_nfft.setStyleSheet("font-weight: bold; font-size: 10pt;")
        default_nfft = 2048  # Default FFT size
        txt_nfft = QtWidgets.QLineEdit(str(default_nfft))
        txt_nfft.setFixedWidth(100)
        txt_nfft.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_nfft.setToolTip("Number of FFT points (power of 2 recommended)")
        
        # Window type parameter
        label_window = QtWidgets.QLabel("Window:")
        label_window.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        combo_window = QtWidgets.QComboBox()
        combo_window.addItems(['hann', 'hamming', 'blackman', 'bartlett', 'none'])
        combo_window.setCurrentText('hann')
        combo_window.setFixedWidth(120)
        combo_window.setStyleSheet("font-size: 10pt; padding: 4px;")
        combo_window.setToolTip("Windowing function to reduce spectral leakage")
        
        # Max frequency parameter
        label_fmax = QtWidgets.QLabel("Max Freq (Hz):")
        label_fmax.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        txt_fmax = QtWidgets.QLineEdit(f"{fs_hz/2:.1f}")
        txt_fmax.setFixedWidth(100)
        txt_fmax.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_fmax.setToolTip("Maximum frequency to display")
        
        # Y-axis scale parameter
        label_scale = QtWidgets.QLabel("Y Scale:")
        label_scale.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        combo_scale = QtWidgets.QComboBox()
        combo_scale.addItems(['Linear', 'Log'])
        combo_scale.setCurrentText('Linear')
        combo_scale.setFixedWidth(100)
        combo_scale.setStyleSheet("font-size: 10pt; padding: 4px;")
        combo_scale.setToolTip("Y-axis scale (Linear or Logarithmic)")
        
        # Update button
        btn_update = QtWidgets.QPushButton("Apply")
        btn_update.setFixedWidth(120)
        btn_update.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;
                font-weight: bold;
                font-size: 10pt;
                padding: 6px;
                border: 1px solid #888;
                border-radius: 3px;
                margin-left: 15px;
            }
            QPushButton:hover {
                background-color: #76BDDA;
            }
        """)
        
        # Add widgets to layout
        param_layout.addWidget(label_nfft)
        param_layout.addWidget(txt_nfft)
        param_layout.addWidget(label_window)
        param_layout.addWidget(combo_window)
        param_layout.addWidget(label_fmax)
        param_layout.addWidget(txt_fmax)
        param_layout.addWidget(label_scale)
        param_layout.addWidget(combo_scale)
        param_layout.addWidget(btn_update)
        param_layout.addStretch()
        
        # Add parameter widget to the matplotlib window
        canvas_widget = spec_fig.canvas
        main_window = canvas_widget.parent()
        if main_window and hasattr(main_window, 'addToolBar'):
            toolbar = QtWidgets.QToolBar()
            toolbar.addWidget(param_widget)
            main_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
        else:
            param_widget.setWindowTitle("FFT Parameters")
            param_widget.show()
        
        # ------------------------------------------------------------------------
        # FFT COMPUTATION AND PLOTTING
        # ------------------------------------------------------------------------
        
        def compute_and_plot_spectrum():
            """
            Compute FFT with current parameters and update the plot.
            This function handles:
            - Parameter validation
            - FFT computation with windowing
            - Spectrum plotting with proper formatting
            - Legend generation with channel information
            """
            ax_spec.clear()
            
            # Get FFT parameters from UI
            try:
                nfft = int(txt_nfft.text())
                fmax = float(txt_fmax.text())
                window_type = combo_window.currentText()
                scale_type = combo_scale.currentText()
                
                # Validate parameters
                if nfft <= 0:
                    raise ValueError("NFFT must be positive")
                if fmax <= 0 or fmax > fs_hz / 2:
                    fmax = fs_hz / 2
                    txt_fmax.setText(f"{fmax:.1f}")
                
            except ValueError as e:
                ax_spec.text(0.5, 0.5, f"Invalid parameters: {e}", 
                           ha='center', va='center', fontsize=14, color='red')
                spec_fig.canvas.draw_idle()
                return
            
            # Extract time series data for selected channels within the time window
            # Shape: (num_to_analyze, num_time_samples)
            signals = tr[selected_channels, ti0:ti1]
            num_samples = signals.shape[1]
            
            # Apply windowing function if specified
            if window_type != 'none':
                if window_type == 'hann':
                    window = np.hanning(num_samples)
                elif window_type == 'hamming':
                    window = np.hamming(num_samples)
                elif window_type == 'blackman':
                    window = np.blackman(num_samples)
                elif window_type == 'bartlett':
                    window = np.bartlett(num_samples)
                
                # Apply window to each channel
                signals_windowed = signals * window[np.newaxis, :]
            else:
                signals_windowed = signals
            
            # Compute FFT for each selected channel
            spectra = []
            for i in range(len(selected_channels)):
                # Apply FFT to the windowed time series
                fft_result = np.fft.rfft(signals_windowed[i, :], n=nfft)
                
                # Compute magnitude spectrum (absolute value of complex FFT)
                magnitude = np.abs(fft_result)
                
                spectra.append(magnitude)
            
            # Convert list to numpy array for easier manipulation
            # Shape: (num_to_analyze, num_frequency_bins)
            spectra = np.array(spectra)
            
            # Compute frequency axis corresponding to the FFT bins
            freqs = np.fft.rfftfreq(nfft, d=1.0/fs_hz)
            
            # Compute average spectrum across all selected channels
            avg_spectrum = np.mean(spectra, axis=0)
            
            # Generate colors for individual spectra using a perceptually uniform colormap
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected_channels)))
            
            # Filter frequencies up to fmax for display
            freq_mask = freqs <= fmax
            freqs_filtered = freqs[freq_mask]
            
            # Plot individual spectra with transparency
            legend_entries = []
            for i, (channel_idx, spectrum) in enumerate(zip(selected_channels, spectra)):
                spectrum_filtered = spectrum[freq_mask]
                line, = ax_spec.plot(
                    freqs_filtered, 
                    spectrum_filtered, 
                    linewidth=0.8,           # Thin lines to reduce visual clutter
                    alpha=0.4,                # Transparency for overlapping spectra
                    color=colors[i]
                )
                # Add to legend: Channel number and distance
                legend_entries.append((line, f'CH {channel_idx}: {dist_m[channel_idx]:.2f}m'))
            
            # Plot average spectrum with emphasis
            avg_filtered = avg_spectrum[freq_mask]
            avg_line, = ax_spec.plot(
                freqs_filtered, 
                avg_filtered, 
                linewidth=2.5,               # Thicker line for emphasis
                color='black',               # High contrast color
                alpha=0.9,                   # Nearly opaque to stand out
                zorder=10                    # Draw on top of individual spectra
            )
            legend_entries.append((avg_line, 'Average Spectrum'))
            
            # Set Y-axis scale
            if scale_type == 'Log':
                ax_spec.set_yscale('log')
            else:
                ax_spec.set_yscale('linear')
            
            # Set axis labels with descriptive units
            ax_spec.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
            ax_spec.set_ylabel('Magnitude (a.u.)', fontsize=14, fontweight='bold')
            
            # Create informative title showing analysis parameters
            title = (f'Spectral Analysis - Event ID: {event_data["ID"]}\n'
                    f'Time: {t0:.2f}-{t1:.2f}s | Distance: {dist_m[di0]:.2f}-{dist_m[di1-1]:.2f}m | '
                    f'Channels: {len(selected_channels)}/{num_channels}')
            ax_spec.set_title(title, fontsize=16, fontweight='bold', pad=10)
            
            # Add grid for easier reading of values
            ax_spec.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Adjust tick label sizes for readability
            ax_spec.tick_params(labelsize=12)
            
            # Create legend on the right side with channel information
            # Extract lines and labels from legend_entries
            lines, labels = zip(*legend_entries)
            
            # Create legend outside the plot area on the right
            ax_spec.legend(
                lines, labels,
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                fontsize=9,
                framealpha=0.95,
                edgecolor='gray',
                title='Channels',
                title_fontsize=10
            )
            
            # Add information text box showing FFT parameters
            info_text = (f'FFT Size: {nfft}\n'
            f'Window: {window_type}\n'
            f'Sampling Rate: {fs_hz} Hz\n'
            f'Freq. Resolution: {freqs[1]-freqs[0]:.4f} Hz')

            ax_spec.text(
                0.98, 0.98, info_text,
                transform=ax_spec.transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
            )
            
            spec_fig.canvas.draw_idle()
        
        # Connect update button
        btn_update.clicked.connect(compute_and_plot_spectrum)
        
        # Connect Enter key in text fields to update
        txt_nfft.returnPressed.connect(compute_and_plot_spectrum)
        txt_fmax.returnPressed.connect(compute_and_plot_spectrum)
        
        # Initial plot with default parameters
        compute_and_plot_spectrum()
        
        # Display the spectral analysis window
        spec_fig.show()
    
    # ========================================================================
    # END OF SPECTRAL ANALYSIS WINDOW FUNCTION
    # ========================================================================

    # ========================================================================
    # SPECTROGRAM WINDOW FUNCTION (WITH DYNAMIC RANGE CONTROLS)
    # ========================================================================
    
    def show_spectrogram_window(event_data):
        """
        Open a new interactive window showing the spectrogram of channels
        within the bounding box, with a slider to navigate through channels
        and Qt widgets to control dynamic range (vMin/vMax).
        
        Parameters:
        -----------
        event_data : dict
            Event dictionary containing bounding box coordinates and indices
        """
        # Extract bounding box information
        di0, di1 = event_data["di0"], event_data["di1"]
        t0, t1 = event_data["t0"], event_data["t1"]
        
        # Ensure we have at least one channel
        if di1 <= di0:
            di1 = di0 + 1
        
        # Calculate central channel as default
        central_channel = int((di0 + di1) / 2)
        
        # Get the current colormap from main plot
        current_cmap = im.get_cmap().name
        
        # Create new figure for spectrogram with optimized spacing
        spec_fig = plt.figure(figsize=(14, 10))
        spec_fig.canvas.manager.set_window_title(f"Spectrogram - Event ID: {event_data['ID']}")
        
        # Create main axes for spectrogram with optimized margins
        # Reduced margins on all sides for better space utilization
        ax_spec = spec_fig.add_axes([0.04, 0.06, 0.92, 0.87])
        
        # Create axes for VERTICAL slider on the right side
        # Positioned very close to the plot edge
        ax_slider = spec_fig.add_axes([0.97, 0.06, 0.02, 0.87])
        
        # Store colorbar reference to prevent multiple colorbars
        cbar_ref = {'cbar': None}
        
        # Store current dynamic range values (will be initialized with first plot)
        dynamic_range = {'vmin': None, 'vmax': None}
        
        # ------------------------------------------------------------------------
        # Qt WIDGETS FOR SPECTROGRAM PARAMETERS
        # ------------------------------------------------------------------------
        
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        
        # Create container widget for spectrogram parameters
        param_widget = QtWidgets.QWidget()
        param_layout = QtWidgets.QHBoxLayout(param_widget)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(10)
        
        # NFFT parameter (FIRST in order)
        label_nfft = QtWidgets.QLabel("NFFT:")
        label_nfft.setStyleSheet("font-weight: bold; font-size: 10pt;")
        default_nfft = 1024; default_nperseg = 1024  # Fixed values
        txt_nfft = QtWidgets.QLineEdit(str(default_nperseg))
        txt_nfft.setFixedWidth(100)
        txt_nfft.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_nfft.setToolTip("Number of FFT points (should be >= NPERSEG)")
        
        # NPERSEG parameter (SECOND in order)
        label_nperseg = QtWidgets.QLabel("NPERSEG:")
        label_nperseg.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        txt_nperseg = QtWidgets.QLineEdit(str(default_nperseg))
        txt_nperseg.setFixedWidth(100)
        txt_nperseg.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_nperseg.setToolTip("Number of data points in each FFT segment")
        
        # OVERLAP parameter (THIRD in order)
        label_overlap = QtWidgets.QLabel("Overlap (%):")
        label_overlap.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        txt_overlap = QtWidgets.QLineEdit("50")
        txt_overlap.setFixedWidth(80)
        txt_overlap.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_overlap.setToolTip("Overlap between segments (0-99%)")
        
        # MAX FREQUENCY parameter (FOURTH in order)
        label_fmax = QtWidgets.QLabel("Max Freq (Hz):")
        label_fmax.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        txt_fmax = QtWidgets.QLineEdit("100")
        txt_fmax.setFixedWidth(100)
        txt_fmax.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_fmax.setToolTip("Maximum frequency to display")
        
        # Update button (FIFTH in order)
        btn_update_spec = QtWidgets.QPushButton("Apply")
        btn_update_spec.setFixedWidth(150)
        btn_update_spec.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;
                font-weight: bold;
                font-size: 10pt;
                padding: 6px;
                border: 1px solid #888;
                border-radius: 3px;
                margin-left: 15px;
            }
            QPushButton:hover {
                background-color: #76BDDA;
            }
        """)
        
        # Channel info label (updates dynamically)
        label_channel_info = QtWidgets.QLabel("")
        label_channel_info.setStyleSheet("font-size: 10pt; margin-left: 20px;")
        
        # Add widgets to layout in the specified order
        param_layout.addWidget(label_nfft)
        param_layout.addWidget(txt_nfft)
        param_layout.addWidget(label_nperseg)
        param_layout.addWidget(txt_nperseg)
        param_layout.addWidget(label_overlap)
        param_layout.addWidget(txt_overlap)
        param_layout.addWidget(label_fmax)
        param_layout.addWidget(txt_fmax)
        param_layout.addWidget(btn_update_spec)
        param_layout.addWidget(label_channel_info)
        param_layout.addStretch()
        
        # ------------------------------------------------------------------------
        # DYNAMIC RANGE CONTROLS (vMin/vMax) - SECOND TOOLBAR
        # ------------------------------------------------------------------------
        
        # Create container widget for dynamic range controls
        range_widget = QtWidgets.QWidget()
        range_layout = QtWidgets.QHBoxLayout(range_widget)
        range_layout.setContentsMargins(5, 5, 5, 5)
        range_layout.setSpacing(10)
        
        # vMin parameter
        label_vmin_spec = QtWidgets.QLabel("vMin:")
        label_vmin_spec.setStyleSheet("font-weight: bold; font-size: 10pt;")
        txt_vmin_spec = QtWidgets.QLineEdit("")
        txt_vmin_spec.setFixedWidth(150)
        txt_vmin_spec.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_vmin_spec.setToolTip("Minimum value for colormap")
        
        # vMax parameter
        label_vmax_spec = QtWidgets.QLabel("vMax:")
        label_vmax_spec.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 15px;")
        txt_vmax_spec = QtWidgets.QLineEdit("")
        txt_vmax_spec.setFixedWidth(150)
        txt_vmax_spec.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_vmax_spec.setToolTip("Maximum value for colormap")
        
        # Apply button for dynamic range
        btn_apply_range = QtWidgets.QPushButton("Apply")
        btn_apply_range.setFixedWidth(150)
        btn_apply_range.setStyleSheet("""
            QPushButton {
                background-color: #FFD700;
                font-weight: bold;
                font-size: 10pt;
                padding: 6px;
                border: 1px solid #888;
                border-radius: 3px;
                margin-left: 15px;
            }
            QPushButton:hover {
                background-color: #FFC700;
            }
        """)
        
        # Add widgets to range layout
        range_layout.addWidget(label_vmin_spec)
        range_layout.addWidget(txt_vmin_spec)
        range_layout.addWidget(label_vmax_spec)
        range_layout.addWidget(txt_vmax_spec)
        range_layout.addWidget(btn_apply_range)
        range_layout.addStretch()
        
        # Add both parameter widgets to the matplotlib window
        canvas_widget = spec_fig.canvas
        main_window = canvas_widget.parent()
        if main_window and hasattr(main_window, 'addToolBar'):
            # Add parameter toolbar
            toolbar1 = QtWidgets.QToolBar()
            toolbar1.addWidget(param_widget)
            main_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar1)
            
            # Add range toolbar
            toolbar2 = QtWidgets.QToolBar()
            toolbar2.addWidget(range_widget)
            main_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar2)
        else:
            param_widget.setWindowTitle("Spectrogram Parameters")
            param_widget.show()
            range_widget.setWindowTitle("Dynamic Range")
            range_widget.show()
        
        # ------------------------------------------------------------------------
        # SPECTROGRAM COMPUTATION AND PLOTTING
        # ------------------------------------------------------------------------
        
        # Store spectrogram data for each channel
        spec_cache = {}
        
        def compute_spectrogram(channel_idx, nperseg, noverlap, nfft):
            """
            Compute spectrogram for a given channel using scipy.signal.spectrogram.
            Returns PSD (Power Spectral Density).
            
            Parameters:
            -----------
            channel_idx : int
                Index of the channel (row) in the tr array
            nperseg : int
                Length of each segment for FFT
            noverlap : int
                Number of points to overlap between segments
            nfft : int
                Length of the FFT
                
            Returns:
            --------
            f : np.ndarray
                Frequency bins
            t : np.ndarray
                Time bins
            Sxx : np.ndarray
                Spectrogram (power spectral density in dB)
            """
            # Get the signal for this channel (entire time series)
            signal_data = tr[channel_idx, :]
            
            # Compute spectrogram using scipy - PSD mode
            f, t, Sxx = signal.spectrogram(
                signal_data,
                fs=fs_hz,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                scaling='density',  # Power Spectral Density
                mode='psd'          # PSD mode
            )
            
            # Convert to dB scale for better visualization
            # PSD is already in power units, so 10*log10 is correct
            Sxx_db = 10 * np.log10(Sxx + 1e-20)  # Add small value to avoid log(0)
            
            # Adjust time to match the original time_s reference
            t_adjusted = t + time_s[0]
            
            return f, t_adjusted, Sxx_db
        
        def plot_spectrogram(channel_idx, update_range_widgets=True):
            """
            Plot the spectrogram for the given channel index.
            
            Parameters:
            -----------
            channel_idx : int
                Index of the channel to display
            update_range_widgets : bool
                If True, update vMin/vMax text fields with current values
            """
            ax_spec.clear()
            
            # Get spectrogram parameters from UI
            try:
                nperseg = int(txt_nperseg.text())
                overlap_pct = float(txt_overlap.text())
                nfft = int(txt_nfft.text())
                fmax = float(txt_fmax.text())
                
                # Validate parameters
                if nperseg <= 0 or nfft <= 0:
                    raise ValueError("NPERSEG and NFFT must be positive")
                if not (0 <= overlap_pct < 100):
                    raise ValueError("Overlap must be between 0 and 99")
                if nfft < nperseg:
                    nfft = nperseg
                    txt_nfft.setText(str(nfft))
                if fmax <= 0 or fmax > fs_hz / 2:
                    fmax = fs_hz / 2
                    txt_fmax.setText(f"{fmax:.1f}")
                
                noverlap = int(nperseg * overlap_pct / 100)
                
            except ValueError as e:
                ax_spec.text(0.5, 0.5, f"Invalid parameters: {e}", 
                           ha='center', va='center', fontsize=14, color='red')
                spec_fig.canvas.draw_idle()
                return
            
            # Create cache key
            cache_key = (channel_idx, nperseg, noverlap, nfft)
            
            # Check if we have cached data
            if cache_key not in spec_cache:
                spec_cache[cache_key] = compute_spectrogram(channel_idx, nperseg, noverlap, nfft)
            
            f, t, Sxx_db = spec_cache[cache_key]
            
            # Filter frequencies up to fmax
            freq_mask = f <= fmax
            f_filtered = f[freq_mask]
            Sxx_filtered = Sxx_db[freq_mask, :]
            
            # Initialize dynamic range from first plot if not set
            if dynamic_range['vmin'] is None or dynamic_range['vmax'] is None:
                dynamic_range['vmin'] = np.min(Sxx_filtered)
                dynamic_range['vmax'] = np.max(Sxx_filtered)
                if update_range_widgets:
                    txt_vmin_spec.setText(f"{dynamic_range['vmin']:.2f}")
                    txt_vmax_spec.setText(f"{dynamic_range['vmax']:.2f}")
            
            # Plot spectrogram with current dynamic range
            pcm = ax_spec.pcolormesh(
                t, f_filtered, Sxx_filtered, 
                shading='gouraud', 
                cmap=current_cmap,
                vmin=dynamic_range['vmin'],
                vmax=dynamic_range['vmax']
            )
            
            # Add vertical dashed WHITE lines for bounding box time limits (thinner)
            ax_spec.axvline(x=t0, color='white', linewidth=1.5, linestyle='--')
            ax_spec.axvline(x=t1, color='white', linewidth=1.5, linestyle='--')
            
            # Labels and formatting
            ax_spec.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
            ax_spec.set_ylabel('Frequency (Hz)', fontsize=14, fontweight='bold')
            ax_spec.set_title(f'Spectrogram (PSD) - Channel {channel_idx} (Distance: {dist_m[channel_idx]:.2f} m)', 
                            fontsize=16, fontweight='bold')
            ax_spec.tick_params(labelsize=12)
            ax_spec.grid(True, alpha=0.3)
            
            # Handle colorbar - IMPROVED METHOD
            # Instead of removing, we update the existing one or create new
            if cbar_ref['cbar'] is not None:
                try:
                    # Try to update existing colorbar
                    cbar_ref['cbar'].update_normal(pcm)
                except:
                    # If update fails, set to None and create new one
                    cbar_ref['cbar'] = None
            
            # Create colorbar only if it doesn't exist
            if cbar_ref['cbar'] is None:
                cbar_ref['cbar'] = spec_fig.colorbar(pcm, ax=ax_spec, pad=0.01, fraction=0.046)
                cbar_ref['cbar'].set_label('PSD (dB/Hz)', fontsize=12, fontweight='bold')
                cbar_ref['cbar'].ax.tick_params(labelsize=10)
            
            # Update channel info label in Qt toolbar
            label_channel_info.setText(
                f"Channel: {channel_idx}/{tr.shape[0]-1} | "
                f"Distance: {dist_m[channel_idx]:.2f} m | "
                f"Freq Range: 0-{fmax:.1f} Hz"
            )
            
            spec_fig.canvas.draw_idle()
        
        # ------------------------------------------------------------------------
        # VERTICAL SLIDER FOR CHANNEL NAVIGATION
        # ------------------------------------------------------------------------
        
        # Create VERTICAL slider to navigate through channels
        # Start at central channel
        slider = Slider(
            ax_slider,
            'Channel',
            di0,
            di1 - 1,
            valinit=central_channel,  # Start at center
            valstep=1,
            color='#87CEEB',
            orientation='vertical'
        )
        
        # Slider callback - updates plot WITHOUT updating range widgets
        def on_slider_change(val):
            channel_idx = int(slider.val)
            plot_spectrogram(channel_idx, update_range_widgets=False)
        
        slider.on_changed(on_slider_change)
        
        # Update button callback
        def on_update_spectrogram():
            # Clear cache when parameters change
            spec_cache.clear()
            # Reset dynamic range to auto-calculate for new parameters
            dynamic_range['vmin'] = None
            dynamic_range['vmax'] = None
            channel_idx = int(slider.val)
            plot_spectrogram(channel_idx, update_range_widgets=True)
        
        btn_update_spec.clicked.connect(on_update_spectrogram)
        
        # Dynamic range apply button callback
        def on_apply_range():
            try:
                vmin_val = float(txt_vmin_spec.text())
                vmax_val = float(txt_vmax_spec.text())
                dynamic_range['vmin'] = vmin_val
                dynamic_range['vmax'] = vmax_val
                channel_idx = int(slider.val)
                plot_spectrogram(channel_idx, update_range_widgets=False)
            except ValueError:
                pass  # Ignore invalid input
        
        btn_apply_range.clicked.connect(on_apply_range)
        
        # Keyboard shortcuts for updating
        txt_nperseg.returnPressed.connect(on_update_spectrogram)
        txt_overlap.returnPressed.connect(on_update_spectrogram)
        txt_nfft.returnPressed.connect(on_update_spectrogram)
        txt_fmax.returnPressed.connect(on_update_spectrogram)
        
        # Keyboard shortcuts for range (Enter key applies)
        txt_vmin_spec.returnPressed.connect(on_apply_range)
        txt_vmax_spec.returnPressed.connect(on_apply_range)
        
        # Initial plot at central channel
        plot_spectrogram(central_channel, update_range_widgets=True)
        
        # Show the spectrogram window
        spec_fig.show()
    
    # ========================================================================
    # END OF SPECTROGRAM WINDOW FUNCTION
    # ========================================================================

    # ========================================================================
    # SIGNAL IN TIME DOMAIN WINDOW FUNCTION
    # ========================================================================

    def show_signal_window(event_data):
        """
        Open a new interactive window showing the raw time-domain signal
        (a single row of tr) for channels within the bounding box.

        Features:
        - Vertical slider to navigate through channels inside the bounding box.
        - "Fix channel" button: pins the currently displayed channel so it
          stays plotted even when the slider moves to another channel.
          Multiple channels can be fixed simultaneously.
        - Qt toolbar with yMin / yMax controls to adjust the y-axis range.
        - Fixed channels are drawn with semi-transparent colored lines;
          the active (slider) channel is drawn as a solid black line on top.
        - Vertical dashed white lines mark the bounding-box time limits.

        Parameters:
        -----------
        event_data : dict
            Event dictionary containing bounding box coordinates and indices.
        """

        # ----------------------------------------------------------------
        # Extract bounding-box limits
        # ----------------------------------------------------------------
        di0, di1 = event_data["di0"], event_data["di1"]
        t0,  t1  = event_data["t0"],  event_data["t1"]

        # Ensure at least one channel
        if di1 <= di0:
            di1 = di0 + 1

        central_channel = int((di0 + di1) / 2)

        # ----------------------------------------------------------------
        # Figure layout
        # ----------------------------------------------------------------
        sig_fig = plt.figure(figsize=(14, 6))
        sig_fig.canvas.manager.set_window_title(
            f"Signal (time domain) – Event ID: {event_data['ID']}"
        )

        # Main axes (leave right margin for the vertical slider)
        ax_sig    = sig_fig.add_axes([0.06, 0.10, 0.88, 0.82])
        ax_slider = sig_fig.add_axes([0.96, 0.10, 0.02, 0.82])

        # ----------------------------------------------------------------
        # State: fixed channels list and color cycle for them
        # ----------------------------------------------------------------
        fixed_channels  = []          # list of channel indices that are pinned
        fixed_colors_sig = plt.cm.tab10.colors   # colours for fixed traces

        # ----------------------------------------------------------------
        # Qt TOOLBAR – yMin / yMax + Fix channel button
        # ----------------------------------------------------------------
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        ctrl_widget  = QtWidgets.QWidget()
        ctrl_layout  = QtWidgets.QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(5, 5, 5, 5)
        ctrl_layout.setSpacing(10)

        # yMin
        lbl_ymin = QtWidgets.QLabel("yMin:")
        lbl_ymin.setStyleSheet("font-weight: bold; font-size: 10pt;")
        txt_ymin_sig = QtWidgets.QLineEdit("")
        txt_ymin_sig.setFixedWidth(160)
        txt_ymin_sig.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_ymin_sig.setToolTip("Minimum amplitude for y-axis")

        # yMax
        lbl_ymax = QtWidgets.QLabel("yMax:")
        lbl_ymax.setStyleSheet(
            "font-weight: bold; font-size: 10pt; margin-left: 15px;"
        )
        txt_ymax_sig = QtWidgets.QLineEdit("")
        txt_ymax_sig.setFixedWidth(160)
        txt_ymax_sig.setStyleSheet("font-size: 10pt; padding: 4px;")
        txt_ymax_sig.setToolTip("Maximum amplitude for y-axis")

        # Apply y-axis button
        btn_apply_y = QtWidgets.QPushButton("Apply")
        btn_apply_y.setFixedWidth(120)
        btn_apply_y.setStyleSheet("""
            QPushButton {
                background-color: #FFD700;
                font-weight: bold; font-size: 10pt;
                padding: 6px; border: 1px solid #888;
                border-radius: 3px; margin-left: 15px;
            }
            QPushButton:hover { background-color: #FFC700; }
        """)

        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setStyleSheet("background-color: #888; margin-left: 10px;")

        # Fix channel button
        btn_fix = QtWidgets.QPushButton("📌 Fix channel")
        btn_fix.setFixedWidth(160)
        btn_fix.setStyleSheet("""
            QPushButton {
                background-color: #90EE90;
                font-weight: bold; font-size: 10pt;
                padding: 6px; border: 1px solid #888;
                border-radius: 3px; margin-left: 10px;
            }
            QPushButton:hover { background-color: #7FDD7F; }
        """)
        btn_fix.setToolTip(
            "Pin the current channel so it stays visible when navigating"
        )

        # Clear fixed button
        btn_clear_fix = QtWidgets.QPushButton("🗑 Clear fixed")
        btn_clear_fix.setFixedWidth(160)
        btn_clear_fix.setStyleSheet("""
            QPushButton {
                background-color: #f08080;
                font-weight: bold; font-size: 10pt;
                padding: 6px; border: 1px solid #888;
                border-radius: 3px; margin-left: 10px;
            }
            QPushButton:hover { background-color: #e07070; }
        """)
        btn_clear_fix.setToolTip("Remove all pinned channels")

        # Channel info label (updated dynamically)
        lbl_ch_info = QtWidgets.QLabel("")
        lbl_ch_info.setStyleSheet("font-size: 10pt; margin-left: 20px; color: #333;")

        ctrl_layout.addWidget(lbl_ymin)
        ctrl_layout.addWidget(txt_ymin_sig)
        ctrl_layout.addWidget(lbl_ymax)
        ctrl_layout.addWidget(txt_ymax_sig)
        ctrl_layout.addWidget(btn_apply_y)
        ctrl_layout.addWidget(sep)
        ctrl_layout.addWidget(btn_fix)
        ctrl_layout.addWidget(btn_clear_fix)
        ctrl_layout.addWidget(lbl_ch_info)
        ctrl_layout.addStretch()

        # Attach toolbar to the matplotlib window
        canvas_sig   = sig_fig.canvas
        main_win_sig = canvas_sig.parent()
        if main_win_sig and hasattr(main_win_sig, 'addToolBar'):
            tb = QtWidgets.QToolBar()
            tb.addWidget(ctrl_widget)
            main_win_sig.addToolBar(QtCore.Qt.TopToolBarArea, tb)
        else:
            ctrl_widget.setWindowTitle("Signal controls")
            ctrl_widget.show()

        # ----------------------------------------------------------------
        # Core plot function
        # ----------------------------------------------------------------
        def plot_signal(channel_idx):
            """
            Redraw ax_sig with:
            - All fixed channels (semi-transparent, coloured).
            - The currently active channel (solid black, on top).

            The y-axis range is kept if the user has already set it
            manually via the yMin/yMax widgets; otherwise it is
            auto-fitted to the active channel's amplitude range.
            """
            ax_sig.clear()

            # Draw fixed channels first (background)
            for k, ch in enumerate(fixed_channels):
                color_k = fixed_colors_sig[k % len(fixed_colors_sig)]
                sig_k   = tr[ch, :]
                ax_sig.plot(
                    time_s, sig_k,
                    linewidth=1.2,
                    alpha=0.55,
                    color=color_k,
                    label=f"CH {ch} ({dist_m[ch]:.2f} m) [fixed]",
                    zorder=2
                )

            # Draw active channel on top
            sig_active = tr[channel_idx, :]
            ax_sig.plot(
                time_s, sig_active,
                linewidth=1.8,
                color='black',
                alpha=0.92,
                label=f"CH {channel_idx} ({dist_m[channel_idx]:.2f} m)",
                zorder=5
            )

            # Bounding-box time markers
            ax_sig.axvline(x=t0, color='red',  linewidth=1.5,
                           linestyle='--', alpha=0.8, zorder=6)
            ax_sig.axvline(x=t1, color='red',  linewidth=1.5,
                           linestyle='--', alpha=0.8, zorder=6)

            # Labels
            ax_sig.set_xlabel("Time (s)", fontsize=13, fontweight='bold')
            ax_sig.set_ylabel("Amplitude (a.u.)", fontsize=13, fontweight='bold')
            ax_sig.set_title(
                f"Time-domain signal – Event ID: {event_data['ID']} | "
                f"CH {channel_idx}  ({dist_m[channel_idx]:.2f} m)",
                fontsize=14, fontweight='bold'
            )
            ax_sig.tick_params(labelsize=11)
            ax_sig.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            if fixed_channels:
                ax_sig.legend(
                    loc='upper right', fontsize=9,
                    framealpha=0.9, edgecolor='gray'
                )

            # Apply y-axis limits: use widget values if set, else auto
            try:
                ymin_v = float(txt_ymin_sig.text())
                ymax_v = float(txt_ymax_sig.text())
                ax_sig.set_ylim(ymin_v, ymax_v)
            except ValueError:
                # Auto-range: fit active channel (fixed channels may be different scale)
                amp     = sig_active
                margin  = (amp.max() - amp.min()) * 0.05 if amp.max() != amp.min() else 1e-12
                auto_mn = amp.min() - margin
                auto_mx = amp.max() + margin
                ax_sig.set_ylim(auto_mn, auto_mx)
                # Write auto values into the widgets so the user can see them
                txt_ymin_sig.setText(f"{auto_mn:.6e}")
                txt_ymax_sig.setText(f"{auto_mx:.6e}")

            # Update info label
            n_fixed = len(fixed_channels)
            lbl_ch_info.setText(
                f"Channel: {channel_idx}/{tr.shape[0]-1}  |  "
                f"Distance: {dist_m[channel_idx]:.2f} m  |  "
                f"Fixed: {n_fixed}"
            )

            sig_fig.canvas.draw_idle()

        # ----------------------------------------------------------------
        # Vertical slider
        # ----------------------------------------------------------------
        # Only create the slider if the bounding box covers > 1 channel
        if di1 - 1 > di0:
            slider_sig = Slider(
                ax_slider, 'CH',
                di0, di1 - 1,
                valinit=central_channel,
                valstep=1,
                color='#90EE90',
                orientation='vertical'
            )

            def on_slider_sig(val):
                plot_signal(int(slider_sig.val))

            slider_sig.on_changed(on_slider_sig)
        else:
            # Single channel: hide slider axes
            ax_slider.set_visible(False)

        # ----------------------------------------------------------------
        # Button callbacks
        # ----------------------------------------------------------------
        def on_fix_channel():
            """Pin the currently visible channel."""
            ch = int(slider_sig.val) if di1 - 1 > di0 else di0
            if ch not in fixed_channels:
                fixed_channels.append(ch)
                print(f"📌 Fixed channel {ch} ({dist_m[ch]:.2f} m)")
                plot_signal(ch)

        def on_clear_fixed():
            """Remove all pinned channels."""
            fixed_channels.clear()
            ch = int(slider_sig.val) if di1 - 1 > di0 else di0
            print("🗑 Cleared all fixed channels")
            plot_signal(ch)

        def on_apply_ylim():
            """Apply the yMin/yMax values from the widgets."""
            ch = int(slider_sig.val) if di1 - 1 > di0 else di0
            plot_signal(ch)

        btn_fix.clicked.connect(on_fix_channel)
        btn_clear_fix.clicked.connect(on_clear_fixed)
        btn_apply_y.clicked.connect(on_apply_ylim)
        txt_ymin_sig.returnPressed.connect(on_apply_ylim)
        txt_ymax_sig.returnPressed.connect(on_apply_ylim)

        # ----------------------------------------------------------------
        # Initial plot
        # ----------------------------------------------------------------
        plot_signal(central_channel)
        sig_fig.show()

    # ========================================================================
    # END OF SIGNAL WINDOW FUNCTION
    # ========================================================================

    def onselect(eclick, erelease):
        """
        Callback function triggered when the user completes a rectangle selection.
        This function:
        1. Validates the selection coordinates
        2. Draws a yellow rectangle on the plot
        3. Asks the user for an event ID and optional comment
        4. Saves the event if an ID is provided
        
        Parameters:
        -----------
        eclick : matplotlib.backend_bases.MouseEvent
            Mouse event at the start of the selection
        erelease : matplotlib.backend_bases.MouseEvent
            Mouse event at the end of the selection
        """
        if not selector.active: return
        
        # Extract coordinates from the mouse events
        tx, ty = eclick.xdata, erelease.xdata
        dx, dy = eclick.ydata, erelease.ydata
        if None in [tx, ty, dx, dy]: return

        # Ensure coordinates are ordered (min, max)
        t_start, t_end = sorted([tx, ty])
        d_start, d_end = sorted([dx, dy])

        # Remove the previous rectangle if it exists
        if current_rect["patch"] is not None:
            current_rect["patch"].remove()

        # Draw a new yellow rectangle for the annotation
        rect = plt.Rectangle((t_start, d_start), t_end - t_start, d_end - d_start,
                             fill=False, edgecolor="yellow", linewidth=2.5)
        ax.add_patch(rect)
        current_rect["coords"] = (t_start, t_end, d_start, d_end)
        current_rect["patch"] = rect
        fig.canvas.draw_idle()

        # Ask user for event ID and comment
        event_id, comment = ask_event_id_and_comment()
        if event_id is None:
            # User cancelled - remove the rectangle
            rect.remove()
            current_rect["coords"] = None
            current_rect["patch"] = None
            fig.canvas.draw_idle()
        else:
            # Save the event with the provided ID and comment
            save_event_with_id(event_id, comment)

    def on_click(event):
        """
        Handle right-click events for editing ID/comment, removing annotations,
        showing spectrogram, or performing spectral analysis.
        When the user right-clicks inside an existing annotation,
        a context menu appears with four options:
        1. Edit ID & Comment  (opens a unified edit dialog)
        2. Remove event
        3. Show spectrogram (time-frequency representation)
        4. Spectral analysis (frequency domain representation)
        
        Parameters:
        -----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse click event
        """
        if event.button == 3 and event.inaxes == ax:  # Right-click
            # Check if click is inside any existing event
            for i, ev in enumerate(events):
                if (ev["t0"] <= event.xdata <= ev["t1"] and 
                    ev["d0"] <= event.ydata <= ev["d1"]):
                    # Create context menu with five options
                    menu = QtWidgets.QMenu()
                    edit_id_act = menu.addAction("✏️ Edit event")
                    remove_act = menu.addAction("🗑️ Remove event")
                    menu.addSeparator()  # Visual separator
                    signal_act  = menu.addAction("📉 Show signal(t)")
                    spec_act = menu.addAction("📊 Show spectrogram")
                    spectral_act = menu.addAction("📈 Spectral analysis")
                    
                    # Show menu at click position
                    pos = fig.canvas.mapToGlobal(QtCore.QPoint(
                        int(event.x), int(fig.canvas.get_width_height()[1] - event.y)))
                    action = menu.exec_(pos)
                    
                    # Handle user selection
                    if action == edit_id_act:
                        # --------------------------------------------------------
                        # EDIT ID & COMMENT  –  unified dialog
                        # --------------------------------------------------------
                        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

                        dialog = QtWidgets.QDialog()
                        dialog.setWindowTitle("Edit Annotation")
                        dialog.setMinimumWidth(440)

                        dlg_layout = QtWidgets.QVBoxLayout(dialog)
                        dlg_layout.setSpacing(10)
                        dlg_layout.setContentsMargins(16, 16, 16, 16)

                        # Current values shown as a small info label
                        info_label = QtWidgets.QLabel(
                            f"Editing event at t=[{ev['t0']:.2f}, {ev['t1']:.2f}] s   "
                            f"d=[{ev['d0']:.2f}, {ev['d1']:.2f}] m"
                        )
                        info_label.setStyleSheet("color: #555; font-size: 9pt;")
                        dlg_layout.addWidget(info_label)

                        # --- ID row ---
                        id_row = QtWidgets.QHBoxLayout()
                        lbl_id = QtWidgets.QLabel("Event ID:")
                        lbl_id.setStyleSheet("font-weight: bold; font-size: 10pt;")
                        spin_id = QtWidgets.QSpinBox()
                        spin_id.setRange(0, 99999)
                        spin_id.setValue(int(ev['ID']))
                        spin_id.setFixedWidth(100)
                        spin_id.setStyleSheet("font-size: 10pt; padding: 4px;")
                        id_row.addWidget(lbl_id)
                        id_row.addWidget(spin_id)
                        id_row.addStretch()
                        dlg_layout.addLayout(id_row)

                        # --- Comment row ---
                        lbl_comment = QtWidgets.QLabel("Comment:")
                        lbl_comment.setStyleSheet("font-weight: bold; font-size: 10pt;")
                        dlg_layout.addWidget(lbl_comment)

                        txt_comment_edit = QtWidgets.QTextEdit()
                        txt_comment_edit.setPlaceholderText("Enter a description for this event…")
                        txt_comment_edit.setFixedHeight(100)
                        txt_comment_edit.setStyleSheet("font-size: 10pt; padding: 4px;")
                        # Pre-fill with existing comment (empty string if none)
                        txt_comment_edit.setPlainText(ev.get('comment', ''))
                        dlg_layout.addWidget(txt_comment_edit)

                        # --- OK / Cancel buttons ---
                        btn_box = QtWidgets.QDialogButtonBox(
                            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                        )
                        btn_box.accepted.connect(dialog.accept)
                        btn_box.rejected.connect(dialog.reject)
                        spin_id.setFocus()
                        spin_id.selectAll()
                        dlg_layout.addWidget(btn_box)

                        result = dialog.exec_()

                        if result == QtWidgets.QDialog.Accepted:
                            new_id      = spin_id.value()
                            new_comment = txt_comment_edit.toPlainText().strip()

                            old_id = ev['ID']

                            # Update stored values
                            ev['ID']      = new_id
                            ev['comment'] = new_comment

                            # Get new color for the new ID
                            new_color = get_color_for_id(new_id)

                            # Update rectangle color
                            ev["artist_patch"].set_edgecolor(new_color)

                            # Update text label
                            ev["artist_text"].remove()
                            txt = ax.text(
                                ev['t1'], ev['d1'], f" ID: {new_id} ",
                                color='white', fontsize=10, fontweight='bold',
                                va='bottom', ha='right',
                                bbox=dict(facecolor=new_color, edgecolor=new_color,
                                         boxstyle='round,pad=0.2', alpha=0.8)
                            )
                            ev["artist_text"] = txt

                            # Invalidate crosshair background
                            crosshair_state['background'] = None

                            # Redraw
                            fig.canvas.draw_idle()

                            print(f"Event updated → ID: {old_id} → {new_id} | "
                                  f"Comment: \"{new_comment}\"")
                    
                    elif action == remove_act:
                        # Remove event
                        ev["artist_patch"].remove()
                        ev["artist_text"].remove()
                        events.pop(i)
                        crosshair_state['background'] = None
                        fig.canvas.draw_idle()
                        print(f"🗑️ Removed event ID: {ev['ID']}")
                    
                    elif action == signal_act:
                        # Show time-domain signal
                        show_signal_window(ev)

                    elif action == spec_act:
                        # Show spectrogram (time-frequency analysis)
                        show_spectrogram_window(ev)
                    
                    elif action == spectral_act:
                        # Show spectral analysis (frequency domain)
                        show_spectral_analysis_window(ev)
                    
                    break

    def on_mouse_move(event):
        """
        Handle mouse movement to update crosshair position when enabled.
        Only updates when crosshair is enabled via SPACE key.
        Uses blitting for optimal performance.
        
        Parameters:
        -----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse motion event
        """
        # Only proceed if crosshair is enabled and we're in the axes
        if not crosshair_state['enabled'] or event.inaxes != ax:
            return
        
        # Only update if SELECT mode is active
        if not selector.active:
            return
        
        # Update crosshair position
        crosshair_h.set_ydata([event.ydata, event.ydata])
        crosshair_v.set_xdata([event.xdata, event.xdata])
        
        # Make visible if not already
        if not crosshair_h.get_visible():
            crosshair_h.set_visible(True)
            crosshair_v.set_visible(True)
        
        # Use blitting for fast update
        if crosshair_state['background'] is None:
            crosshair_state['background'] = fig.canvas.copy_from_bbox(ax.bbox)
        
        # Restore background
        fig.canvas.restore_region(crosshair_state['background'])
        
        # Redraw only the crosshair lines
        ax.draw_artist(crosshair_h)
        ax.draw_artist(crosshair_v)
        
        # Blit the changes
        fig.canvas.blit(ax.bbox)

    # ------------------------------------------------------------------------
    # RECTANGLE SELECTOR SETUP
    # ------------------------------------------------------------------------
    
    # Create the rectangle selector for drawing annotations
    # useblit=True for better performance
    # button=[1] means only left mouse button activates it
    selector = RectangleSelector(ax, onselect, useblit=True, button=[1])
    selector.set_active(False)  # Start inactive; activated by SELECT button
    
    # Connect the right-click handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Connect the mouse motion handler for crosshair
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # ------------------------------------------------------------------------
    # UI TOOLBAR GEOMETRY
    # ------------------------------------------------------------------------
    
    # Define positions and dimensions for toolbar buttons
    y_pos = 0.93      # Vertical position (top of figure)
    h = 0.035         # Button height
    x_start = 0.045   # Starting x position
    w_btn = 0.055     # Width for action buttons
    w_cmap = 0.065    # Width for colormap buttons
    w_time = 0.050    # Width for time window buttons
    w_nav = 0.030     # Width for navigation buttons (◄◄, ►►)
    gap = 0.008       # Gap between buttons

    # Calculate x positions for each button
    p_load   = x_start
    p_select = p_load + w_btn + gap
    p_export = p_select + w_btn + gap
    p_seis   = p_export + w_btn + gap + 0.02  # Extra gap before colormaps
    p_gray   = p_seis + w_cmap + gap
    p_rain   = p_gray + w_cmap + gap
    
    # Time navigation buttons (new section)
    p_nav_back = p_rain + w_cmap + gap + 0.02  # Extra gap before navigation
    p_5sec   = p_nav_back + w_nav + gap
    p_10sec  = p_5sec + w_time + gap
    p_20sec  = p_10sec + w_time + gap
    p_60sec  = p_20sec + w_time + gap
    p_full   = p_60sec + w_time + gap
    p_nav_fwd = p_full + w_time + gap
    
    p_info   = 0.955 - w_btn - gap  # Aligned to the right

    # Create axes for each button
    ax_load     = plt.axes([p_load, y_pos, w_btn, h])
    ax_select   = plt.axes([p_select, y_pos, w_btn, h])
    ax_export   = plt.axes([p_export, y_pos, w_btn, h])
    ax_seis     = plt.axes([p_seis, y_pos, w_cmap, h])
    ax_gray     = plt.axes([p_gray, y_pos, w_cmap, h])
    ax_rain     = plt.axes([p_rain, y_pos, w_cmap, h])
    ax_nav_back = plt.axes([p_nav_back, y_pos, w_nav, h])
    ax_5sec     = plt.axes([p_5sec, y_pos, w_time, h])
    ax_10sec    = plt.axes([p_10sec, y_pos, w_time, h])
    ax_20sec    = plt.axes([p_20sec, y_pos, w_time, h])
    ax_60sec    = plt.axes([p_60sec, y_pos, w_time, h])
    ax_full     = plt.axes([p_full, y_pos, w_time, h])
    ax_nav_fwd  = plt.axes([p_nav_fwd, y_pos, w_nav, h])
    ax_info     = plt.axes([p_info, y_pos, w_btn, h])

    # Create button widgets
    btn_load     = Button(ax_load, "LOAD", color="#ADD8E6")      # Light blue
    btn_select   = Button(ax_select, "SELECT", color="#90ee90")  # Light green
    btn_export   = Button(ax_export, "EXPORT", color="#f08080")  # Light red
    btn_seis     = Button(ax_seis, "SEISMIC", color="#d3d3d3")   # Light gray
    btn_gray     = Button(ax_gray, "GRAY", color="#d3d3d3")      # Light gray
    btn_rain     = Button(ax_rain, "RAINBOW", color="#d3d3d3")   # Light gray
    btn_nav_back = Button(ax_nav_back, "◄◄", color="#FFD79A")    # Darker moccasin
    btn_5sec     = Button(ax_5sec, "5 sec", color="#FFE4B5")     # Moccasin
    btn_10sec    = Button(ax_10sec, "10 sec", color="#FFE4B5")   # Moccasin
    btn_20sec    = Button(ax_20sec, "20 sec", color="#FFE4B5")   # Moccasin
    btn_60sec    = Button(ax_60sec, "60 sec", color="#FFE4B5")   # Moccasin
    btn_full     = Button(ax_full, "Full", color="#FFE4B5")      # Moccasin
    btn_nav_fwd  = Button(ax_nav_fwd, "►►", color="#FFD79A")     # Darker moccasin
    btn_info     = Button(ax_info, "INFO", color="#f5f5f5")      # Very light gray

    # Style all button labels
    for b in (btn_load, btn_select, btn_export, btn_seis, btn_gray, btn_rain, btn_info):
        b.label.set_fontsize(11)
        b.label.set_fontweight("bold")

    # Style time navigation buttons with larger font
    for b in (btn_nav_back, btn_5sec, btn_10sec, btn_20sec, btn_60sec, btn_full, btn_nav_fwd):
        b.label.set_fontsize(13)
        b.label.set_fontweight("bold")
    
    # Different style for the "Info" button
    ax_info.spines['top'].set_visible(False)
    ax_info.spines['right'].set_visible(False)
    ax_info.spines['bottom'].set_visible(False)
    ax_info.spines['left'].set_visible(False)

    # ------------------------------------------------------------------------
    # BUTTON ACTION CALLBACKS
    # ------------------------------------------------------------------------
    
    # LOAD button - currently no action defined
    btn_load.on_clicked(lambda e: None)
    
    # SELECT button - activate annotation mode
    def activate_select(e):
        disable_toolbar_modes()
        selector.set_active(True)
        # Crosshair will only show if user presses SPACE
    
    btn_select.on_clicked(activate_select)
    
    # Colormap buttons - change the visualization colormap
    btn_seis.on_clicked(lambda e: (im.set_cmap("seismic"), fig.canvas.draw_idle()))
    btn_gray.on_clicked(lambda e: (im.set_cmap("gray"), fig.canvas.draw_idle()))
    btn_rain.on_clicked(lambda e: (im.set_cmap("nipy_spectral"), fig.canvas.draw_idle()))

    def set_time_window(window_size):
        """
        Set the time window size and adjust the plot view.
        
        Parameters:
        -----------
        window_size : float or None
            Size of the window in seconds (5, 10, 20, 60) or None for full view
        """
        nonlocal current_window_size, current_window_start
        
        current_window_size = window_size
        
        if window_size is None:
            # Full view
            ax.set_xlim(time_s[0], time_s[-1])
            current_window_start = time_s[0]
        else:
            # Set window to start from current position or beginning
            if current_window_start < time_s[0]:
                current_window_start = time_s[0]
            
            window_end = min(current_window_start + window_size, time_s[-1])
            
            # If window exceeds end, adjust start
            if window_end == time_s[-1] and (window_end - current_window_start) < window_size:
                current_window_start = max(time_s[0], time_s[-1] - window_size)
            
            ax.set_xlim(current_window_start, window_end)
        
        # Invalidate background when view changes
        crosshair_state['background'] = None
        fig.canvas.draw_idle()
    
    def navigate_time(direction):
        """
        Navigate forward or backward in time by the current window size.
        
        Parameters:
        -----------
        direction : str
            'forward' or 'backward'
        """
        nonlocal current_window_start
        
        if current_window_size is None:
            return  # No navigation in Full view
        
        if direction == 'forward':
            # Move forward by window size
            new_start = current_window_start + current_window_size
            if new_start + current_window_size > time_s[-1]:
                # Clamp to end
                new_start = max(time_s[0], time_s[-1] - current_window_size)
            current_window_start = new_start
        elif direction == 'backward':
            # Move backward by window size
            new_start = current_window_start - current_window_size
            if new_start < time_s[0]:
                new_start = time_s[0]
            current_window_start = new_start
        
        # Apply the new window
        window_end = min(current_window_start + current_window_size, time_s[-1])
        ax.set_xlim(current_window_start, window_end)
        
        # Invalidate background when view changes
        crosshair_state['background'] = None
        fig.canvas.draw_idle()
    
    # Time navigation button callbacks
    btn_5sec.on_clicked(lambda e: set_time_window(5))
    btn_10sec.on_clicked(lambda e: set_time_window(10))
    btn_20sec.on_clicked(lambda e: set_time_window(20))
    btn_60sec.on_clicked(lambda e: set_time_window(60))
    btn_full.on_clicked(lambda e: set_time_window(None))
    btn_nav_back.on_clicked(lambda e: navigate_time('backward'))
    btn_nav_fwd.on_clicked(lambda e: navigate_time('forward'))

    def show_info(event):
        """
        Display an information dialog with author details.
        This is triggered by clicking the INFO button.
        
        Parameters:
        -----------
        event : matplotlib event
            The button click event (unused)
        """
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle("About")
        msg_box.setText("YOLO4DAS Annotation Tool\n\n"
                       "Author: Sergio Morell Monzó\n"
                       "Email: sermomon@upv.es\n\n"
                       "Shortcuts:\n"
                       "- SPACE: Toggle crosshair on/off (in SELECT mode)")
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.exec_()

    btn_info.on_clicked(show_info)

    def export_events(event):
        """
        Export annotated events to files.
        This function can export:
        1. Annotations to CSV format (if annotations_output_format='csv')
           The CSV includes a 'comment' column with the free-text description.
        2. Processed data to NPZ format (if data_output_format='npz')
        
        The CSV export will OVERWRITE existing files with the same name.
        
        Parameters:
        -----------
        event : matplotlib event
            The button click event (unused)
        """
        if not events: return  # Nothing to export
        
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        base = os.path.splitext(os.path.basename(filename))[0] if filename else "das_export"
        if save_dir and os.path.isdir(save_dir):
            initial_dir = save_dir
        else:
            initial_dir = os.getcwd()
        if annotations_output_format == 'csv':
            default_csv_name = f"{base}_events.csv"
            if save_directly:
                csv_fname = os.path.join(initial_dir, default_csv_name)
                if os.path.exists(csv_fname):
                    reply = QtWidgets.QMessageBox.question(
                        None,
                        'Confirm Overwrite',
                        f'File "{os.path.basename(csv_fname)}" already exists.\n\nOverwrite?',
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        csv_fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                            None,
                            "Save CSV:",
                            os.path.join(initial_dir, default_csv_name),
                            "CSV (*.csv)"
                        )
                        if not csv_fname:
                            return  # User cancelled
            else:
                csv_fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                    None,
                    "Save CSV:",
                    os.path.join(initial_dir, default_csv_name),
                    "CSV (*.csv)"
                )
                if not csv_fname:
                    return  # User cancelled

            # Write CSV file – 'comment' column is appended after spatial metadata
            fieldnames = [
                "ID", "t0", "t1", "d0", "d1",
                "ti0", "ti1", "di0", "di1",
                "nx", "nt", "downsample",
                "start_datetime_utc", "comment"
            ]
            with open(csv_fname, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(events)
            print(f"Exported {len(events)} annotations to {csv_fname}")

        # Export processed data to NPZ
        if data_output_format == 'npz':
            if save_dir:
                npz_fname = os.path.join(save_dir, f"{base}_processed.npz")
            else:
                initial_path1 = f"{base}_processed.npz"
                npz_fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                    None, "Save NPZ:", initial_path1, "NPZ (*.npz)")
            
            if npz_fname:
                np.savez(npz_fname, 
                        tr=tr, 
                        dist_m=dist_m, 
                        time_s=time_s, 
                        fs_hz=fs_hz,
                        start_datetime_utc=start_datetime_utc.isoformat())
                
                print(f"Exported processed data to {npz_fname}")

    btn_export.on_clicked(export_events)

    # ------------------------------------------------------------------------
    # KEYBOARD SHORTCUTS
    # ------------------------------------------------------------------------
    
    def on_key(event):
        """
        Handle keyboard shortcuts for common actions.
        
        Shortcuts:
        ----------
        - 'l' : Trigger LOAD button
        - 's' : Activate SELECT mode
        - ' ' (SPACE) : Toggle crosshair on/off (when SELECT mode is active)
        - 'e' : Export annotations
        - 'Ctrl++' or 'Ctrl+=' : Zoom in (reduce view by 25%)
        - 'Ctrl+-' : Zoom out (expand view by 25%)
        - '5' : Set 5 second time window
        - 'Ctrl+1' : Set 10 second time window
        - 'Ctrl+2' : Set 20 second time window
        - 'Ctrl+6' : Set 60 second time window
        - 'f' : Full time view
        - 'Left Arrow' : Navigate backward
        - 'Right Arrow' : Navigate forward
        
        Parameters:
        -----------
        event : matplotlib.backend_bases.KeyEvent
            The keyboard event
        """
        # Main action shortcuts
        if event.key == 'l': 
            btn_load.on_clicked(lambda e: None)
            for func in btn_load.observers.values():
                func(event)
        elif event.key == 's': 
            disable_toolbar_modes()
            selector.set_active(True)
        elif event.key == 'e': 
            export_events(event)
        
        # SPACE: Toggle crosshair on/off (only when SELECT mode is active)
        elif event.key == ' ':
            if selector.active:
                # Toggle the crosshair state
                crosshair_state['enabled'] = not crosshair_state['enabled']
                
                if not crosshair_state['enabled']:
                    # Hide crosshair when disabled
                    crosshair_h.set_visible(False)
                    crosshair_v.set_visible(False)
                    crosshair_state['background'] = None
                    fig.canvas.draw_idle()
                else:
                    # Prepare background for blitting when enabled
                    crosshair_state['background'] = None
                    # Status message could be added here if desired
                    print(f"Crosshair: {'ON' if crosshair_state['enabled'] else 'OFF'}")
        
        # Time window shortcuts
        elif event.key == '5':
            set_time_window(5)
        elif event.key == 'ctrl+1':
            set_time_window(10)
        elif event.key == 'ctrl+2':
            set_time_window(20)
        elif event.key == 'ctrl+6':
            set_time_window(60)
        elif event.key == 'f':
            set_time_window(None)
        elif event.key == 'left':
            navigate_time('backward')
        elif event.key == 'right':
            navigate_time('forward')
        
        # Zoom in: reduce the view range by 25%
        elif event.key == 'ctrl++' or event.key == 'ctrl+=':
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * 0.75  # Reduce by 25%
            y_range = (ylim[1] - ylim[0]) * 0.75
            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            crosshair_state['background'] = None  # Invalidate background
            fig.canvas.draw_idle()
        
        # Zoom out: expand the view range by 25%
        elif event.key == 'ctrl+-':
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) * 1.25  # Expand by 25%
            y_range = (ylim[1] - ylim[0]) * 1.25
            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
            crosshair_state['background'] = None  # Invalidate background
            fig.canvas.draw_idle()

    # Connect keyboard event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # ------------------------------------------------------------------------
    # DISPLAY PLOT AND CREATE QT WIDGETS
    # ------------------------------------------------------------------------
    
    # Show the plot first (non-blocking) to ensure window is created and maximized
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure window is fully initialized
    
    # Now create Qt widgets for dynamic range controls
    # This is done after showing the plot to avoid startup delays
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    
    # Create container widget for all controls
    control_widget = QtWidgets.QWidget()
    control_layout = QtWidgets.QHBoxLayout(control_widget)
    control_layout.setContentsMargins(5, 5, 5, 5)
    control_layout.setSpacing(5)
    
    # ------------------------------------------------------------------------
    # AMPLITUDE RANGE CONTROLS (vMin, vMax)
    # ------------------------------------------------------------------------
    
    # Label and input for minimum amplitude
    label_vmin = QtWidgets.QLabel("vMin:")
    label_vmin.setStyleSheet("font-weight: bold; font-size: 10pt;")
    txt_vmin = QtWidgets.QLineEdit(f"{v_min:.6e}")
    txt_vmin.setFixedWidth(300)
    txt_vmin.setStyleSheet("font-size: 10pt; padding: 4px;")
    
    # Label and input for maximum amplitude
    label_vmax = QtWidgets.QLabel("vMax:")
    label_vmax.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 10px;")
    txt_vmax = QtWidgets.QLineEdit(f"{v_max:.6e}")
    txt_vmax.setFixedWidth(300)
    txt_vmax.setStyleSheet("font-size: 10pt; padding: 4px;")
    
    # Apply button for amplitude range
    btn_apply_qt = QtWidgets.QPushButton("Apply")
    btn_apply_qt.setFixedWidth(105)
    btn_apply_qt.setStyleSheet("""
        QPushButton {
            background-color: #FFD700;
            font-weight: bold;
            font-size: 10pt;
            padding: 5px;
            border: 1px solid #888;
            border-radius: 3px;
            margin-left: 10px;
        }
        QPushButton:hover {
            background-color: #FFC700;
        }
    """)
    
    # Visual separator between amplitude and distance controls
    separator = QtWidgets.QFrame()
    separator.setFrameShape(QtWidgets.QFrame.VLine)
    separator.setFrameShadow(QtWidgets.QFrame.Sunken)
    separator.setStyleSheet("background-color: #888;")
    
    # ------------------------------------------------------------------------
    # DISTANCE LIMIT CONTROLS (dMin, dMax)
    # ------------------------------------------------------------------------
    
    # Label and input for minimum distance
    label_dmin = QtWidgets.QLabel("dMin:")
    label_dmin.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 20px;")
    txt_dmin = QtWidgets.QLineEdit(f"{d_min:.2f}")
    txt_dmin.setFixedWidth(300)
    txt_dmin.setStyleSheet("font-size: 10pt; padding: 4px;")
    
    # Label and input for maximum distance
    label_dmax = QtWidgets.QLabel("dMax:")
    label_dmax.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 10px;")
    txt_dmax = QtWidgets.QLineEdit(f"{d_max:.2f}")
    txt_dmax.setFixedWidth(300)
    txt_dmax.setStyleSheet("font-size: 10pt; padding: 4px;")
    
    # Apply button for distance limits
    btn_apply_dist = QtWidgets.QPushButton("Apply")
    btn_apply_dist.setFixedWidth(105)
    btn_apply_dist.setStyleSheet("""
        QPushButton {
            background-color: #90EE90;
            font-weight: bold;
            font-size: 10pt;
            padding: 5px;
            border: 1px solid #888;
            border-radius: 3px;
            margin-left: 10px;
        }
        QPushButton:hover {
            background-color: #7FDD7F;
        }
    """)
    
    # Visual separator entre distance y y-axis controls
    separator2 = QtWidgets.QFrame()
    separator2.setFrameShape(QtWidgets.QFrame.VLine)
    separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
    separator2.setStyleSheet("background-color: #888;")
    
    # ------------------------------------------------------------------------
    # Y-AXIS LIMIT CONTROLS (yMin, yMax)
    # ------------------------------------------------------------------------
    
    # Label and input for minimum y-axis
    label_ymin = QtWidgets.QLabel("yMin:")
    label_ymin.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 20px;")
    txt_ymin = QtWidgets.QLineEdit(f"{ymin_val:.2f}")
    txt_ymin.setFixedWidth(300)
    txt_ymin.setStyleSheet("font-size: 10pt; padding: 4px;")
    
    # Label and input for maximum y-axis
    label_ymax = QtWidgets.QLabel("yMax:")
    label_ymax.setStyleSheet("font-weight: bold; font-size: 10pt; margin-left: 10px;")
    txt_ymax = QtWidgets.QLineEdit(f"{ymax_val:.2f}")
    txt_ymax.setFixedWidth(300)
    txt_ymax.setStyleSheet("font-size: 10pt; padding: 4px;")
    
    # Apply button for y-axis limits
    btn_apply_yaxis = QtWidgets.QPushButton("Apply")
    btn_apply_yaxis.setFixedWidth(105)
    btn_apply_yaxis.setStyleSheet("""
        QPushButton {
            background-color: #87CEEB;
            font-weight: bold;
            font-size: 10pt;
            padding: 5px;
            border: 1px solid #888;
            border-radius: 3px;
            margin-left: 10px;
        }
        QPushButton:hover {
            background-color: #76BDDA;
        }
    """)
    
    # Add all widgets to the horizontal layout
    control_layout.addWidget(label_vmin)
    control_layout.addWidget(txt_vmin)
    control_layout.addWidget(label_vmax)
    control_layout.addWidget(txt_vmax)
    control_layout.addWidget(btn_apply_qt)
    control_layout.addWidget(separator)
    control_layout.addWidget(label_dmin)
    control_layout.addWidget(txt_dmin)
    control_layout.addWidget(label_dmax)
    control_layout.addWidget(txt_dmax)
    control_layout.addWidget(btn_apply_dist)
    control_layout.addWidget(separator2)
    control_layout.addWidget(label_ymin)
    control_layout.addWidget(txt_ymin)
    control_layout.addWidget(label_ymax)
    control_layout.addWidget(txt_ymax)
    control_layout.addWidget(btn_apply_yaxis)
    control_layout.addStretch()  # Push everything to the left
    
    # Add the control widget to the matplotlib window as a toolbar
    canvas_widget = fig.canvas
    main_window = canvas_widget.parent()
    if main_window and hasattr(main_window, 'addToolBar'):
        toolbar = QtWidgets.QToolBar()
        toolbar.addWidget(control_widget)
        main_window.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)
    else:
        # Fallback: create a floating dialog if main window is not available
        control_widget.setWindowTitle("Dynamic Range Control")
        control_widget.show()
    
    # ------------------------------------------------------------------------
    # CONTROL CALLBACKS
    # ------------------------------------------------------------------------
    
    def update_clim():
        """
        Update the colormap intensity limits from user input.
        This adjusts the amplitude range displayed in the visualization.
        Invalid inputs are silently ignored.
        """
        try:
            vmin_val = float(txt_vmin.text())
            vmax_val = float(txt_vmax.text())
            im.set_clim(vmin_val, vmax_val)
            crosshair_state['background'] = None  # Invalidate background
            fig.canvas.draw_idle()
        except ValueError:
            pass  # Ignore invalid input

    def update_distance_limits():
        """
        Update the white horizontal lines showing distance limits.
        This allows users to adjust the region of interest markers.
        Invalid inputs are silently ignored.
        """
        try:
            dmin_val = float(txt_dmin.text())
            dmax_val = float(txt_dmax.text())
            
            # Update line positions
            line_dmin.set_ydata([dmin_val, dmin_val])
            line_dmax.set_ydata([dmax_val, dmax_val])
            
            crosshair_state['background'] = None  # Invalidate background
            fig.canvas.draw_idle()
        except ValueError:
            pass  # Ignore invalid input

    def update_yaxis_limits():
        """
        Update the y-axis (distance) display limits from user input.
        This changes the visible range but doesn't modify the data array,
        ensuring annotations remain coherent with original coordinates.
        Invalid inputs are silently ignored.
        """
        try:
            ymin_val = float(txt_ymin.text())
            ymax_val = float(txt_ymax.text())
            
            # Validate that limits are within data range
            if ymin_val < dist_m[0]:
                ymin_val = dist_m[0]
                txt_ymin.setText(f"{ymin_val:.2f}")
            if ymax_val > dist_m[-1]:
                ymax_val = dist_m[-1]
                txt_ymax.setText(f"{ymax_val:.2f}")
            
            # Update y-axis limits
            ax.set_ylim(ymin_val, ymax_val)
            crosshair_state['background'] = None  # Invalidate background
            fig.canvas.draw_idle()
        except ValueError:
            pass  # Ignore invalid input

    # Connect button callbacks
    btn_apply_qt.clicked.connect(update_clim)
    btn_apply_dist.clicked.connect(update_distance_limits)
    btn_apply_yaxis.clicked.connect(update_yaxis_limits)
    
    # Connect Enter key to apply changes
    txt_vmin.returnPressed.connect(update_clim)
    txt_vmax.returnPressed.connect(update_clim)
    txt_dmin.returnPressed.connect(update_distance_limits)
    txt_dmax.returnPressed.connect(update_distance_limits)
    txt_ymin.returnPressed.connect(update_yaxis_limits)
    txt_ymax.returnPressed.connect(update_yaxis_limits)
    
    # Block until window is closed (blocking show)
    plt.show()

def das_csv_to_yolo(
    csv_path: str,
    output_dir: str,
    group_by: str = None,
    class_offset: int = 0,
) -> None:
    """
    Convert a YOLO4DAS annotation CSV (v2) to YOLO bounding-box label files.

    One .txt file is generated per group (defined by ``group_by``).
    Bounding-box indices di0/di1 must already be corrected to the original
    (pre-downsampling) coordinate space. nx and nt are read from each row.

    Axis mapping:
        X (horizontal) → time     (ti0, ti1 normalised over nt)
        Y (vertical)   → distance (di0, di1 normalised over nx)

    Parameters
    ----------
    csv_path : str
        Path to the CSV file exported by the annotation tool.
    output_dir : str
        Directory where the YOLO .txt files will be saved.
    group_by : str
        CSV column used to split rows into separate label files.
        Pass None to write everything into a single file.
    class_offset : int
        Subtracted from the CSV ID column to get zero-based YOLO class.
        Use 1 if annotation IDs start at 1 (default), 0 if already zero-based.
    """
    import csv
    import numpy as np
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load all rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty, nothing to convert.")
        return

    # Group rows
    groups = {}
    for row in rows:
        key = row.get(group_by, Path(csv_path).stem) if group_by else Path(csv_path).stem
        groups.setdefault(key, []).append(row)

    for group_key, group_rows in groups.items():
        lines = []
        for row in group_rows:
            class_id = int(row["ID"]) - class_offset
            ti0, ti1 = int(row["ti0"]), int(row["ti1"])
            di0, di1 = int(row["di0"]), int(row["di1"])  # already corrected for downsampling
            nt = int(row["nt"])
            nx = int(row["nx"])  # original (pre-downsampling) number of channels

            # Normalise to [0, 1]
            x_center = ((ti0 + ti1) / 2.0) / nt
            y_center = ((di0 + di1) / 2.0) / nx
            width    = (ti1 - ti0) / nt
            height   = (di1 - di0) / nx

            # Clamp to valid range
            x_center = np.clip(x_center, 0.0, 1.0)
            y_center = np.clip(y_center, 0.0, 1.0)
            width    = np.clip(width,    0.0, 1.0)
            height   = np.clip(height,   0.0, 1.0)

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Sanitise group key for use as filename
        out_path = Path(output_dir) / (Path(csv_path).stem + ".txt")
        out_path.write_text("\n".join(lines) + "\n")
        print(f"{out_path.name}  —  {len(lines)} box(es)")