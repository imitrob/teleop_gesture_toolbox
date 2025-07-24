

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
from typing import List
import time

def show_pointing_experiment_plot(
        T: int = 21,
        object_names: List[str] = ["cube", "sphere", "cylinder"],
        descriptions: dict[float, str] = {
            3.0: '"red cube"',
            8.0: '"big sphere"',
            14.0: '"stone cylinder"',
        },
        object_positions = np.array([[0.20, 0.15],   # cube
                        [0.75, 0.40],   # sphere
                        [0.45, 0.80]]),  # cylinder
        target_pointing = np.array([
            [0.0,0.0, 2.0],
            [0.0,0.0, 2.0],
            [0.0,0.0, 2.0],
            [0.0,0.0, 2.0],
            [1.0,0.0, 1.0],
            [1.0,0.0, 1.0],
            [1.0,0.0, 1.0],
            [1.0,0.0, 1.0],
            [1.0,0.0, 1.0],
            [1.0,0.1, 0.8],
            [1.0,0.1, 0.8],
            [1.0,0.1, 0.8],
            [1.0,0.1, 0.8],
            [1.0,0.5, 1.0],
            [1.0,0.5, 1.0],
            [1.0,0.5, 1.0],
            [1.0,0.5, 1.0],
            [1.0,1.0, 0.01],
            [1.0,1.0, 0.01],
            [1.0,1.0, 0.01],
            [1.0,1.0, 0.01],
        ]),
        valid_objects = [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
            [False, False, True],
            [False, False, True],
            [False, False, True],
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ],
        pointing_likelihoods = np.array([
            [0.0,0.0,0.0],
            [0.0,0.0,0.2],
            [0.0,0.0,0.2],
            [0.0,0.0,0.2],
            [0.0,0.5,0.2],
            [0.0,0.5,0.2],
            [0.0,0.5,0.2],
            [0.0,0.5,0.2],
            [1.0,0.5,0.2],
            [1.0,0.5,0.2],
            [1.0,0.5,0.2],
            [1.0,0.5,0.2],
            [1.0,0.5,0.2],
            [1.0,0.5,0.2],
            [1.0,0.5,0.2],
            [0.4,0.5,0.2],
            [0.4,0.5,0.2],
            [0.4,0.5,0.2],
            [0.4,0.5,0.2],
            [0.4,0.5,0.2],
            [0.4,0.5,10.0],
        ]),
        time_start: float = 0.0
    ):
    """
    Interactive visualiser for an object-selection experiment.

    Left  : per-object selection probabilities + text-event markers  
    Right : 2-D scene layout; objects dim/brighten with probability, plus a
            “pointing target” circle whose size reflects pointing variance.
    Slider: scrub through time; clicking on the left plot also jumps time
    """
    if T == 0: 
        print("No pointing data")
        return
    # ---------------------------------------------------------------------
    # 1) Example data ------------------------------------------------------
    # ---------------------------------------------------------------------
    time        = np.arange(T)
    n_obj       = len(object_names)
    probs = pointing_likelihoods / pointing_likelihoods.sum(axis=1, keepdims=True)  # rows sum to 1

    text_times = []
    text_descriptions = []
    for k,v in descriptions.items():
        if k - time_start > 0.0:
            text_times.append(10 * (k - time_start))
            text_descriptions.append(v)

    positions = object_positions
    # Simulated pointing targets (centre) + variance (radius)
    ptr_centre   = np.array(target_pointing)[:,0:2]
    ptr_variance = 0.05 * np.ones((T))

    # ---------------------------------------------------------------------
    # 2) Create a single figure with 2 sub-plots + slider ------------------
    # ---------------------------------------------------------------------
    plt.style.use("default")   # explicitly NOT seaborn

    FIG_W, FIG_H = 16, 7
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Axes:  [left plot] 70 % width, full height minus slider
    #        [right]     25 % width, full height minus slider
    #        [slider]    full width, 10 % height at bottom
    left  = fig.add_axes([0.07, 0.25, 0.60, 0.68])   # [x0,y0,w,h]
    right = fig.add_axes([0.72, 0.25, 0.23, 0.68])
    s_ax  = fig.add_axes([0.07, 0.08, 0.88, 0.08])

    # ---------------------------------------------------------------------
    # 3) Left plot: probabilities & text markers --------------------------
    # ---------------------------------------------------------------------
    for i, name in enumerate(object_names):
        left.plot(time, probs[:, i], label=name, lw=1.8)

    left.set_xlim(time[0], time[-1])
    left.set_ylim(0, 1)
    left.set_xlabel("Samples (10smp/s)")
    left.set_ylabel("probability")
    left.set_title("Selection probability over time")
    vline = left.axvline(0, color='k', lw=1.2)
    left.legend(loc="upper right")

    # Text-event annotations above the axes (axes-fraction coords)
    for t, txt in zip(text_times, text_descriptions):
        left.annotate(
            txt,
            xy=(t, 0.7), xycoords=('data', 'axes fraction'),
            rotation=80, ha="left", va="bottom", fontsize=10
        )

    # ---------------------------------------------------------------------
    # 4) Right plot: scene map + pointing target --------------------------
    # ---------------------------------------------------------------------
    scatter_handles = []
    for (x, y), name in zip(positions, object_names):
        sc = right.scatter(x, y, s=220)
        right.text(x, y + 0.035, name, ha="center")
        scatter_handles.append(sc)

    # Pointing target as a Circle patch
    ptr_circle = Circle(
        xy=ptr_centre[0],
        radius=ptr_variance[0],
        edgecolor="blue", facecolor=(0, 0, 1, 0.15), lw=2
    )
    right.add_patch(ptr_circle)

    right.set_xlim(-0.3, 1.07)
    right.set_ylim(-0.4, 0.4)
    right.set_xlabel("x")
    right.set_ylabel("y")
    right.set_title("Scene objects (bright = viable)")

    # Force equal aspect so circles look round
    right.set_aspect("equal", adjustable="box")

    # ---------------------------------------------------------------------
    # 5) Slider + interaction ---------------------------------------------
    # ---------------------------------------------------------------------
    slider = Slider(s_ax, "time", valmin=0, valmax=T-1, valinit=0,
                    valstep=1, color="0.5")

    def refresh(t_idx):
        """Update visuals to reflect new time index."""
        t_idx = int(t_idx)
        vline.set_xdata([t_idx, t_idx])

        # Fade / brighten objects
        for n, (p, sc) in enumerate(zip(probs[t_idx], scatter_handles)):
            sc.set_alpha(1.0 if valid_objects[t_idx][n] else 0.2)

        # Update pointing circle
        ptr_circle.center = ptr_centre[t_idx]
        ptr_circle.radius = ptr_variance[t_idx]

        fig.canvas.draw_idle()

    slider.on_changed(refresh)

    # Clicking on left plot moves slider
    def on_click(event):
        if event.inaxes is left and event.xdata is not None:
            t_idx = int(np.clip(round(event.xdata), 0, T-1))
            slider.set_val(t_idx)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ---------------------------------------------------------------------
    # 6) Show GUI ----------------------------------------------------------
    # ---------------------------------------------------------------------
    plt.show()
