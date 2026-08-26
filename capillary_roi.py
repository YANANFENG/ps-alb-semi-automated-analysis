
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def select_and_track_capillary(
    registered_stack,
    to_uint8_func,
    tracking_radius=12,
    default_radius=11,
    preview_margin=50,
    window_name="Click capillary-loop centre"
):
    """
    Select a capillary-loop centre, preview and freely choose
    a circular measurement ROI radius, then track the loop
    through the registered image sequence using OpenCV CSRT.

    Parameters
    ----------
    registered_stack : ndarray
        Registered structural image stack, e.g. R18.
        Expected shape: (frames, height, width).

    to_uint8_func : function
        Function that converts a microscopy frame to uint8
        for OpenCV display.

    tracking_radius : int, default=12
        Half-width of the square region used by CSRT tracking.
        This is independent of the measurement ROI radius.

    default_radius : int, default=11
        Initial measurement ROI radius shown to the user.

    preview_margin : int, default=50
        Zoom margin around the selected capillary loop.

    window_name : str
        Name of the OpenCV selection window.

    Returns
    -------
    selected_centre : tuple
        Initial selected centre as (x, y).

    roi_radius : int
        Final user-selected circular measurement ROI radius.

    tracked_centres : ndarray
        Tracked (x, y) centre coordinates for all frames.

    tracking_success : ndarray
        Boolean tracking status for all frames.

    tracked_boxes : list
        CSRT bounding boxes for all frames.
    """

    registered_stack = np.asarray(registered_stack)

    if registered_stack.ndim != 3:
        raise ValueError(
            "registered_stack must have shape "
            "(frames, height, width)."
        )

    # ========================================================
    # 1. Select capillary-loop centre
    # ========================================================

    selected_centre = None
    selection_done = False

    display_frame = to_uint8_func(
        registered_stack[0]
    )

    display_bgr = cv2.cvtColor(
        display_frame,
        cv2.COLOR_GRAY2BGR
    )

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):
        nonlocal selected_centre
        nonlocal selection_done

        if event == cv2.EVENT_LBUTTONDOWN:
            selected_centre = (
                int(x),
                int(y)
            )
            selection_done = True

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        window_name,
        display_bgr
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print(
        "Left-click once on the centre "
        "of the capillary loop."
    )

    while not selection_done:

        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Esc
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if selected_centre is None:
        raise RuntimeError(
            "ROI selection cancelled."
        )

    centre_x, centre_y = (
        selected_centre
    )

    print(
        "Selected centre:",
        centre_x,
        centre_y
    )

    # ========================================================
    # 2. Show initial ROI preview
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(
        registered_stack[0],
        cmap="gray"
    )

    initial_circle = Circle(
        (
            centre_x,
            centre_y
        ),
        default_radius,
        fill=False,
        linewidth=2
    )

    ax.add_patch(
        initial_circle
    )

    initial_margin = max(
        preview_margin,
        default_radius * 4
    )

    ax.set_xlim(
        centre_x - initial_margin,
        centre_x + initial_margin
    )

    ax.set_ylim(
        centre_y + initial_margin,
        centre_y - initial_margin
    )

    ax.set_title(
        f"Initial ROI preview: "
        f"radius = {default_radius}px"
    )

    ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ========================================================
    # 3. Let user freely choose measurement ROI radius
    # ========================================================

    user_input = input(
        f"Enter measurement ROI radius "
        f"[default {default_radius}]: "
    ).strip()

    if user_input == "":
        roi_radius = int(
            default_radius
        )

    else:
        try:
            roi_radius = int(
                user_input
            )

        except ValueError:
            raise ValueError(
                "ROI radius must be an integer."
            )

    if roi_radius <= 0:
        raise ValueError(
            "ROI radius must be greater than zero."
        )

    print(
        "Confirmed measurement ROI radius:",
        roi_radius
    )

    print(
        "Tracking radius:",
        tracking_radius
    )

    # ========================================================
    # 4. Show final ROI preview
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.imshow(
        registered_stack[0],
        cmap="gray"
    )

    final_circle = Circle(
        (
            centre_x,
            centre_y
        ),
        roi_radius,
        fill=False,
        linewidth=2
    )

    ax.add_patch(
        final_circle
    )

    final_margin = max(
        preview_margin,
        roi_radius * 4
    )

    ax.set_xlim(
        centre_x - final_margin,
        centre_x + final_margin
    )

    ax.set_ylim(
        centre_y + final_margin,
        centre_y - final_margin
    )

    ax.set_title(
        f"Confirmed ROI: "
        f"radius = {roi_radius}px"
    )

    ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ========================================================
    # 5. Build independent CSRT tracking box
    # ========================================================

    bbox = (
        int(
            centre_x
            - tracking_radius
        ),
        int(
            centre_y
            - tracking_radius
        ),
        int(
            2 * tracking_radius
        ),
        int(
            2 * tracking_radius
        )
    )

    # ========================================================
    # 6. Initialise CSRT tracker
    # ========================================================

    tracker = (
        cv2.TrackerCSRT_create()
    )

    first_frame_bgr = cv2.cvtColor(
        to_uint8_func(
            registered_stack[0]
        ),
        cv2.COLOR_GRAY2BGR
    )

    tracker.init(
        first_frame_bgr,
        bbox
    )

    tracked_boxes = [
        bbox
    ]

    tracking_success = [
        True
    ]

    # ========================================================
    # 7. Track remaining frames
    # ========================================================

    for frame_index in range(
        1,
        len(registered_stack)
    ):

        frame_bgr = cv2.cvtColor(
            to_uint8_func(
                registered_stack[
                    frame_index
                ]
            ),
            cv2.COLOR_GRAY2BGR
        )

        ok, box = tracker.update(
            frame_bgr
        )

        tracking_success.append(
            bool(ok)
        )

        if ok:
            tracked_boxes.append(
                box
            )

        else:
            tracked_boxes.append(
                (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan
                )
            )

            print(
                f"Tracking failed at frame "
                f"{frame_index + 1}"
            )

    # ========================================================
    # 8. Convert tracking boxes to centres
    # ========================================================

    tracked_centres = []

    for box in tracked_boxes:

        x, y, w, h = box

        if np.isfinite(x):

            tracked_centres.append(
                (
                    x + w / 2,
                    y + h / 2
                )
            )

        else:

            tracked_centres.append(
                (
                    np.nan,
                    np.nan
                )
            )

    tracked_centres = np.asarray(
        tracked_centres,
        dtype=float
    )

    tracking_success = np.asarray(
        tracking_success,
        dtype=bool
    )

    print(
        f"Tracking success: "
        f"{tracking_success.sum()} / "
        f"{len(registered_stack)}"
    )

    return (
        selected_centre,
        roi_radius,
        tracked_centres,
        tracking_success,
        tracked_boxes
    )


def extract_af488_curve(
    af488_registered,
    tracked_centres,
    roi_radius,
    frame_interval=4.0
):
    """
    Extract mean AF488 fluorescence intensity from the tracked
    circular capillary-loop ROI.

    Parameters
    ----------
    af488_registered : ndarray
        Registered AF488 fluorescence image stack.

    tracked_centres : ndarray
        Tracked (x, y) centre coordinates.

    roi_radius : int or float
        Circular measurement ROI radius.

    frame_interval : float, default=4.0
        Time interval between consecutive frames in seconds.

    Returns
    -------
    time_seconds : ndarray
        Time coordinate for each frame.

    intensity : ndarray
        Mean AF488 fluorescence intensity for each frame.
    """

    af488_registered = np.asarray(
        af488_registered
    )

    tracked_centres = np.asarray(
        tracked_centres,
        dtype=float
    )

    if af488_registered.ndim != 3:
        raise ValueError(
            "af488_registered must have shape "
            "(frames, height, width)."
        )

    if (
        len(af488_registered)
        != len(tracked_centres)
    ):
        raise ValueError(
            "AF488 stack and tracked centres "
            "must contain the same number of frames."
        )

    height, width = (
        af488_registered[0].shape
    )

    Y, X = np.ogrid[
        :height,
        :width
    ]

    intensity = []

    for frame_index, (
        centre_x,
        centre_y
    ) in enumerate(
        tracked_centres
    ):

        if (
            not np.isfinite(
                centre_x
            )
            or
            not np.isfinite(
                centre_y
            )
        ):

            intensity.append(
                np.nan
            )

            continue

        mask = (
            (
                X - centre_x
            ) ** 2
            +
            (
                Y - centre_y
            ) ** 2
            <= roi_radius ** 2
        )

        mean_intensity = np.mean(
            af488_registered[
                frame_index
            ][mask]
        )

        intensity.append(
            mean_intensity
        )

    intensity = np.asarray(
        intensity,
        dtype=float
    )

    time_seconds = (
        np.arange(
            len(intensity)
        )
        * float(
            frame_interval
        )
    )

    return (
        time_seconds,
        intensity
    )
