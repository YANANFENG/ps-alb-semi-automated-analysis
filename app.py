import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from registration_utils import (
    analyse_nd2_drift,
    register_stack,
    estimate_translation,
)


st.set_page_config(
    page_title="Semi-automated Ps′alb Analysis",
    layout="wide"
)

st.title("Semi-automated Ps′alb Analysis")
st.caption(
    "Prototype interface for glomerular capillary permeability "
    "analysis from time-lapse fluorescence microscopy."
)


def to_uint8(frame):
    frame = np.asarray(frame, dtype=np.float32)
    low, high = np.percentile(frame, [1, 99])

    if high <= low:
        return np.zeros_like(frame, dtype=np.uint8)

    frame = np.clip(frame, low, high)
    frame = (frame - low) / (high - low) * 255
    return frame.astype(np.uint8)


def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy, "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is unavailable. "
        "Install opencv-contrib-python."
    )


def track_from_centre(
    registered_stack,
    centre_x,
    centre_y,
    tracking_radius
):
    bbox = (
        int(centre_x - tracking_radius),
        int(centre_y - tracking_radius),
        int(2 * tracking_radius),
        int(2 * tracking_radius),
    )

    tracker = create_csrt_tracker()

    first_frame = cv2.cvtColor(
        to_uint8(registered_stack[0]),
        cv2.COLOR_GRAY2BGR
    )
    tracker.init(first_frame, bbox)

    tracked_centres = [
        (float(centre_x), float(centre_y))
    ]
    tracking_success = [True]

    for frame_index in range(1, len(registered_stack)):
        frame = cv2.cvtColor(
            to_uint8(registered_stack[frame_index]),
            cv2.COLOR_GRAY2BGR
        )

        ok, box = tracker.update(frame)
        tracking_success.append(bool(ok))

        if ok:
            x, y, w, h = box
            tracked_centres.append(
                (x + w / 2, y + h / 2)
            )
        else:
            tracked_centres.append(
                (np.nan, np.nan)
            )

    return (
        np.asarray(tracked_centres, dtype=float),
        np.asarray(tracking_success, dtype=bool),
    )


def extract_intensity(
    fluorescence_stack,
    tracked_centres,
    roi_radius
):
    height, width = fluorescence_stack[0].shape
    yy, xx = np.ogrid[:height, :width]

    intensity = []

    for frame_index, centre in enumerate(tracked_centres):
        centre_x, centre_y = centre

        if (
            not np.isfinite(centre_x)
            or not np.isfinite(centre_y)
        ):
            intensity.append(np.nan)
            continue

        mask = (
            (xx - centre_x) ** 2
            + (yy - centre_y) ** 2
            <= roi_radius ** 2
        )

        intensity.append(
            np.mean(
                fluorescence_stack[frame_index][mask]
            )
        )

    return np.asarray(intensity, dtype=float)


def fit_log_decay(
    time_seconds,
    intensity,
    start_time,
    end_time
):
    valid = (
        (time_seconds >= start_time)
        & (time_seconds <= end_time)
        & np.isfinite(intensity)
        & (intensity > 0)
    )

    x = time_seconds[valid]
    y_raw = intensity[valid]

    if len(x) < 3:
        raise ValueError(
            "At least three valid points are required "
            "for decay fitting."
        )

    y_log = np.log(y_raw)
    slope, intercept = np.polyfit(
        x,
        y_log,
        1
    )

    predicted_log = slope * x + intercept

    ss_res = np.sum(
        (y_log - predicted_log) ** 2
    )
    ss_tot = np.sum(
        (y_log - np.mean(y_log)) ** 2
    )

    r_squared = (
        np.nan
        if ss_tot == 0
        else 1 - ss_res / ss_tot
    )

    return {
        "x": x,
        "raw": y_raw,
        "log": y_log,
        "predicted_log": predicted_log,
        "slope": slope,
        "k": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def calculate_residual_drift(registered_stack):
    reference = registered_stack[0]

    residual_x = [0.0]
    residual_y = [0.0]

    for frame_index in range(1, len(registered_stack)):
        result = estimate_translation(
            reference,
            registered_stack[frame_index]
        )
        residual_x.append(result["x"])
        residual_y.append(result["y"])

    residual_x = np.asarray(
        residual_x,
        dtype=float
    )
    residual_y = np.asarray(
        residual_y,
        dtype=float
    )

    residual_magnitude = np.hypot(
        residual_x,
        residual_y
    )

    return residual_magnitude


def distance_between_points(point_1, point_2):
    return float(
        np.hypot(
            point_2[0] - point_1[0],
            point_2[1] - point_1[1]
        )
    )


def make_measurement_image(
    crop_uint8,
    points,
    display_scale=4
):
    base = Image.fromarray(
        crop_uint8
    ).convert("RGB")

    width, height = base.size

    enlarged = base.resize(
        (
            width * display_scale,
            height * display_scale,
        ),
        Image.Resampling.BICUBIC
    )

    draw = ImageDraw.Draw(enlarged)

    display_points = [
        (
            int(point[0] * display_scale),
            int(point[1] * display_scale),
        )
        for point in points
    ]

    marker_radius = 6

    for x, y in display_points:
        draw.ellipse(
            (
                x - marker_radius,
                y - marker_radius,
                x + marker_radius,
                y + marker_radius,
            ),
            outline="red",
            width=3,
        )

    if len(display_points) == 2:
        draw.line(
            display_points,
            fill="red",
            width=4,
        )

    return enlarged


def register_measurement_click(
    state_key,
    last_click_key,
    click,
    display_scale
):
    if click is None:
        return

    point = (
        float(click["x"]) / display_scale,
        float(click["y"]) / display_scale,
    )

    last_click = st.session_state.get(
        last_click_key
    )

    if (
        last_click is not None
        and np.allclose(point, last_click)
    ):
        return

    st.session_state[last_click_key] = point

    if len(st.session_state[state_key]) < 2:
        st.session_state[state_key].append(point)


defaults = {
    "nd2_path": None,
    "data": None,
    "transforms": None,
    "drift_df": None,
    "r18_registered": None,
    "af488_registered": None,
    "brightfield_registered": None,
    "max_residual_drift": None,
    "selected_centre": None,
    "roi_radius": 11,
    "tracked_centres": None,
    "tracking_success": None,
    "tracking_radius": None,
    "intensity": None,
    "frame_interval": 4.0,
    "fit_result": None,
    "pixel_size_um": 0.1838527729,
    "d1_points": [],
    "d2_points": [],
    "d1_last_click": None,
    "d2_last_click": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = (
            list(value)
            if isinstance(value, list)
            else value
        )


st.sidebar.header("Analysis workflow")
st.sidebar.markdown(
    """
    **1.** Upload ND2  
    **2.** Image registration  
    **3.** Select capillary ROI  
    **4.** ROI tracking  
    **5.** Fluorescence decay  
    **6.** Measure D1/D2 and calculate Ps′alb
    """
)
st.sidebar.info(
    "Registration: SimpleITK\n\n"
    "Local tracking: OpenCV CSRT"
)


st.header("1. Upload ND2 recording")

uploaded_file = st.file_uploader(
    "Select a time-lapse ND2 microscopy recording",
    type=["nd2"]
)

if uploaded_file is not None:
    file_signature = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        st.session_state.get("file_signature")
        != file_signature
    ):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".nd2"
        ) as tmp:
            tmp.write(
                uploaded_file.getbuffer()
            )
            st.session_state.nd2_path = tmp.name

        st.session_state.file_signature = file_signature

        for key in [
            "data",
            "transforms",
            "drift_df",
            "r18_registered",
            "af488_registered",
            "brightfield_registered",
            "max_residual_drift",
            "selected_centre",
            "tracked_centres",
            "tracking_success",
            "tracking_radius",
            "intensity",
            "fit_result",
        ]:
            st.session_state[key] = None

        st.session_state.roi_radius = 11
        st.session_state.d1_points = []
        st.session_state.d2_points = []
        st.session_state.d1_last_click = None
        st.session_state.d2_last_click = None

    st.success(
        f"Loaded: {uploaded_file.name}"
    )


st.header(
    "2. Image registration and motion correction"
)

st.write(
    "Global translational motion is estimated from the R18 "
    "structural channel using SimpleITK. The same estimated "
    "transformations are then applied to AF488 and "
    "transmitted-light frames."
)

if st.session_state.nd2_path is None:
    st.info(
        "Upload an ND2 recording to begin."
    )
else:
    if st.button(
        "Run image registration",
        type="primary"
    ):
        with st.spinner(
            "Running SimpleITK translation registration..."
        ):
            try:
                (
                    data,
                    registration_stack,
                    transforms,
                    drift_df,
                ) = analyse_nd2_drift(
                    st.session_state.nd2_path,
                    registration_channel=1,
                )

                af488_stack = (
                    data
                    .isel(C=0)
                    .compute()
                    .values
                )

                r18_stack = (
                    data
                    .isel(C=1)
                    .compute()
                    .values
                )

                if (
                    "C" in data.dims
                    and data.sizes["C"] > 2
                ):
                    brightfield_stack = (
                        data
                        .isel(C=2)
                        .compute()
                        .values
                    )
                else:
                    brightfield_stack = None

                r18_registered = register_stack(
                    r18_stack,
                    transforms,
                )

                af488_registered = register_stack(
                    af488_stack,
                    transforms,
                )

                if brightfield_stack is not None:
                    brightfield_registered = (
                        register_stack(
                            brightfield_stack,
                            transforms,
                        )
                    )
                else:
                    brightfield_registered = None

                residual_magnitude = (
                    calculate_residual_drift(
                        r18_registered
                    )
                )

                st.session_state.data = data
                st.session_state.transforms = transforms
                st.session_state.drift_df = drift_df
                st.session_state.r18_registered = (
                    r18_registered
                )
                st.session_state.af488_registered = (
                    af488_registered
                )
                st.session_state.brightfield_registered = (
                    brightfield_registered
                )
                st.session_state.max_residual_drift = float(
                    np.nanmax(residual_magnitude)
                )

                st.success(
                    "Registration completed."
                )

            except Exception as error:
                st.error(
                    f"Registration failed: {error}"
                )


if st.session_state.drift_df is not None:
    st.subheader(
        "Registration quality control"
    )

    drift_df = st.session_state.drift_df

    max_estimated_drift = float(
        np.nanmax(
            drift_df["magnitude"].to_numpy()
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Maximum estimated drift",
            f"{max_estimated_drift:.2f} pixels"
        )

    with col2:
        st.metric(
            "Maximum residual drift",
            (
                f"{st.session_state.max_residual_drift:.3f} "
                "pixels"
            )
        )


st.header(
    "3. Capillary-loop ROI selection"
)

if st.session_state.r18_registered is None:
    st.info(
        "Complete image registration first."
    )
else:
    r18_registered = (
        st.session_state.r18_registered
    )

    st.write(
        "Click once on the centre of a suitable capillary loop "
        "in registered Frame 1."
    )

    display_frame = to_uint8(
        r18_registered[0]
    )

    display_image = Image.fromarray(
        display_frame
    )

    click = streamlit_image_coordinates(
        display_image,
        key="capillary_click"
    )

    if click is not None:
        new_centre = (
            int(click["x"]),
            int(click["y"]),
        )

        if (
            st.session_state.selected_centre
            != new_centre
        ):
            st.session_state.selected_centre = (
                new_centre
            )
            st.session_state.tracked_centres = None
            st.session_state.tracking_success = None
            st.session_state.intensity = None
            st.session_state.fit_result = None
            st.session_state.d1_points = []
            st.session_state.d2_points = []

    if (
        st.session_state.selected_centre
        is not None
    ):
        centre_x, centre_y = (
            st.session_state.selected_centre
        )

        st.success(
            f"Selected ROI centre: "
            f"x = {centre_x}, y = {centre_y}"
        )

        roi_radius = st.slider(
            "Measurement ROI radius (pixels)",
            min_value=3,
            max_value=30,
            value=int(
                st.session_state.roi_radius
            ),
            step=1
        )

        st.session_state.roi_radius = (
            roi_radius
        )

        fig, ax = plt.subplots(
            figsize=(5, 5)
        )

        ax.imshow(
            r18_registered[0],
            cmap="gray"
        )

        ax.add_patch(
            Circle(
                (
                    centre_x,
                    centre_y
                ),
                roi_radius,
                fill=False,
                linewidth=2,
            )
        )

        preview_margin = max(
            40,
            roi_radius * 4
        )

        ax.set_xlim(
            centre_x - preview_margin,
            centre_x + preview_margin
        )
        ax.set_ylim(
            centre_y + preview_margin,
            centre_y - preview_margin
        )
        ax.set_title(
            f"Measurement ROI "
            f"(radius = {roi_radius}px)"
        )
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)


st.header(
    "4. Capillary-loop tracking"
)

if st.session_state.selected_centre is None:
    st.info(
        "Select a capillary-loop ROI first."
    )
else:
    tracking_radius = st.selectbox(
        "Tracking radius (pixels)",
        options=[12, 14, 16, 18],
        index=3,
        help=(
            "The tracking radius defines the surrounding image "
            "region used by the CSRT tracker. It is independent "
            "of the measurement ROI radius."
        )
    )

    if st.button(
        "Run ROI tracking"
    ):
        centre_x, centre_y = (
            st.session_state.selected_centre
        )

        with st.spinner(
            "Tracking capillary loop..."
        ):
            try:
                (
                    tracked_centres,
                    tracking_success,
                ) = track_from_centre(
                    st.session_state.r18_registered,
                    centre_x,
                    centre_y,
                    tracking_radius,
                )

                st.session_state.tracked_centres = (
                    tracked_centres
                )
                st.session_state.tracking_success = (
                    tracking_success
                )
                st.session_state.tracking_radius = (
                    tracking_radius
                )

                st.success(
                    "Tracking completed."
                )

            except Exception as error:
                st.error(
                    f"Tracking failed: {error}"
                )


if st.session_state.tracked_centres is not None:
    tracked_centres = (
        st.session_state.tracked_centres
    )
    tracking_success = (
        st.session_state.tracking_success
    )
    roi_radius = (
        st.session_state.roi_radius
    )

    success_count = int(
        tracking_success.sum()
    )
    total_count = len(
        tracking_success
    )

    st.metric(
        "Tracking success",
        f"{success_count} / {total_count} frames"
    )

    st.subheader(
        "Tracking quality-control preview"
    )

    check_frames = np.linspace(
        0,
        len(
            st.session_state.r18_registered
        ) - 1,
        9,
        dtype=int,
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9, 9),
    )

    qc_margin = max(
        35,
        int(
            st.session_state.get(
                "tracking_radius",
                18
            )
            * 3
        ),
    )

    for ax, frame_index in zip(
        axes.ravel(),
        check_frames,
    ):
        ax.imshow(
            st.session_state.r18_registered[
                frame_index
            ],
            cmap="gray",
        )

        centre_x, centre_y = (
            tracked_centres[
                frame_index
            ]
        )

        if (
            np.isfinite(centre_x)
            and np.isfinite(centre_y)
        ):
            ax.add_patch(
                Circle(
                    (
                        centre_x,
                        centre_y,
                    ),
                    roi_radius,
                    fill=False,
                    linewidth=2,
                )
            )

            ax.set_xlim(
                centre_x - qc_margin,
                centre_x + qc_margin,
            )
            ax.set_ylim(
                centre_y + qc_margin,
                centre_y - qc_margin,
            )

        ax.set_title(
            f"Frame {frame_index + 1}"
        )
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.header(
    "5. AF488 fluorescence analysis"
)

if st.session_state.tracked_centres is None:
    st.info(
        "Complete ROI tracking first."
    )
else:
    frame_interval = st.number_input(
        "Frame interval (seconds)",
        min_value=0.1,
        value=float(
            st.session_state.frame_interval
        ),
        step=0.1,
    )

    if st.button(
        "Extract AF488 fluorescence"
    ):
        intensity = extract_intensity(
            st.session_state.af488_registered,
            st.session_state.tracked_centres,
            st.session_state.roi_radius,
        )

        st.session_state.intensity = (
            intensity
        )
        st.session_state.frame_interval = (
            frame_interval
        )

        st.success(
            "AF488 fluorescence extracted."
        )


if st.session_state.intensity is not None:
    intensity = st.session_state.intensity
    frame_interval = (
        st.session_state.frame_interval
    )

    time_seconds = (
        np.arange(len(intensity))
        * frame_interval
    )

    st.subheader(
        "Fluorescence intensity profile"
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        time_seconds,
        intensity,
        marker="o",
        markersize=3,
    )

    ax.set_xlabel(
        "Time (s)"
    )
    ax.set_ylabel(
        "Mean AF488 intensity"
    )
    ax.set_title(
        "Tracked capillary-loop fluorescence"
    )

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "**Select the post-wash decay window**"
    )

    st.caption(
        "For manual-versus-semi-automated validation, use the "
        "same washout endpoint as the manual analysis."
    )

    col1, col2 = st.columns(2)

    max_time = float(
        time_seconds[-1]
    )

    with col1:
        fit_start = st.number_input(
            "Washout end / fit start (s)",
            min_value=0.0,
            max_value=max_time,
            value=min(
                60.0,
                max_time,
            ),
            step=float(
                frame_interval
            ),
        )

    with col2:
        default_end = min(
            fit_start + 60.0,
            max_time,
        )

        fit_end = st.number_input(
            "Fit end (s)",
            min_value=0.0,
            max_value=max_time,
            value=float(
                default_end
            ),
            step=float(
                frame_interval
            ),
        )

    if st.button(
        "Fit fluorescence decay"
    ):
        try:
            fit_result = fit_log_decay(
                time_seconds,
                intensity,
                fit_start,
                fit_end,
            )

            st.session_state.fit_result = (
                fit_result
            )

        except Exception as error:
            st.error(
                f"Decay fitting failed: {error}"
            )


if st.session_state.fit_result is not None:
    fit_result = (
        st.session_state.fit_result
    )

    st.subheader(
        "Decay fitting result"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Fitted decay slope k",
            f"{fit_result['k']:.6f} s⁻¹"
        )

    with col2:
        st.metric(
            "R²",
            f"{fit_result['r_squared']:.4f}"
        )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.scatter(
        fit_result["x"],
        fit_result["log"],
        label="Observed ln(intensity)",
    )

    ax.plot(
        fit_result["x"],
        fit_result["predicted_log"],
        label="Linear fit",
    )

    ax.set_xlabel(
        "Time (s)"
    )
    ax.set_ylabel(
        "ln(AF488 intensity)"
    )
    ax.set_title(
        "Post-wash fluorescence decay fit"
    )
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


st.header(
    "6. Capillary dimensions and Ps′alb calculation"
)

if st.session_state.fit_result is None:
    st.info(
        "Fit the fluorescence decay first."
    )

elif st.session_state.selected_centre is None:
    st.info(
        "Select a capillary loop first."
    )

else:
    st.write(
        "Measure two capillary-loop diameters directly on the "
        "registered structural image. Click two endpoints for D1 "
        "and two endpoints for D2."
    )

    pixel_size = st.number_input(
        "Pixel size (µm/pixel)",
        min_value=0.000001,
        value=float(
            st.session_state.pixel_size_um
        ),
        format="%.10f",
    )

    centre_x, centre_y = (
        st.session_state.selected_centre
    )

    measurement_margin = st.slider(
        "Diameter measurement view (half-width, pixels)",
        min_value=20,
        max_value=80,
        value=45,
        step=5,
    )

    source_frame = (
        st.session_state.r18_registered[0]
    )

    image_height, image_width = (
        source_frame.shape
    )

    x0 = max(
        0,
        int(
            centre_x
            - measurement_margin
        ),
    )
    x1 = min(
        image_width,
        int(
            centre_x
            + measurement_margin
        ),
    )
    y0 = max(
        0,
        int(
            centre_y
            - measurement_margin
        ),
    )
    y1 = min(
        image_height,
        int(
            centre_y
            + measurement_margin
        ),
    )

    measurement_crop = source_frame[
        y0:y1,
        x0:x1,
    ]

    crop_uint8 = to_uint8(
        measurement_crop
    )

    display_scale = 4

    d1_col, d2_col = st.columns(2)

    with d1_col:
        st.markdown(
            "**D1 — click two endpoints**"
        )

        d1_image = make_measurement_image(
            crop_uint8,
            st.session_state.d1_points,
            display_scale=display_scale,
        )

        d1_click = streamlit_image_coordinates(
            d1_image,
            key="d1_measurement_image",
        )

        register_measurement_click(
            "d1_points",
            "d1_last_click",
            d1_click,
            display_scale,
        )

        if st.button(
            "Reset D1",
            key="reset_d1",
        ):
            st.session_state.d1_points = []
            st.session_state.d1_last_click = None
            st.rerun()

        st.caption(
            f"{len(st.session_state.d1_points)} / 2 points selected"
        )

    with d2_col:
        st.markdown(
            "**D2 — click two endpoints**"
        )

        d2_image = make_measurement_image(
            crop_uint8,
            st.session_state.d2_points,
            display_scale=display_scale,
        )

        d2_click = streamlit_image_coordinates(
            d2_image,
            key="d2_measurement_image",
        )

        register_measurement_click(
            "d2_points",
            "d2_last_click",
            d2_click,
            display_scale,
        )

        if st.button(
            "Reset D2",
            key="reset_d2",
        ):
            st.session_state.d2_points = []
            st.session_state.d2_last_click = None
            st.rerun()

        st.caption(
            f"{len(st.session_state.d2_points)} / 2 points selected"
        )

    if (
        len(st.session_state.d1_points) == 2
        and len(st.session_state.d2_points) == 2
    ):
        d1_pixels = distance_between_points(
            st.session_state.d1_points[0],
            st.session_state.d1_points[1],
        )

        d2_pixels = distance_between_points(
            st.session_state.d2_points[0],
            st.session_state.d2_points[1],
        )

        d1_um = d1_pixels * pixel_size
        d2_um = d2_pixels * pixel_size

        radius_um = (
            d1_um + d2_um
        ) / 4.0

        radius_cm = (
            radius_um
            * 1e-4
        )

        k = (
            st.session_state.fit_result[
                "k"
            ]
        )

        ps_alb = -(
            k
            * radius_cm
        ) / 2.0

        st.divider()

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:
            st.metric(
                "D1",
                f"{d1_um:.3f} µm",
            )

        with col2:
            st.metric(
                "D2",
                f"{d2_um:.3f} µm",
            )

        with col3:
            st.metric(
                "Capillary radius R",
                f"{radius_cm:.3e} cm",
            )

        with col4:
            st.metric(
                "Ps′alb",
                f"{ps_alb:.3e} cm/s",
            )

        st.latex(
            r"Ps'_{alb} = -\frac{kR}{2}"
        )


if (
    st.session_state.fit_result is not None
    and st.session_state.intensity is not None
):
    st.header(
        "7. Analysis summary"
    )

    if (
        st.session_state.tracking_success
        is not None
    ):
        tracking_percent = (
            100
            * st.session_state
            .tracking_success
            .mean()
        )
    else:
        tracking_percent = np.nan

    st.write(
        f"""
        **Selected ROI:** {st.session_state.selected_centre}  
        **Measurement ROI radius:** {st.session_state.roi_radius} pixels  
        **Tracking radius:** {st.session_state.get('tracking_radius', 'N/A')} pixels  
        **Tracking success:** {tracking_percent:.1f}%  
        **Decay fit R²:** {st.session_state.fit_result['r_squared']:.4f}  
        **Fitted decay slope k:** {st.session_state.fit_result['k']:.6f} s⁻¹
        """
    )

    st.success(
        "Analysis complete."
    )
