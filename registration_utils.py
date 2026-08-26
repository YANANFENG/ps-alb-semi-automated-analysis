
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
import SimpleITK as sitk


def estimate_translation(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
):
    """Estimate a 2D translation that aligns moving_np to fixed_np."""

    if fixed_np.shape != moving_np.shape:
        raise ValueError(
            f"Image shapes differ: {fixed_np.shape} and {moving_np.shape}"
        )

    fixed_img = sitk.GetImageFromArray(
        fixed_np.astype(np.float32)
    )
    moving_img = sitk.GetImageFromArray(
        moving_np.astype(np.float32)
    )

    fixed_reg = sitk.Normalize(fixed_img)
    moving_reg = sitk.Normalize(moving_img)

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsCorrelation()
    registration.SetMetricSamplingStrategy(
        registration.NONE
    )
    registration.SetInterpolator(
        sitk.sitkLinear
    )

    initial_transform = sitk.TranslationTransform(2)

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False,
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=300,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    transform = registration.Execute(
        fixed_reg,
        moving_reg,
    )

    return {
        "transform": transform,
        "x": float(transform.GetParameters()[0]),
        "y": float(transform.GetParameters()[1]),
        "metric": float(registration.GetMetricValue()),
        "stop_condition": (
            registration
            .GetOptimizerStopConditionDescription()
        ),
    }


def analyse_nd2_drift(
    nd2_path,
    registration_channel=1,
):
    """
    Estimate translation drift for each frame relative to frame 1.

    Parameters
    ----------
    nd2_path:
        Path to the ND2 file.
    registration_channel:
        Channel index used to estimate motion.
        For the current dataset:
        0 = AF488/FITC
        1 = R18/Rhodamine
        2 = transmitted light

    Returns
    -------
    data:
        Labelled xarray representation of the ND2.
    registration_stack:
        Registration-channel image stack.
    transforms:
        One transform per frame.
    drift_df:
        Table containing frame, x, y, metric and magnitude.
    """

    nd2_path = Path(nd2_path)

    if not nd2_path.exists():
        raise FileNotFoundError(
            f"ND2 file not found: {nd2_path}"
        )

    with nd2.ND2File(nd2_path) as file:
        data = file.to_xarray()
        registration_stack = (
            data
            .isel(C=registration_channel)
            .compute()
            .values
        )

    fixed_frame = registration_stack[0]

    records = []
    transforms = []

    for frame_index, moving_frame in enumerate(
        registration_stack
    ):
        if frame_index == 0:
            transform = sitk.TranslationTransform(2)

            transforms.append(transform)

            records.append({
                "frame": 1,
                "x": 0.0,
                "y": 0.0,
                "metric": np.nan,
                "stop_condition": "Reference frame",
            })

            continue

        result = estimate_translation(
            fixed_frame,
            moving_frame,
        )

        transforms.append(result["transform"])

        records.append({
            "frame": frame_index + 1,
            "x": result["x"],
            "y": result["y"],
            "metric": result["metric"],
            "stop_condition": result[
                "stop_condition"
            ],
        })

    drift_df = pd.DataFrame(records)

    drift_df["magnitude"] = np.hypot(
        drift_df["x"],
        drift_df["y"],
    )

    return (
        data,
        registration_stack,
        transforms,
        drift_df,
    )
def apply_transform_to_frame(
    frame_np: np.ndarray,
    reference_np: np.ndarray,
    transform,
    interpolator=sitk.sitkLinear,
) -> np.ndarray:
    """Apply a transform and resample a frame into reference coordinates."""

    moving_img = sitk.GetImageFromArray(
        frame_np.astype(np.float32)
    )
    reference_img = sitk.GetImageFromArray(
        reference_np.astype(np.float32)
    )

    registered_img = sitk.Resample(
        moving_img,
        reference_img,
        transform,
        interpolator,
        0.0,
        sitk.sitkFloat32,
    )

    return sitk.GetArrayFromImage(registered_img)


def register_stack(
    stack_np: np.ndarray,
    transforms,
    interpolator=sitk.sitkLinear,
) -> np.ndarray:
    """Apply one transform per frame relative to the first frame."""

    if len(stack_np) != len(transforms):
        raise ValueError(
            "Number of frames and transforms must match."
        )

    reference = stack_np[0]
    registered_frames = []

    for frame, transform in zip(stack_np, transforms):
        registered = apply_transform_to_frame(
            frame,
            reference,
            transform,
            interpolator,
        )
        registered_frames.append(registered)

    return np.stack(registered_frames)