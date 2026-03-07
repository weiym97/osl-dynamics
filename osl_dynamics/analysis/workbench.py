"""Functions to use `Connectome Workbench <https://www.humanconnectome.org\
/software/connectome-workbench>`_.

"""

import os
import pathlib
import re
import subprocess
import warnings
import numpy as np

import nibabel as nib
from tqdm.auto import trange

from osl_dynamics import files

surfs = {
    0: [files.mask.surf_left, files.mask.surf_right],
    1: [files.mask.surf_left_inf, files.mask.surf_right_inf],
    2: [files.mask.surf_left_vinf, files.mask.surf_right_vinf],
}


def setup(path):
    """Sets up workbench.

    Adds workbench to the PATH environmental variable.

    Parameters
    ----------
    path : str
        Path to workbench installation.
    """
    if path not in os.environ["PATH"]:
        os.environ["PATH"] = f"{path}:{os.environ['PATH']}"


def compute_global_symmetric_range_from_cifti(cifti_path, pct=(1, 99), mask_zeros=True):
    """Compute symmetric (vmin, vmax) from a CIFTI (.dscalar or .dtseries).

    Uses percentile of absolute values across all states.
    """
    img = nib.load(str(cifti_path))
    data = img.get_fdata()
    data = np.asarray(data, dtype=float).ravel()

    if mask_zeros:
        data = data[data != 0]

    if data.size == 0:
        return -1.0, 1.0

    abs_data = np.abs(data)
    _, high = np.percentile(abs_data, [pct[0], pct[1]])
    maxval = float(high) if float(high) != 0 else 1e-6
    return -maxval, maxval


def set_metric_palette_user_scale(
    metric_file: str,
    vmin: float,
    vmax: float,
    palette_name: str = "ROY-BIG-BL",
):
    """Set Workbench metric palette to a user-defined symmetric range.

    Parameters
    ----------
    metric_file : str
        Path to the ``.func.gii`` metric file.
    vmin : float
        Lower (negative) bound of the colour scale.
    vmax : float
        Upper (positive) bound of the colour scale.
    palette_name : str, optional
        Workbench palette name.  Default is ``"ROY-BIG-BL"``.
    """
    cmd = [
        "wb_command",
        "-metric-palette",
        str(metric_file),
        "MODE_USER_SCALE",
        "-pos-user", "0", str(float(vmax)),
        "-neg-user", str(float(vmin)), "0",
        "-palette-name", palette_name,
        "-disp-pos", "true",
        "-disp-neg", "true",
        "-disp-zero", "false",
    ]
    subprocess.run(cmd, check=True)


def render(
    img,
    save_dir=None,
    interptype="trilinear",
    gui=True,
    inflation=0,
    image_name=None,
    input_is_cifti=False,
    width=1920,
    height=1080,
):
    """Render map in workbench.

    Parameters
    ----------
    img : str
        Path to image file.
    save_dir : str, optional
        Path to save rendered surface plots.
    interptype : str, optional
        Interpolation type. Default is :code:`'trilinear'`.
    gui : bool, optional
        Should we display the rendered plots in workbench?
        Default is :code:`True`.
    image_name : str, optional
        Filename of image to save.
    input_is_cifti : bool, optional
        Whether the input file is a CIFTI file.
    width : int, optional
        Width in pixels of saved images.  Default is ``1920``.
    height : int, optional
        Height in pixels of saved images.  Default is ``1080``.
    """
    img = pathlib.Path(img)

    if ".nii" not in img.suffixes:
        raise ValueError(f"img should be a nii or nii.gz file, got {img}.")

    if not img.exists():
        raise FileNotFoundError(img)

    vmin, vmax = compute_global_symmetric_range_from_cifti(img, pct=(1, 99))

    if save_dir is None:
        save_dir = pathlib.Path.cwd()
    else:
        save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_file = save_dir / img.stem
    surf_left, surf_right = surfs.get(inflation, surfs[0])

    stem_right = out_file.with_name(out_file.stem + "_right")
    stem_left  = out_file.with_name(out_file.stem + "_left")

    output_right = stem_right.with_suffix(".func.gii")
    output_left  = stem_left.with_suffix(".func.gii")

    if input_is_cifti:
        subprocess.run(
            [
                "wb_command", "-cifti-separate", str(img), "COLUMN",
                "-metric", "CORTEX_LEFT",  str(output_left),
                "-metric", "CORTEX_RIGHT", str(output_right),
            ]
        )
    else:
        volume_to_surface(img, surf=surf_right, output=output_right, interptype=interptype)
        volume_to_surface(img, surf=surf_left,  output=output_left,  interptype=interptype)

    cifti_right = stem_right.with_suffix(".dtseries.nii")
    cifti_left  = stem_left.with_suffix(".dtseries.nii")

    dense_timeseries(cifti=cifti_right, output=output_right, left_or_right="right")
    dense_timeseries(cifti=cifti_left,  output=output_left,  left_or_right="left")

    try:
        set_metric_palette_user_scale(str(output_left),  vmin, vmax, palette_name="ROY-BIG-BL")
        set_metric_palette_user_scale(str(output_right), vmin, vmax, palette_name="ROY-BIG-BL")
    except subprocess.CalledProcessError:
        warnings.warn(
            "wb_command -metric-palette failed; check Workbench flags and file permissions."
        )

    # Scene file is saved permanently into save_dir so it can be shared /
    # inspected and re-used for colorbar debugging.
    scene_path = save_dir / "scene.scene"
    print("Scene path will be saved to:", scene_path)

    if image_name:
        image(
            cifti_left=cifti_left,
            cifti_right=cifti_right,
            file_name=image_name,
            inflation=inflation,
            scene_path=scene_path,
            palette_name="ROY-BIG-BL",
            vmin=vmin,
            vmax=vmax,
            width=width,
            height=height,
        )

    if gui:
        visualise(
            cifti_left=cifti_left,
            cifti_right=cifti_right,
            inflation=inflation,
            scene_path=scene_path,
            palette_name="ROY-BIG-BL",
            vmin=vmin,
            vmax=vmax,
        )


def create_scene(
    cifti_left,
    cifti_right,
    inflation,
    scene_path,
    palette_name="ROY-BIG-BL",
    vmin=None,
    vmax=None,
):
    """Build a Workbench scene file from the template and save it to *scene_path*.

    The scene file is written to disk and its path is returned so callers can
    pass it to ``wb_command -show-scene`` or ``wb_view``, or share it for
    inspection.

    Parameters
    ----------
    cifti_left, cifti_right : path-like
        Left / right hemisphere CIFTI dense-timeseries files.
    inflation : int
        Surface inflation level (key into :data:`surfs`).
    scene_path : path-like
        Destination path for the scene file (e.g. ``save_dir / "scene.scene"``).
    palette_name : str, optional
        Workbench palette name stored in the scene.
    vmin, vmax : float, optional
        Colour-scale bounds stored in the scene.

    Returns
    -------
    pathlib.Path
        Path to the written scene file.
    """
    scene_file = files.scene.mode_scene
    scene_path = pathlib.Path(scene_path)

    surf_left, surf_right = surfs.get(inflation, surfs[0])

    scene = scene_file.read_text()
    scene = re.sub("{left_series}",             str(cifti_left.name),  scene)
    scene = re.sub("{right_series}",            str(cifti_right.name), scene)
    scene = re.sub("{parcellation_file_left}",  surf_left,             scene)
    scene = re.sub("{parcellation_file_right}", surf_right,            scene)

    scene_path.write_text(scene)
    return scene_path


def visualise(
    cifti_left,
    cifti_right,
    inflation=0,
    scene_path=None,
    palette_name="ROY-BIG-BL",
    vmin=None,
    vmax=None,
):
    """Launch ``wb_view`` with the generated scene.

    Parameters
    ----------
    cifti_left, cifti_right : path-like
        Left / right hemisphere CIFTI dense-timeseries files.
    inflation : int, optional
        Surface inflation level.
    scene_path : path-like, optional
        Destination path for the scene file.  Defaults to
        ``"scene.scene"`` in the current directory.
    palette_name : str, optional
        Workbench palette name.
    vmin, vmax : float, optional
        Colour-scale bounds.
    """
    surface = surfs.get(inflation, None)
    if surface is None:
        warnings.warn(
            f"Inflation of {inflation} is not a valid selection. Using '0' instead.",
            RuntimeWarning,
        )
        surface = surfs[0]

    if scene_path is None:
        scene_path = pathlib.Path("scene.scene")

    scene_path = create_scene(
        cifti_left, cifti_right, inflation, scene_path,
        palette_name=palette_name, vmin=vmin, vmax=vmax,
    )

    subprocess.run(
        [
            "wb_view",
            "-scene-load", str(scene_path),
            "ready",
            *surface,
            cifti_left,
            cifti_right,
        ]
    )


def image(
    cifti_left,
    cifti_right,
    file_name,
    inflation=0,
    scene_path=None,
    palette_name="ROY-BIG-BL",
    vmin=None,
    vmax=None,
    width=1920,
    height=1080,
):
    """Save each map frame as an image using ``wb_command -show-scene``.

    The scene file is written to *scene_path* (or alongside the output images
    if not specified) and kept on disk so it can be shared or inspected.

    Parameters
    ----------
    cifti_left, cifti_right : path-like
        Left / right hemisphere CIFTI dense-timeseries files.
    file_name : str
        Base output path (extension defaults to ``.png``).
    inflation : int, optional
        Surface inflation level.
    scene_path : path-like, optional
        Destination path for the scene file.  Defaults to ``scene.scene``
        next to the output images.
    palette_name : str, optional
        Workbench palette name.
    vmin, vmax : float, optional
        Colour-scale bounds.
    width : int, optional
        Render width in pixels.  Default is ``1920``.
    height : int, optional
        Render height in pixels.  Default is ``1080``.

    Returns
    -------
    pathlib.Path
        Path to the saved scene file.
    """
    file_path = pathlib.Path(file_name)
    suffix = file_path.suffix or ".png"
    file_path = file_path.with_suffix("")

    if scene_path is None:
        scene_path = file_path.parent / "scene.scene"

    scene_path = create_scene(
        cifti_left, cifti_right, inflation, scene_path,
        palette_name=palette_name, vmin=vmin, vmax=vmax,
    )

    n_modes = nib.load(cifti_left).shape[0]
    max_int_length = len(str(n_modes))

    pathlib.Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    file_pattern = f"{file_path}{{:0{max_int_length}d}}{suffix}"

    for i in trange(n_modes, desc="Saving images"):
        out_img = file_pattern.format(i)

        subprocess.run(
            [
                "wb_command",
                "-show-scene",
                str(scene_path),
                "ready",
                out_img,
                str(width),
                str(height),
                "-set-map-yoke",
                "I",
                f"{i + 1}",
            ],
            capture_output=True,
        )

    return scene_path


def volume_to_surface(nii, surf, output, interptype="trilinear"):
    subprocess.run(
        [
            "wb_command",
            "-volume-to-surface-mapping",
            str(nii),
            str(surf),
            str(output),
            f"-{interptype}",
        ]
    )


def dense_timeseries(cifti, output, left_or_right):
    subprocess.run(
        [
            "wb_command",
            "-cifti-create-dense-timeseries",
            cifti,
            f"-{left_or_right}-metric",
            output,
        ]
    )