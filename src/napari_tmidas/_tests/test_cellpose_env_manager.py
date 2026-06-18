import types

import numpy as np
import pytest

from napari_tmidas.processing_functions import cellpose_env_manager


def test_normalize_zarr_selection_coerces_integral_float_slices():
    """Integral numpy floats should become built-in slice bounds."""
    selection = (
        slice(np.float64(0.0), np.float64(155.0), None),
        slice(np.float64(254.0), np.float64(514.0), None),
        np.int64(3),
    )

    normalized = cellpose_env_manager._normalize_zarr_selection(selection)

    assert normalized == (slice(0, 155, None), slice(254, 514, None), 3)
    assert isinstance(normalized[0].start, int)
    assert isinstance(normalized[0].stop, int)
    assert isinstance(normalized[1].start, int)
    assert isinstance(normalized[1].stop, int)
    assert isinstance(normalized[2], int)


def test_normalize_zarr_selection_rejects_non_integral_float_bounds():
    """Non-integral float slice bounds should fail early."""
    with pytest.raises(TypeError):
        cellpose_env_manager._normalize_zarr_selection(
            slice(np.float64(1.5), np.float64(10.0), None)
        )


def test_patch_cellpose_distributed_crop_selection_normalizes_crop():
    """Patched Cellpose distributed reader should receive integer slices."""
    captured = {}

    def fake_read_preprocess_and_segment(input_zarr, crop, *args, **kwargs):
        del input_zarr, args, kwargs
        captured["crop"] = crop
        return "ok"

    module = types.SimpleNamespace(
        read_preprocess_and_segment=fake_read_preprocess_and_segment
    )

    patched = cellpose_env_manager._patch_cellpose_distributed_crop_selection(
        module
    )
    result = module.read_preprocess_and_segment(
        object(),
        (
            slice(np.float64(0.0), np.float64(12.0), None),
            slice(np.float64(20.0), np.float64(30.0), None),
        ),
    )

    assert patched is True
    assert result == "ok"
    assert captured["crop"] == (
        slice(0, 12, None),
        slice(20, 30, None),
    )
