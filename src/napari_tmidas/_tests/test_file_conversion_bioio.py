"""Tests for BioIO-backed file conversion helpers."""

from types import SimpleNamespace

import dask.array as da
import numpy as np

from napari_tmidas import _file_conversion as file_conversion


class FakeBioImage:
    """Minimal BioImage test double for scene-based loading."""

    def __init__(self, filepath: str, reader=None):
        self.filepath = filepath
        self.reader = reader
        self.scenes = ["Image:0", "Image:1"]
        self._scene_index = 0
        self.channel_names = ["DNA", "Actin"]
        self.physical_pixel_sizes = SimpleNamespace(
            Z=1.5,
            Y=0.4,
            X=0.3,
        )

    def set_scene(self, scene) -> None:
        """Select the active test scene."""
        if isinstance(scene, str):
            self._scene_index = self.scenes.index(scene)
        else:
            self._scene_index = int(scene)

    @property
    def dims(self):
        """Return a BioIO-like dimensions object."""
        return SimpleNamespace(order="TCZYX")

    @property
    def dask_data(self):
        """Return scene-specific lazy data in TCZYX order."""
        data = np.full(
            (2, 2, 3, 4, 5),
            fill_value=self._scene_index,
            dtype=np.uint16,
        )
        return da.from_array(data, chunks=(1, 1, 1, 4, 5))


def test_bioio_loader_reads_scene_metadata(monkeypatch):
    """BioIO loader should expose scenes, lazy data, and pixel sizes."""
    monkeypatch.setattr(file_conversion, "BioImage", FakeBioImage)
    monkeypatch.setattr(
        file_conversion,
        "bioio_bioformats",
        SimpleNamespace(Reader=object()),
    )

    filepath = "/tmp/example.fake"

    assert file_conversion.BioIOLoader.can_load(filepath)
    assert file_conversion.BioIOLoader.get_series_count(filepath) == 2

    image_data = file_conversion.BioIOLoader.load_series(filepath, 1)
    assert image_data.shape == (2, 2, 3, 4, 5)
    np.testing.assert_array_equal(
        image_data.compute(),
        np.ones((2, 2, 3, 4, 5), dtype=np.uint16),
    )

    metadata = file_conversion.BioIOLoader.get_metadata(filepath, 1)
    assert metadata["axes"] == "TCZYX"
    assert metadata["resolution"] == (1.0 / 0.3, 1.0 / 0.4)
    assert metadata["spacing"] == 1.5
    assert metadata["channel_names"] == ["DNA", "Actin"]


def test_build_scale_transform_uses_resolution_and_spacing(tmp_path):
    """BioIO metadata should fit the existing OME-Zarr scale logic."""
    worker = file_conversion.ConversionWorker(
        files_to_convert=[],
        output_folder=str(tmp_path),
        use_zarr=True,
        file_loader_func=lambda _filepath: None,
    )

    transform = worker._build_scale_transform(
        metadata={"resolution": (1.0 / 0.3, 1.0 / 0.4), "spacing": 1.5},
        axes="tczyx",
        shape=(2, 2, 3, 4, 5),
    )

    assert transform == {
        "type": "scale",
        "scale": [1.0, 1.0, 1.5, 0.4, 0.3],
    }


def test_rechunk_for_zarr_reduces_oversized_chunks(tmp_path):
    """Oversized chunks should be reduced below the zarr codec limit."""
    worker = file_conversion.ConversionWorker(
        files_to_convert=[],
        output_folder=str(tmp_path),
        use_zarr=True,
        file_loader_func=lambda _filepath: None,
    )

    image_data = da.zeros(
        (1, 3, 115, 37716, 27277),
        chunks=(1, 3, 115, 37716, 27277),
        dtype=np.uint16,
    )

    rechunked = worker._rechunk_for_zarr(
        image_data,
        axes="tczyx",
        max_chunk_bytes=1_500_000_000,
    )

    assert (
        worker._chunk_nbytes(rechunked.chunksize, rechunked.dtype)
        <= 1_500_000_000
    )
