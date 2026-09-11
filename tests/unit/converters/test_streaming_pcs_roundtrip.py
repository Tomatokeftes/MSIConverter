"""Round-trip tests for the streaming converter's PCS (CSC) path.

Regression coverage for the silent-corruption bug where the PCS path
hand-writes the table store, then added the TIC image / optical images by
re-reading the store and calling ``write_element`` inside a try/except that
swallowed any failure.  A failed image write left an image group with
incomplete OME metadata (no ``ome.version``/``multiscales``) and no optical
images, yet ``convert()`` returned ``True`` -- so a corrupt, unreadable zarr
was reported as a success and only blew up later in ``spatialdata.read_zarr``
with ``KeyError: 'version'``.

These tests assert that:
  * the PCS path produces a ``spatialdata.read_zarr``-readable zarr;
  * the TIC (and optical) image groups carry ``ome.version``;
  * a failure while writing images/optical makes ``convert()`` return
    ``False`` (fail loudly) instead of reporting a corrupt success.
"""

import numpy as np
import zarr

from tests.fixtures.mock_msi_generator import MockMSIConfig, MockMSIReader
from thyra.converters.spatialdata.streaming_converter import (
    StreamingSpatialDataConverter,
)


def _image_ome_attrs(zarr_path, image_name):
    """Return the on-disk ``ome`` attribute dict for an image element."""
    root = zarr.open_group(str(zarr_path), mode="r")
    return dict(root["images"][image_name].attrs)["ome"]


def _small_config(**overrides):
    base = dict(n_x=6, n_y=6, n_mz_bins=500, peaks_per_spectrum=(20, 40))
    base.update(overrides)
    return MockMSIConfig(**base)


def test_pcs_path_roundtrips_with_ome_version(tmp_path):
    """PCS path output must be read_zarr-readable with a versioned TIC image."""
    reader = MockMSIReader(_small_config())
    out = tmp_path / "pcs.zarr"

    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=out,
        dataset_id="mock",
        pixel_size_um=10.0,
        use_csc=True,  # force the PCS path
    )

    assert converter.convert() is True, "PCS conversion should succeed"

    # The TIC image group must carry complete OME metadata: the original bug
    # left only ['omero'] (no version/multiscales).
    ome = _image_ome_attrs(out, "mock_z0_tic")
    assert "version" in ome, f"TIC image missing ome.version (got {list(ome)})"
    assert "multiscales" in ome

    # read_zarr must round-trip (this is exactly what broke downstream).
    import spatialdata

    sdata = spatialdata.read_zarr(str(out))
    assert "mock_z0_tic" in sdata.images
    assert "mock_z0_pixels" in sdata.shapes
    assert len(sdata.tables) == 1
    table = list(sdata.tables.values())[0]
    assert table.n_obs == 36  # 6x6 fully-populated grid


def test_pcs_path_writes_optical_image(tmp_path):
    """Optical images requested in the PCS path must be written and readable."""
    import tifffile

    # Large enough to exercise the multi-scale pyramid + chunked write that the
    # PCS path shares with the in-memory converters.
    optical = tmp_path / "optical.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(str(optical), rng.integers(0, 255, (1024, 1024), dtype="uint8"))

    reader = MockMSIReader(_small_config(), optical_image_paths=[optical])
    out = tmp_path / "pcs_optical.zarr"

    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=out,
        dataset_id="mock",
        pixel_size_um=10.0,
        use_csc=True,
        include_optical=True,
    )

    assert converter.convert() is True

    import spatialdata

    sdata = spatialdata.read_zarr(str(out))
    optical_keys = [k for k in sdata.images if "optical" in k.lower()]
    assert optical_keys, f"expected an optical image, got {list(sdata.images)}"

    # Both the TIC and the optical image must carry ome.version.
    assert "version" in _image_ome_attrs(out, "mock_z0_tic")
    assert "version" in _image_ome_attrs(out, optical_keys[0])


def test_image_write_failure_is_not_silent(tmp_path, monkeypatch):
    """A failure writing images/optical must make convert() return False.

    The original code swallowed the exception and returned True, producing a
    silently corrupt zarr.  Whatever the cause, an image-write failure must
    now surface as a failed conversion.
    """
    reader = MockMSIReader(_small_config())
    out = tmp_path / "fail.zarr"

    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=out,
        dataset_id="mock",
        pixel_size_um=10.0,
        use_csc=True,
    )

    def _boom(_data_structures):
        raise RuntimeError("simulated optical/image write failure")

    # _add_optical_images is the shared image-loading seam invoked while
    # writing the TIC + optical elements; forcing it to raise simulates any
    # downstream image/optical write failure.
    monkeypatch.setattr(converter, "_add_optical_images", _boom)

    assert converter.convert() is False, (
        "convert() must fail loudly when image writing fails, not report a "
        "corrupt zarr as success"
    )
