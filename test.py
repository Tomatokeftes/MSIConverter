import os
import shutil

import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
import zarr
from shapely.geometry import Polygon
from spatialdata.models import Image3DModel, Labels3DModel, PointsModel, ShapesModel

# -- 1. Setup and Initialization --
zarr_path = "streaming_experiment_full.zarr"
if os.path.exists(zarr_path):
    shutil.rmtree(zarr_path)  # Clean up previous runs

# --- Define Shapes and Placeholders for all Elements ---

# a) Image Element
image_shape = (1, 50, 256, 256)  # c, z, y, x
image_chunks = (1, 1, 128, 128)
image_dtype = np.uint16
placeholder_image = da.zeros(shape=image_shape, chunks=image_chunks, dtype=image_dtype)

# b) Labels Element (similar to image)
labels_shape = (50, 256, 256)  # z, y, x
labels_chunks = (1, 128, 128)
labels_dtype = np.int32
placeholder_labels = da.zeros(
    shape=labels_shape, chunks=labels_chunks, dtype=labels_dtype
)

# c) Points Element (start with an empty table)
empty_points_ddf = dd.from_pandas(
    pd.DataFrame(np.empty((0, 2)), columns=["x", "y"]), npartitions=1
)
placeholder_points = sd.models.PointsModel.parse(empty_points_ddf)

# d) Shapes Element will be created on-the-fly to avoid initialization errors with empty data.


# --- Create and Write the Initial SpatialData Object ---
sdata_initial = sd.SpatialData(
    images={
        "my_streaming_image": Image3DModel.parse(
            placeholder_image, dims=("c", "z", "y", "x")
        )
    },
    labels={
        "my_streaming_labels": Labels3DModel.parse(
            placeholder_labels, dims=("z", "y", "x")
        )
    },
    points={"my_streaming_points": placeholder_points},
)
sdata_initial.write(zarr_path)

print(f"✅ Initial empty Zarr store created at: {zarr_path}")
print(sdata_initial)


# -- 2. Incremental Writing Loop --

# Open the Zarr group in read-write mode
root = zarr.open(zarr_path, mode="r+")

# a) Get handles for raster data
image_array = root["images/my_streaming_image/0"]
labels_array = root["labels/my_streaming_labels/0"]

print("\n🚀 Starting to stream/write chunks for all elements...")

# b) Incrementally write to Image (slice by slice)
print("   ... Writing Image slices")
for z_index in range(image_shape[1]):
    new_slice = np.random.randint(
        0, 1000, size=(image_shape[2], image_shape[3]), dtype=image_dtype
    )
    image_array[0, z_index, :, :] = new_slice

# c) Incrementally write to Labels (slice by slice)
print("   ... Writing Labels slices")
for z_index in range(labels_shape[0]):
    # Create a label in the middle of the slice
    new_label_slice = np.zeros((labels_shape[1], labels_shape[2]), dtype=labels_dtype)
    new_label_slice[100:150, 100:150] = (
        z_index + 1
    )  # Assign a unique label id per slice
    labels_array[z_index, :, :] = new_label_slice

# d) Incrementally add Points (batch by batch)
print("   ... Writing Points batches")
sdata_for_update = sd.read_zarr(zarr_path)
n_points_to_add = 100
# Get the Dask DataFrame for points once before the loop
points_ddf = sdata_for_update.points["my_streaming_points"]
for i in range(n_points_to_add):
    new_points_coords = np.random.rand(1, 2) * 256  # A single new point
    new_points_df = pd.DataFrame(new_points_coords, columns=["x", "y"])
    new_points_ddf = dd.from_pandas(new_points_df, npartitions=1)

    # CORRECTED: Concatenate the Dask DataFrames directly.
    points_ddf = dd.concat([points_ddf, new_points_ddf])

# Replace the old points element with the new concatenated one
sdata_for_update.points["my_streaming_points"] = PointsModel.parse(points_ddf)


# e) Incrementally add Shapes (one by one)
print("   ... Writing Shapes one by one")
n_shapes_to_add = 20
# Prepare a list to hold new GeoDataFrames
new_shapes_list = []
for i in range(n_shapes_to_add):
    center_x, center_y = np.random.rand(2) * 200
    new_poly = Polygon(
        [(center_x, center_y), (center_x + 10, center_y), (center_x + 5, center_y + 10)]
    )
    new_shapes_list.append(gpd.GeoDataFrame({"geometry": [new_poly]}))

# CORRECTED: Concatenate all new shapes with the existing ones (if any).
if len(new_shapes_list) > 0:
    all_new_shapes_gdf = pd.concat(new_shapes_list, ignore_index=True)
    if "my_streaming_shapes" in sdata_for_update.shapes:
        existing_shapes_gdf = sdata_for_update.shapes["my_streaming_shapes"]
        concatenated_gdf = pd.concat(
            [existing_shapes_gdf, all_new_shapes_gdf], ignore_index=True
        )
        sdata_for_update.shapes["my_streaming_shapes"] = ShapesModel.parse(
            concatenated_gdf
        )
    else:
        sdata_for_update.shapes["my_streaming_shapes"] = ShapesModel.parse(
            all_new_shapes_gdf
        )


# Write the updated points and shapes data back to disk
sdata_for_update.write(zarr_path, overwrite=True)


print("✅ Streaming finished.")

# -- 3. Verification --
print("\n🔍 Verifying final data...")
sdata_final = sd.read_zarr(zarr_path)
print(sdata_final)

# Verify Image
sample_image = sdata_final["my_streaming_image"].data[0, 25, 100:105, 100:105].compute()
assert np.all(sample_image != 0), "Image verification failed!"
print("✅ Image data verified.")

# Verify Labels
sample_labels = sdata_final["my_streaming_labels"].data[25, 120:125, 120:125].compute()
assert np.all(sample_labels == 26), "Labels verification failed!"
print("✅ Labels data verified.")

# Verify Points
final_points_count = len(sdata_final["my_streaming_points"])
assert final_points_count == n_points_to_add, "Points verification failed!"
print(f"✅ Points data verified ({final_points_count} points).")

# Verify Shapes
final_shapes_count = len(sdata_final["my_streaming_shapes"])
assert final_shapes_count == n_shapes_to_add, "Shapes verification failed!"
print(f"✅ Shapes data verified ({final_shapes_count} shapes).")
