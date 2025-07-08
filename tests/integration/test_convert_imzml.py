"""
Integration tests for converting imzML files to various formats.
"""
import pytest
import os
import numpy as np
from pathlib import Path
import shutil


from msiconvert.convert import convert_msi


class TestImzMLConversion:
    """Test the end-to-end conversion of imzML files."""
    
    
    
    
    
    @pytest.mark.skipif(not pytest.importorskip("spatialdata", reason="SpatialData not installed"),
                      reason="SpatialData not installed")
    def test_convert_to_spatialdata(self, create_minimal_imzml, temp_dir):
        """Test converting imzML to SpatialData format."""
        # Skip if SpatialData is not available
        spatialdata = pytest.importorskip("spatialdata")
        
        # Get test data
        imzml_path, _, mzs, _ = create_minimal_imzml
        output_path = temp_dir / "output.zarr"
        
        # Run conversion
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="spatialdata",
            dataset_id="test_dataset",
            pixel_size_um=2.0
        )
        
        # Check result
        assert result is True
        assert output_path.exists()
        assert output_path.is_dir()  # Zarr stores are directories
        
        # Verify the output file by loading it
        try:
            sdata = spatialdata.SpatialData.read(str(output_path))
            
            # Check structure
            assert len(sdata.tables) == 1
            assert "test_dataset" in sdata.tables
            assert len(sdata.shapes) == 1
            
            # Get the table
            table = sdata.tables["test_dataset"]
            
            # Check table structure
            assert table.n_obs == 4  # 2x2 grid = 4 pixels
            assert table.n_vars == len(mzs)  # Should match number of m/z values
            assert 'average_spectrum' in table.uns
            
            # Check spatial coordinates are now in the obs dataframe
            assert "spatial_x" in table.obs.columns
            assert "spatial_y" in table.obs.columns
            
        except Exception as e:
            pytest.fail(f"Failed to load generated SpatialData file: {e}")

    
    def test_convert_nonexistent_file(self, temp_dir):
        """Test error handling with nonexistent input file."""
        # Create nonexistent path
        nonexistent_path = temp_dir / "nonexistent.imzML"
        output_path = temp_dir / "output.zarr"
        
        # Run conversion
        result = convert_msi(
            str(nonexistent_path),
            str(output_path),
            format_type="spatialdata"
        )
        
        # Check result
        assert result is False
        assert not output_path.exists()
    
    def test_conversion_with_existing_output(self, create_minimal_imzml, temp_dir):
        """Test error handling when output file already exists."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "existing_output.zarr"
        
        # Create the output file
        with open(output_path, "w") as f:
            f.write("existing file")
        
        # Run conversion
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="spatialdata"
        )
        
        # Check result
        assert result is False
        
        # Verify file wasn't overwritten
        with open(output_path, "r") as f:
            content = f.read()
        assert content == "existing file"
    
    def test_convert_with_invalid_format(self, create_minimal_imzml, temp_dir):
        """Test error handling with invalid format type."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "output.zarr"
        
        # Run conversion with invalid format
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="invalid_format"
        )
        
        # Check result
        assert result is False
        assert not output_path.exists()