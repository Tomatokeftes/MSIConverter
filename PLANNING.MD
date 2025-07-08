# Project Planning: MSIConverter

This document outlines the high-level architecture, goals, and future development roadmap for the `MSIConverter` project.

## 1. Project Overview

`MSIConverter` is a Python library designed to convert Mass Spectrometry Imaging (MSI) data from various proprietary and open formats into the `SpatialData` format. The primary goal is to provide a flexible and extensible tool for MSI data interoperability.

## 2. Core Architecture

The project follows a modular `Reader-Converter` pattern, which promotes separation of concerns and allows for easy extension.

*   **Readers (`msiconvert/readers`):** Responsible for parsing specific MSI data formats (e.g., `imzML`, `Bruker TSF/TDF`). Each reader implements the `BaseMSIReader` interface, which defines a standard way to access metadata, dimensions, and spectral data.
*   **Converters (`msiconvert/converters`):** Responsible for transforming the data read by a `Reader` into the `SpatialData` output format. Each converter implements the `BaseMSIConverter` interface, which outlines a consistent conversion workflow.
*   **Core (`msiconvert/core`):** Contains the base classes (`BaseMSIReader`, `BaseMSIConverter`) and a central `registry` that dynamically discovers and registers available readers and converters. This allows for a plug-and-play architecture where new formats can be added without modifying the core logic.

## 3. Data Flow

1.  **Initialization:** The user provides an input file path and a desired output format via the command-line interface or programmatically.
2.  **Format Detection:** The `registry` uses a set of registered detector functions to identify the input file format.
3.  **Reader Instantiation:** The appropriate `Reader` class for the detected format is retrieved from the registry and instantiated.
4.  **Converter Instantiation:** The corresponding `Converter` class for the desired output format is retrieved and instantiated, receiving the `Reader` instance.
5.  **Conversion:** The `Converter`'s `convert` method is called. It uses the `Reader` to iterate through the MSI data, processes it, and builds the final data structures in memory.
6.  **Output:** The `Converter` saves the processed data to disk in the target format.

## 4. Key Dependencies

*   **`numpy` & `scipy`:** For numerical data manipulation and sparse matrix operations.
*   **`anndata`:** For creating and managing the `AnnData` object, a core data structure in bioinformatics.
*   **`spatialdata`:** For creating and managing the `SpatialData` object, which is designed to handle spatially resolved omics data.
*   **`zarr`:** For efficient, chunked storage of large numerical arrays, used by the `SpatialData` converter.
*   **`pyimzML`:** For parsing `imzML` files.
*   **`timsdata.dll`:** A proprietary library from Bruker required for reading `TSF/TDF` files.

## 5. Future Development Roadmap

### High Priority
*   **Improve `BrukerReader` Robustness:** Enhance the discovery mechanism for the `timsdata.dll` library to avoid manual path configuration. Add more comprehensive error handling for DLL interactions.
*   **Expand Format Support:** Add readers for other common MSI formats (e.g., Analyze 7.5, NetCDF).

### Medium Priority
*   **Asynchronous Processing:** Investigate and implement asynchronous data reading and processing to improve performance for very large datasets.
*   **Configuration File:** Allow users to specify conversion parameters via a configuration file (e.g., `config.yaml`) for easier batch processing.
*   **Enhanced Metadata Handling:** Improve the extraction and validation of metadata, ensuring compliance with community standards.

### Low Priority
*   **Plugin System:** Refactor the registration system into a more formal plugin architecture (e.g., using `entry_points`) to allow third-party extensions.
*   **GUI:** Develop a simple graphical user interface for users who are not comfortable with the command line.
