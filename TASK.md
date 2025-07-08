# Project Tasks: MSIConverter

This document lists the current development tasks. It is intended to be a living document that is updated as the project evolves.

## High Priority

- [x] **Improve `BrukerReader` DLL Handling:**
    - [x] Implement a more robust search mechanism for `timsdata.dll` to check common installation paths and environment variables.
    - [x] Add clearer error messages to guide the user if the DLL cannot be found.
    - [x] Wrap DLL function calls in `try...except` blocks to handle potential `ctypes` errors gracefully.
- [x] **Update any test that is necessary**
    - [ ] Make sure that we have complete test coverage 
- [x] **Remove lightweight and anndata formats use only spatialdata**

## Medium Priority

- [ ] **Implement Asynchronous Processing Pipeline:**
    - [ ] Research `asyncio` and other relevant libraries for asynchronous file I/O and processing.
    - [ ] Refactor the `BaseMSIConverter` and `BaseMSIReader` to support an async/await pattern.
    - [ ] Update the `convert_msi` function to run the asynchronous pipeline.

- [ ] **Add Support for Configuration Files:**
    - [ ] Choose a configuration file format (e.g., YAML or TOML).
    - [ ] Add a library like `PyYAML` or `toml` to the project dependencies.
    - [ ] Modify the `__main__.py` to accept an optional `--config` argument.
    - [ ] Implement logic to parse the configuration file and override command-line arguments.

## Low Priority

- [ ] **Create a Basic GUI:**
    - [ ] Choose a GUI framework (e.g., `PyQt`, `Tkinter`, or a web-based framework like `Flask` or `FastAPI` with a simple frontend).
    - [ ] Design a simple user interface for selecting input/output files and conversion options.
    - [ ] Implement the GUI to call the `convert_msi` function.