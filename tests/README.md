# MSIConvert Test Suite

This directory contains the test suite for the `msiconvert` package, providing both unit tests and integration tests.

## Test Structure

The tests are organized as follows:

- `unit/`: Fast tests for individual components
  - Core functionality (registry, base classes)
  - Readers (imzML and Bruker)
  - Converters (SpatialData)
  - Utility functions

- `integration/`: End-to-end tests for the full conversion workflow
  - imzML format conversion tests
  - Bruker format conversion tests
  - Command-line interface tests

- `data/`: Test data for running the tests
  - Contains minimal test data for both imzML and Bruker formats

- `conftest.py`: Common fixtures for all tests

## Running the Tests

### Running Unit Tests

To run only the unit tests (fast, no external dependencies):

```bash
pytest
```

or explicitly:

```bash
pytest -m "not integration"
```

### Running Integration Tests

To run the integration tests:

```bash
pytest -m integration
```

### Running All Tests

To run both unit and integration tests:

```bash
pytest -m "unit or integration"
```

### Running Specific Test Files

To run tests from a specific file:

```bash
pytest tests/unit/test_registry.py
```

### Running Tests With Coverage Report

To run tests with coverage:

```bash
pytest --cov=msiconvert
```

For a detailed coverage report:

```bash
pytest --cov=msiconvert --cov-report=html
```

## Test Dependencies

In addition to the regular package dependencies, the test suite requires:

- pytest
- pytest-cov (optional, for coverage reports)
- mock (for mocking and patching)

These can be installed with:

```bash
pip install pytest pytest-cov mock
```

## Special Test Considerations

- **Bruker Tests**: Some Bruker tests require the Bruker timsdata DLL/shared library to be available. Tests will be skipped if these dependencies are not found.

- **SpatialData Tests**: Tests for SpatialData format conversion require the `spatialdata` package to be installed. Tests will be skipped if this package is not available.

- **Mock Data**: The test suite uses fixtures to create minimal test data rather than requiring large real-world datasets.

## Adding New Tests

When adding new functionality to the `msiconvert` package, please follow these guidelines for testing:

1. Add unit tests for new components, functions, or methods
2. Update integration tests if the conversion workflow is affected
3. Add new test fixtures to `conftest.py` if needed
4. Document any special requirements or dependencies for new tests