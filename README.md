# FlowUtils - Pure Python Flow Cytometry Transforms

[![PyPI license](https://img.shields.io/pypi/l/flowutils.svg?colorB=dodgerblue)](https://pypi.python.org/pypi/flowutils/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowutils.svg)](https://pypi.python.org/pypi/flowutils/)
[![PyPI version](https://img.shields.io/pypi/v/flowutils.svg?colorB=blue)](https://pypi.python.org/pypi/flowutils/)

FlowUtils is a **pure Python** package providing optimized implementations of 
**Logicle** and **Hyperlog** transforms for flow cytometry data analysis. 

This library has been completely rewritten in pure Python to provide:
- **Zero C dependencies** - No compilation required
- **NumPy 1.22+ compatibility** - Works with modern NumPy versions  
- **Mathematical accuracy** - Faithful implementation of published algorithms
- **Clean API** - Simple, focused interface for transform operations

## Key Features

- **Logicle Transform**: Bi-exponential transform for flow cytometry data with both positive and negative values
- **Hyperlog Transform**: Flexible log-like transform optimized for flow cytometry applications  
- **Inverse Transforms**: Both forward and inverse transformations with high precision
- **Multi-channel Support**: Efficient processing of multi-dimensional flow cytometry datasets
- **Extensive Testing**: Comprehensive test suite ensuring mathematical accuracy

## Installation

### From PyPI

```bash
pip install flowutils
```

### From GitHub source code

```bash
git clone https://github.com/ashajkofci/flowutils
cd flowutils
pip install .
```

## Quick Start

```python
import numpy as np
from flowutils import transforms

# Sample flow cytometry data
data = np.array([-1000, -100, 0, 100, 1000, 10000])

# Apply Logicle transform
logicle_data = transforms.logicle(data, channel_indices=None, t=10000, m=4.5, w=0.5, a=0)

# Apply Hyperlog transform  
hyperlog_data = transforms.hyperlog(data, channel_indices=None, t=10000, m=4.5, w=0.5, a=0)

# Inverse transforms
original_logicle = transforms.logicle_inverse(logicle_data, channel_indices=None, t=10000, m=4.5, w=0.5, a=0)
original_hyperlog = transforms.hyperlog_inverse(hyperlog_data, channel_indices=None, t=10000, m=4.5, w=0.5, a=0)
```

## Multi-channel Example

```python
import numpy as np
from flowutils import transforms

# Multi-channel flow cytometry data (1000 events, 3 channels)
data = np.random.rand(1000, 3) * 10000

# Transform only fluorescence channels (0 and 2), leave scatter channel (1) unchanged
transformed = transforms.logicle(data, channel_indices=[0, 2], t=10000, m=4.5, w=0.5, a=0)
```

## Transform Parameters

Both transforms accept the following parameters:

- **t**: Top of the scale (e.g., 262144 for typical flow cytometry)
- **m**: Number of decades the true logarithmic scale approaches at the high end
- **w**: Number of decades in the approximately linear region  
- **a**: Number of additional negative decades

## Mathematical Background

### Logicle Transform

The Logicle transformation implements the bi-exponential function as defined in:

> Moore WA and Parks DR. Update for the logicle data scale including operational code implementations. *Cytometry A*, 2012:81A(4):273–277.

### Hyperlog Transform  

The Hyperlog transformation implements the generalized logarithm as defined in:

> Bagwell CB. Hyperlog-a flexible log-like transform for negative, zero, and positive valued data. *Cytometry A*, 2005:64(1):34–42.

## Performance Notes

This pure Python implementation prioritizes:
- **Compatibility**: Works with NumPy 1.22+ and Python 3.7+
- **Reliability**: No compilation dependencies or C API compatibility issues
- **Accuracy**: High-precision mathematical implementations
- **Maintainability**: Clean, readable Python code

For applications requiring maximum performance with very large datasets, consider using optimization tools like Numba for just-in-time compilation.

## Testing

Run the test suite to verify the installation:

```bash
# Run basic transform tests
python -m unittest flowutils.tests.transform_tests

# Run tests with simulated FCS data 
python -m unittest flowutils.tests.test_fcs_data

# Run all tests
python -m unittest discover flowutils.tests
```

## Examples

See the `examples/` directory for detailed usage examples:
- `simple_example.py`: Basic usage demonstration
- `transforms_example.py`: Comprehensive example with visualization
- `fcs_comparison_notebook.ipynb`: Jupyter notebook with FL1-FL2 transform comparison plots

### FL1-FL2 Transform Comparison

The included Jupyter notebook (`examples/fcs_comparison_notebook.ipynb`) provides a comprehensive comparison of different transform methods applied to realistic flow cytometry data:

- **Linear scaling** - Raw data visualization  
- **Log-log transformation** - Traditional logarithmic scaling
- **Hyperlog transformation** - FlowUtils hyperlog implementation
- **Logicle transformation** - FlowUtils logicle implementation

The notebook includes:
- Simulated multi-population FL1-FL2 data
- Side-by-side scatter plot comparisons  
- Population density visualizations
- Transform function behavior analysis
- Statistical summaries and recommendations

## Requirements

- Python ≥ 3.7
- NumPy ≥ 1.22, < 2.0

## License

This project is licensed under the BSD License - see the LICENSE file for details.
