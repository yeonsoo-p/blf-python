# blf_python

A high-performance Python library for reading and decoding BLF (Binary Logging Format) files with DBC (CAN Database) support. Built with C++ for speed, delivering 10-15x faster parsing compared to pure Python alternatives.

## Features

- **Fast**: 10-15x faster than cantools + python-can
- **Zero-copy**: Direct NumPy array access to signal data without copying
- **Simple API**: Intuitive interface for accessing CAN signals
- **Type-safe**: Full type hints for better IDE support
- **Efficient**: Automatic caching for repeated access
- **Flexible**: Support for multiple channels and DBC files
- **Compatible**: Automatic detection of timestamp formats (1ns and 10μs)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd blf-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Build the extension
mkdir build
cd build
cmake ..
cmake --build .
```

## Quick Start

```python
from blf_python import BLF

# Load BLF file with DBC database
blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])

# Access signal data
timestamps = blf.get_signal('GpsStatus', 'Time')
gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')

# Or use dictionary-style access
velocity = blf['Distance']['Distance']
```

## Usage Guide

### Opening BLF Files

#### Single DBC for all channels (wildcard)
```python
blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])
```

#### Specific channel
```python
blf = BLF('recording.blf', [(1, 'vehicle.dbc')])
```

#### Multiple channels with different DBCs
```python
blf = BLF('recording.blf', [
    (1, 'powertrain.dbc'),
    (2, 'chassis.dbc')
])
```

### Exploring Available Data

```python
# List all decoded messages
messages = blf.get_message_names()
print(f"Messages: {messages}")

# Get signals in a message
signals = blf.get_signals('GpsStatus')
print(f"Signals in GpsStatus: {signals}")

# Get sample count
count = blf.get_message_count('GpsStatus')
print(f"GpsStatus has {count} samples")

# Print summary
blf.info()
```

### Accessing Signal Data

#### Method 1: Direct access (recommended)
```python
# Get individual signal
signal_data = blf.get_signal('GpsStatus', 'GpsPosMode')
# Returns: NumPy array of float64

# Get timestamps
timestamps = blf.get_time_series('GpsStatus')
# Equivalent to: blf.get_signal('GpsStatus', 'Time')
```

#### Method 2: Dictionary-style access
```python
# Access via MessageProxy
proxy = blf['GpsStatus']
gps_mode = proxy['GpsPosMode']
# Returns: NumPy array of float64
```

#### Method 3: Get entire message as 2D array
```python
# Get all signals at once
data = blf.get_message('GpsStatus')
# Returns: 2D NumPy array (num_samples, num_signals)
# Column 0 is Time, columns 1+ are signals in order

# Get all messages
all_data = blf.get_all_messages()
# Returns: dict[str, NDArray] mapping message names to 2D arrays
```

### Accessing Signal Metadata

```python
# Get unit string
unit = blf.get_signal_unit('Distance', 'Distance')
print(f"Unit: {unit}")  # e.g., "m" or "km/h"

# Get scaling factor
factor = blf.get_signal_factor('Distance', 'Distance')
print(f"Factor: {factor}")

# Get scaling offset
offset = blf.get_signal_offset('Distance', 'Distance')
print(f"Offset: {offset}")

# Get sampling period (two ways)
period1 = blf.get_period('GpsStatus')  # Method 1: via BLF object
period2 = blf['GpsStatus'].get_period()  # Method 2: via MessageProxy
print(f"Sampling period: {period1} ms")

# Get all metadata at once (via MessageProxy)
proxy = blf['Distance']
units = proxy.get_signal_units()  # dict[str, str]
factors = proxy.get_signal_factors()  # dict[str, float]
offsets = proxy.get_signal_offsets()  # dict[str, float]
```

### Working with NumPy Arrays

All signal data is returned as NumPy arrays, ready for analysis:

```python
import numpy as np
import matplotlib.pyplot as plt

# Get signal data
timestamps = blf.get_signal('Distance', 'Time')
distance = blf.get_signal('Distance', 'Distance')

# Statistical analysis
print(f"Mean distance: {np.mean(distance):.2f}")
print(f"Max distance: {np.max(distance):.2f}")
print(f"Std dev: {np.std(distance):.2f}")

# Plotting
plt.plot(timestamps, distance)
plt.xlabel('Time (s)')
plt.ylabel('Distance')
plt.title('Distance over Time')
plt.show()
```

### Advanced Features

#### Check if message or signal exists
```python
# Check message existence
if 'GpsStatus' in blf:
    print("GpsStatus message found")

# Check signal existence
proxy = blf['GpsStatus']
if 'GpsPosMode' in proxy:
    print("GpsPosMode signal found")
```

#### Get signal names from proxy
```python
proxy = blf['GpsStatus']
signals = proxy.get_signal_names()
# Equivalent to: blf.get_signals('GpsStatus')
```

#### Efficient repeated access (automatic caching)
```python
# First access builds and caches the data
data1 = blf['GpsStatus']['GpsPosMode']

# Subsequent accesses use cached data (very fast)
data2 = blf['GpsStatus']['GpsPosMode']
# data1 and data2 are the same NumPy array (no copy)

# Metadata is also cached
units1 = blf['GpsStatus'].get_signal_units()
units2 = blf['GpsStatus'].get_signal_units()
# units1 is units2  # True (same dict object)
```

## API Reference

### BLF Class

```python
class BLF:
    def __init__(self, filepath: str | Path,
                 channel_dbc_list: list[tuple[int, str | Path]]) -> None:
        """Initialize BLF reader and decode the file."""

    # Message information
    def get_message_names(self) -> list[str]:
        """Get list of all decoded message names."""

    @property
    def messages(self) -> list[str]:
        """Get message names (backward compatibility)."""

    def get_message_count(self, message_name: str) -> int:
        """Get number of samples for a message."""

    def get_signals(self, message_name: str) -> list[str]:
        """Get list of signal names in a message."""

    # Signal data access
    def get_signal(self, message_name: str, signal_name: str) -> NDArray[np.float64]:
        """Get signal data as NumPy array."""

    def get_time_series(self, message_name: str) -> NDArray[np.float64]:
        """Get timestamps for a message."""

    def get_message(self, message_name: str) -> NDArray[np.float64]:
        """Get entire message as 2D array (time + all signals)."""

    def get_all_messages(self) -> dict[str, NDArray[np.float64]]:
        """Get all messages as dictionary."""

    # Metadata access
    def get_signal_unit(self, message_name: str, signal_name: str) -> str:
        """Get unit string for a signal."""

    def get_signal_factor(self, message_name: str, signal_name: str) -> float:
        """Get scaling factor for a signal."""

    def get_signal_offset(self, message_name: str, signal_name: str) -> float:
        """Get scaling offset for a signal."""

    def get_period(self, message_name: str) -> int:
        """Get sampling period for a message in milliseconds."""

    # Dictionary-style access
    def __getitem__(self, message_name: str) -> MessageProxy:
        """Get MessageProxy for dictionary-style access."""

    def __contains__(self, message_name: str) -> bool:
        """Check if message exists."""

    # Display
    def info(self) -> None:
        """Print detailed information about the BLF file."""
```

### MessageProxy Class

```python
class MessageProxy:
    """Proxy for dictionary-style signal access."""

    # Signal data access
    def get_signal(self, signal_name: str) -> NDArray[np.float64]:
        """Get signal data."""

    def get_signal_names(self) -> list[str]:
        """Get list of signal names."""

    def get_all_signals(self) -> NDArray[np.float64]:
        """Get all signals as 2D array."""

    # Metadata access (cached)
    def get_signal_units(self) -> dict[str, str]:
        """Get all signal units as dictionary."""

    def get_signal_unit(self, signal_name: str) -> str:
        """Get unit for a specific signal."""

    def get_signal_factors(self) -> dict[str, float]:
        """Get all signal factors as dictionary."""

    def get_signal_factor(self, signal_name: str) -> float:
        """Get factor for a specific signal."""

    def get_signal_offsets(self) -> dict[str, float]:
        """Get all signal offsets as dictionary."""

    def get_signal_offset(self, signal_name: str) -> float:
        """Get offset for a specific signal."""

    def get_period(self) -> int:
        """Get sampling period for this message in milliseconds."""

    # Dictionary-style access
    def __getitem__(self, signal_name: str) -> NDArray[np.float64]:
        """Get signal using proxy['SignalName'] syntax."""

    def __contains__(self, signal_name: str) -> bool:
        """Check if signal exists."""
```

## Performance

Performance comparison with cantools + python-can on a real BLF file:

| Library | Parse Time | Speedup |
|---------|-----------|---------|
| cantools + python-can | 858 ms | 1.0x |
| blf_python | 58 ms | **14.8x** |

Signal retrieval is also optimized with zero-copy NumPy array views, automatic caching, and efficient C++ data structures.

## Example: Complete Workflow

```python
from blf_python import BLF
import numpy as np
import matplotlib.pyplot as plt

# Load BLF file
print("Loading BLF file...")
blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])

# Display summary
print("\n" + "="*60)
blf.info()
print("="*60)

# Analyze GPS data
print("\nAnalyzing GPS data...")
timestamps = blf.get_time_series('GpsStatus')
gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
num_sats = blf.get_signal('GpsStatus', 'GpsNumSats')

print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"GPS samples: {len(timestamps)}")
print(f"Average satellites: {np.mean(num_sats):.1f}")
print(f"GPS mode (most common): {np.bincount(gps_mode.astype(int)).argmax()}")

# Plot velocity over time
print("\nPlotting velocity...")
vel_timestamps = blf.get_time_series('Velocity')
vel_north = blf.get_signal('Velocity', 'VelNorth')
vel_east = blf.get_signal('Velocity', 'VelEast')
vel_down = blf.get_signal('Velocity', 'VelDown')

# Calculate total velocity
vel_total = np.sqrt(vel_north**2 + vel_east**2 + vel_down**2)

plt.figure(figsize=(12, 6))
plt.plot(vel_timestamps, vel_total, label='Total Velocity')
plt.plot(vel_timestamps, vel_north, label='North', alpha=0.7)
plt.plot(vel_timestamps, vel_east, label='East', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Components Over Time')
plt.legend()
plt.grid(True)
plt.savefig('velocity_plot.png', dpi=150)
print("Plot saved to velocity_plot.png")
```

## Requirements

- Python 3.10 or higher
- NumPy
- CMake 3.15 or higher
- C++17 compatible compiler
- Vector DBC library (included)
- Vector BinLog library (included)

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please visit the project repository.
