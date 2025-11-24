"""
BLF File Reader and Decoder

This module provides a high-level interface for reading and decoding BLF (Binary Logging Format)
files using DBC (CAN Database) files.

Example:
    >>> from blf_python import BLF
    >>> blf = BLF('recording.blf', [(-1, 'example/IMU.dbc')])
    >>> # Access signal data
    >>> timestamps = blf.get_signal('GpsStatus', 'Time')
    >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
    >>> # Or use dictionary-style access
    >>> velocity = blf['Distance']['Distance']
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from . import blf_python as _blf_c


class MessageProxy:
    """Proxy object for dictionary-style access to signals within a message."""

    def __init__(self, blf: _blf_c.BLF, message_name: str) -> None:
        # Validate message_name type early to prevent cryptic C++ errors
        if not isinstance(message_name, str):
            raise TypeError(f"message_name must be str, got {type(message_name).__name__}")
        if "\x00" in message_name:
            raise ValueError(f"message_name cannot contain null bytes: {message_name!r}")

        self._blf: _blf_c.BLF = blf
        self._message_name: str = message_name
        self._cached_data: NDArray[np.float64] | None = None  # Cache for 2D array from get_all_signals()
        self._cached_units: dict[str, str] | None = None  # Cache for units dict
        self._cached_factors: dict[str, float] | None = None  # Cache for factors dict
        self._cached_offsets: dict[str, float] | None = None  # Cache for offsets dict

    # ========================================================================
    # Discovery Methods
    # ========================================================================

    def get_signal_names(self) -> list[str]:
        """Get list of all signal names in this message."""
        return self._blf.get_signals(self._message_name)

    # ========================================================================
    # Data Access Methods
    # ========================================================================

    def get_signal(self, signal_name: str) -> NDArray[np.float64]:
        """Get signal data for a specific signal."""
        # Just delegate to C++ - it returns zero-copy strided view
        return self._blf.get_signal(self._message_name, signal_name)

    def get_all_signals(self) -> NDArray[np.float64]:
        """Get all signals as a 2D array (time + all signals)."""
        # Cache the 2D array for repeated access
        if self._cached_data is None:
            self._cached_data = self._blf.get_message_data(self._message_name)
        return self._cached_data

    # ========================================================================
    # Metadata Methods
    # ========================================================================

    def get_signal_units(self) -> dict[str, str]:
        """Get dictionary of signal name -> unit string."""
        if self._cached_units is None:
            try:
                self._cached_units = self._blf.get_signal_units(self._message_name)
            except (TypeError, RuntimeError, ValueError) as e:
                raise type(e)(f"Failed to get signal units for message '{self._message_name}': {e}") from e
        return self._cached_units

    def get_signal_unit(self, signal_name: str) -> str:
        """Get unit string for a specific signal."""
        units = self.get_signal_units()
        if signal_name not in units:
            raise KeyError(f"Signal '{signal_name}' not found in message '{self._message_name}'")
        return units[signal_name]

    def get_signal_factors(self) -> dict[str, float]:
        """Get dictionary of signal name -> scaling factor."""
        if self._cached_factors is None:
            try:
                self._cached_factors = self._blf.get_signal_factors(self._message_name)
            except (TypeError, RuntimeError, ValueError) as e:
                raise type(e)(f"Failed to get signal factors for message '{self._message_name}': {e}") from e
        return self._cached_factors

    def get_signal_factor(self, signal_name: str) -> float:
        """Get scaling factor for a specific signal."""
        factors = self.get_signal_factors()
        if signal_name not in factors:
            raise KeyError(f"Signal '{signal_name}' not found in message '{self._message_name}'")
        return factors[signal_name]

    def get_signal_offsets(self) -> dict[str, float]:
        """Get dictionary of signal name -> scaling offset."""
        if self._cached_offsets is None:
            try:
                self._cached_offsets = self._blf.get_signal_offsets(self._message_name)
            except (TypeError, RuntimeError, ValueError) as e:
                raise type(e)(f"Failed to get signal offsets for message '{self._message_name}': {e}") from e
        return self._cached_offsets

    def get_signal_offset(self, signal_name: str) -> float:
        """Get scaling offset for a specific signal."""
        offsets = self.get_signal_offsets()
        if signal_name not in offsets:
            raise KeyError(f"Signal '{signal_name}' not found in message '{self._message_name}'")
        return offsets[signal_name]

    def get_period(self) -> int:
        """
        Get sampling period for this message in milliseconds.

        Calculates the average sampling period by computing:
        period = ((last_timestamp - first_timestamp) / (num_samples - 1)) * 1000

        Returns:
            Sampling period in milliseconds (rounded to nearest integer)

        Raises:
            ValueError: If message has insufficient samples or invalid time range

        Example:
            >>> period = blf['GpsStatus'].get_period()
            >>> print(f"Sampling period: {period} ms")
        """
        return self._blf.get_period(self._message_name)

    # ========================================================================
    # Special Methods
    # ========================================================================

    def __getitem__(self, signal_name: str) -> NDArray[np.float64]:
        """Get signal data using dictionary-style access."""
        return self.get_signal(signal_name)

    def __contains__(self, signal_name: str) -> bool:
        """Check if a signal exists in this message."""
        try:
            signals = self._blf.get_signals(self._message_name)
            return signal_name in signals
        except KeyError:
            return False

    def __repr__(self) -> str:
        """String representation of MessageProxy."""
        signals = self._blf.get_signals(self._message_name)
        return f"MessageProxy(message='{self._message_name}', signals={len(signals)})"


class BLF:
    """
    BLF file reader with DBC decoding support.

    This class reads BLF files and decodes CAN messages using provided DBC database files.
    The decoded data is accessed via zero-copy NumPy arrays.

    The class implements instance caching - opening the same file path multiple times
    returns the same cached instance, preserving all internal caches for efficiency.

    Attributes:
        messages (list[str]): List of all decoded message names

    Example:
        >>> blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])
        >>> print(blf.messages)  # List all decoded messages
        >>> print(blf.get_signals('GpsStatus'))  # List signals in a message
        >>> # Access signals
        >>> timestamps = blf.get_signal('GpsStatus', 'Time')
        >>> gps_mode = blf.get_signal('GpsStatus', 'GpsPosMode')
        >>> # Or use dictionary-style access
        >>> velocity = blf['Distance']['Distance']
        >>> # Get metadata
        >>> unit = blf.get_signal_unit('Distance', 'Distance')
        >>> factor = blf.get_signal_factor('Distance', 'Distance')
    """

    # Class-level cache for instances
    _instances: dict[tuple[Path, tuple[tuple[int, str], ...]], "BLF"] = {}

    def __new__(cls, filepath: str | Path, channel_dbc_list: list[tuple[int, str | Path]]) -> "BLF":
        """
        Create or return cached BLF instance.

        Args:
            filepath: Path to the BLF file (str or Path)
            channel_dbc_list: List of (channel, dbc_filepath) tuples

        Returns:
            BLF instance (cached if filepath and channel_dbc_list were previously used)
        """
        # Normalize filepath to absolute Path
        filepath_obj = Path(filepath).resolve()

        # Normalize channel_dbc_list to a hashable tuple of (channel, resolved_dbc_path)
        normalized_channels = tuple((channel, str(Path(dbc_path).resolve())) for channel, dbc_path in channel_dbc_list)

        # Create cache key from filepath and channel configuration
        cache_key = (filepath_obj, normalized_channels)

        # Return cached instance if it exists
        if cache_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cache_key] = instance

        return cls._instances[cache_key]

    def __init__(self, filepath: str | Path, channel_dbc_list: list[tuple[int, str | Path]]) -> None:
        """
        Initialize BLF reader and decode the file.

        The file is automatically parsed upon initialization. If this exact filepath
        and channel configuration was previously opened, the cached instance is returned
        with all data already parsed and cached.

        Args:
            filepath: Path to the BLF file to read (str or Path)
            channel_dbc_list: List of (channel, dbc_filepath) tuples
                Each tuple specifies which DBC file to use for which CAN channel.
                Use -1 for wildcard (matches all channels)

        Raises:
            FileNotFoundError: If BLF file or any DBC file doesn't exist
            ValueError: If channel_dbc_list is empty or contains invalid entries
            IOError: If BLF file cannot be opened or read

        Example:
            >>> # Single DBC for all channels (wildcard)
            >>> blf = BLF('recording.blf', [(-1, 'vehicle.dbc')])
            >>> # Specific channel
            >>> blf = BLF('recording.blf', [(4, 'vehicle.dbc')])
            >>> # Multiple channels with different DBCs
            >>> blf = BLF('recording.blf', [(1, 'powertrain.dbc'), (2, 'chassis.dbc')])

        Note:
            The BLF file is processed immediately upon instantiation.
            Each channel can have its own DBC file for decoding.
            Instance caching means opening the same file twice returns the same object.
        """
        # Only initialize if not already initialized (for cached instances)
        if hasattr(self, "_blf"):
            return  # Already initialized from cache

        # Convert to Path for validation
        filepath_obj = Path(filepath)
        if not filepath_obj.exists():
            raise FileNotFoundError(f"BLF file not found: {filepath}")

        if not channel_dbc_list:
            raise ValueError("At least one (channel, dbc_filepath) tuple must be provided")

        # Validate and convert channel_dbc_list
        validated_list = []
        dbc_files_for_display = []

        for entry in channel_dbc_list:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each entry must be a (channel, dbc_filepath) tuple")

            channel, dbc_path = entry

            if not isinstance(channel, int):
                raise ValueError("Channel must be an integer")

            dbc_path_obj = Path(dbc_path)
            if not dbc_path_obj.exists():
                raise FileNotFoundError(f"DBC file not found: {dbc_path}")

            validated_list.append((channel, str(dbc_path_obj)))
            dbc_files_for_display.append(str(dbc_path_obj))

        # Store parameters
        self.filepath: str = str(filepath_obj)
        self.dbc_files: list[str] = dbc_files_for_display
        self.channel_dbc_list: list[tuple[int, str]] = validated_list

        # Create C extension BLF object
        self._blf: _blf_c.BLF = _blf_c.BLF(str(filepath_obj), validated_list)

        # Cache for MessageProxy objects (to preserve their internal caches)
        self._message_proxies: dict[str, MessageProxy] = {}

    # ========================================================================
    # Discovery Methods - What's available in the BLF file
    # ========================================================================

    def get_message_names(self) -> list[str]:
        """Get list of all decoded message names."""
        return self._blf.get_message_names()

    @property
    def messages(self) -> list[str]:
        """Get list of all decoded message names (property for backward compatibility)."""
        return self.get_message_names()

    def get_signals(self, message_name: str) -> list[str]:
        """Get list of all signal names for a specific message."""
        return self._blf.get_signals(message_name)

    def get_message_count(self, message_name: str) -> int:
        """Get the number of samples for a specific message."""
        return self._blf.get_message_count(message_name)

    # ========================================================================
    # Data Access Methods - Getting signal and message data
    # ========================================================================

    def get_signal(self, message_name: str, signal_name: str) -> NDArray[np.float64]:
        """Get signal data as NumPy array with scaling applied."""
        # Use MessageProxy cache for efficiency
        return self[message_name].get_signal(signal_name)

    def get_time_series(self, message_name: str) -> NDArray[np.float64]:
        """Get timestamps for a specific message."""
        return self.get_signal(message_name, "Time")

    def get_message(self, message_name: str) -> NDArray[np.float64]:
        """
        Get entire message as 2D NumPy array (time + all signals).

        Args:
            message_name: Name of the message

        Returns:
            2D NumPy array with shape (num_samples, num_signals+1)
            Column 0 is timestamps, columns 1+ are signals
        """
        return self[message_name].get_all_signals()

    def get_all_messages(self) -> dict[str, NDArray[np.float64]]:
        """
        Get all messages as dictionary of message_name -> 2D array.

        Returns:
            Dictionary mapping message names to 2D NumPy arrays
        """
        return {msg: self.get_message(msg) for msg in self.get_message_names()}

    # ========================================================================
    # Metadata Methods - Signal units, factors, and offsets
    # ========================================================================

    def get_signal_unit(self, message_name: str, signal_name: str) -> str:
        """Get unit string for a signal."""
        return self[message_name].get_signal_unit(signal_name)

    def get_signal_factor(self, message_name: str, signal_name: str) -> float:
        """Get scaling factor for a signal."""
        return self[message_name].get_signal_factor(signal_name)

    def get_signal_offset(self, message_name: str, signal_name: str) -> float:
        """Get scaling offset for a signal."""
        return self[message_name].get_signal_offset(signal_name)

    def get_period(self, message_name: str) -> int:
        """
        Get sampling period for a message in milliseconds.

        Calculates the average sampling period by computing:
        period = ((last_timestamp - first_timestamp) / (num_samples - 1)) * 1000

        Args:
            message_name: Name of the message

        Returns:
            Sampling period in milliseconds (rounded to nearest integer)

        Raises:
            KeyError: If message not found
            ValueError: If message has insufficient samples or invalid time range

        Example:
            >>> period = blf.get_period('GpsStatus')
            >>> print(f"Sampling period: {period} ms")
        """
        return self._blf.get_period(message_name)

    # ========================================================================
    # Special Methods - Dictionary-style access and operators
    # ========================================================================

    def __getitem__(self, message_name: str) -> MessageProxy:
        """Get message proxy for dictionary-style signal access."""
        # Use cached MessageProxy to preserve internal data cache
        if message_name not in self._message_proxies:
            self._message_proxies[message_name] = MessageProxy(self._blf, message_name)
        return self._message_proxies[message_name]

    def __contains__(self, message_name: str) -> bool:
        """Check if a message exists in the BLF file."""
        return message_name in self.get_message_names()

    # ========================================================================
    # Display Methods - String representations and info
    # ========================================================================

    def __repr__(self) -> str:
        """String representation of BLF object."""
        channels = [ch for ch, _ in self.channel_dbc_list]
        return f"BLF(filepath='{self.filepath}', messages={len(self.get_message_names())}, channels={channels})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"BLF File: {self.filepath}"]
        channels_str = ", ".join(str(ch) for ch, _ in self.channel_dbc_list)
        lines.append(f"Channels: {channels_str}")
        lines.append(f"DBC Files: {', '.join(Path(f).name for f in self.dbc_files)}")
        lines.append(f"\nMessages ({len(self.get_message_names())}):")

        for msg_name in sorted(self.get_message_names()):
            signals = self.get_signals(msg_name)
            time_array = self.get_signal(msg_name, "Time")
            count = len(time_array)
            signal_count = len(signals) - 1  # Exclude 'Time'
            lines.append(f"  {msg_name}: {count} samples, {signal_count} signals")

        return "\n".join(lines)

    def info(self) -> None:
        """Print detailed information about the loaded BLF file."""
        print(str(self))
