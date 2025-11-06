"""
Test script for BLF Python module

This script demonstrates how to:
1. Load a BLF file with DBC database
2. Access message and signal data
3. Plot signal data using matplotlib
"""

import sys
from pathlib import Path
import time
import argparse
import matplotlib.pyplot as plt
import cantools
import can
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from blf_python import BLF


def test_basic_loading(blf_file, channel_dbc_list):
    """Test basic BLF file loading and data access."""
    print("=" * 60)
    print("Test 1: Basic BLF Loading")
    print("=" * 60)

    # Check if files exist
    if not blf_file.exists():
        print(f"ERROR: BLF file not found: {blf_file}")
        print("Please update the blf_file variable with your actual BLF file path.")
        return None

    for channel, dbc_file in channel_dbc_list:
        if not dbc_file.exists():
            print(f"ERROR: DBC file not found: {dbc_file}")
            print("Please update the channel_dbc_list with valid DBC file paths.")
            return None

    try:
        # Load BLF file
        print(f"\nLoading BLF file: {blf_file}")
        channels_str = ', '.join(str(ch) if ch >= 0 else 'All' for ch, _ in channel_dbc_list)
        dbc_files_str = ', '.join(Path(dbc).name for _, dbc in channel_dbc_list)
        print(f"Channels: {channels_str}")
        print(f"DBC file(s): {dbc_files_str}")

        # Benchmark: Parsing time
        start_time = time.perf_counter()
        blf = BLF(blf_file, channel_dbc_list)
        parse_time = time.perf_counter() - start_time

        print(f"\n[BENCHMARK] Parsing time: {parse_time:.3f} seconds")

        # Print summary
        print("\n" + "=" * 60)
        blf.info()
        print("=" * 60)

        return blf

    except Exception as e:
        print(f"\nERROR: Failed to load BLF file")
        print(f"Error message: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_data_access(blf):
    """Test accessing message and signal data."""
    if blf is None:
        return

    print("\n" + "=" * 60)
    print("Test 2: Data Access")
    print("=" * 60)

    if not blf.messages:
        print("No messages found in BLF file!")
        return

    # Show first message details
    msg_name = blf.messages[0]
    print(f"\nExamining message: {msg_name}")
    print(f"  Number of samples: {blf.get_message_count(msg_name)}")
    print(f"  Signals: {blf.get_signals(msg_name)}")

    # Access timestamp data
    timestamps = blf.get_time_series(msg_name)
    print(f"\n  Timestamps:")
    print(f"    Shape: {timestamps.shape}")
    print(f"    Dtype: {timestamps.dtype}")
    print(f"    Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    print(f"    First timestamp: {timestamps[0]:.6f} s")
    print(f"    Last timestamp: {timestamps[-1]:.6f} s")

    # Access first signal data
    signals = blf.get_signals(msg_name)
    if len(signals) > 1:  # More than just 'Time'
        signal_name = [s for s in signals if s != "Time"][0]

        # Benchmark: Signal retrieval time
        start_time = time.perf_counter_ns()
        signal_data = blf[msg_name][signal_name]
        retrieval_time = time.perf_counter_ns() - start_time

        print(f"\n  Signal: {signal_name}")
        print(f"    Shape: {signal_data.shape}")
        print(f"    Dtype: {signal_data.dtype}")
        print(f"    Min: {signal_data.min()}")
        print(f"    Max: {signal_data.max()}")
        print(f"    Mean: {signal_data.mean():.3f}")
        print(f"    First 5 values: {signal_data[:5]}")
        print(f"    [BENCHMARK] Retrieval time: {retrieval_time:,} ns ({retrieval_time / 1e6:.3f} ms)")


def plot_signal(blf, msg_name="Distance", signal_name="Distance"):
    """Plot a specific signal over time."""
    if blf is None:
        return

    print("\n" + "=" * 60)
    print("Test 3: Plotting Signal Data")
    print("=" * 60)

    # Check if message exists
    if msg_name not in blf.messages:
        print(f"\nMessage '{msg_name}' not found!")
        print(f"Available messages: {blf.messages}")

        # Use first available message instead
        if blf.messages:
            msg_name = blf.messages[0]
            print(f"\nUsing first available message: {msg_name}")
        else:
            print("No messages available to plot!")
            return

    # Check if signal exists
    signals = blf.get_signals(msg_name)
    if signal_name not in signals:
        print(f"\nSignal '{signal_name}' not found in message '{msg_name}'!")
        print(f"Available signals: {signals}")

        # Use first non-Time signal instead
        non_time_signals = [s for s in signals if s != "Time"]
        if non_time_signals:
            signal_name = non_time_signals[0]
            print(f"\nUsing first available signal: {signal_name}")
        else:
            print("No signals available to plot!")
            return

    # Get data
    timestamps = blf.get_time_series(msg_name)
    signal_data = blf.get_signal_data(msg_name, signal_name)

    print(f"\nPlotting: {msg_name}.{signal_name}")
    print(f"  Samples: {len(signal_data)}")
    print(f"  Time range: {timestamps[0]:.3f} - {timestamps[-1]:.3f} seconds")

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, signal_data, linewidth=0.8, alpha=0.8)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel(signal_name, fontsize=12)
    plt.title(f"{msg_name}: {signal_name} over Time", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plot_filename = f"plot_{msg_name}_{signal_name}.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlot saved to: {plot_filename}")

    # Show plot
    plt.show()


def plot_multiple_signals(blf, msg_name=None):
    """Plot all signals from a message in subplots."""
    if blf is None:
        return

    print("\n" + "=" * 60)
    print("Test 4: Plotting Multiple Signals")
    print("=" * 60)

    # Use first message if not specified
    if msg_name is None or msg_name not in blf.messages:
        if blf.messages:
            msg_name = blf.messages[0]
            print(f"\nUsing message: {msg_name}")
        else:
            print("No messages available!")
            return

    # Get all signals except Time
    all_signals = blf.get_signals(msg_name)
    signals = [s for s in all_signals if s != "Time"]

    if not signals:
        print(f"No signals to plot in message '{msg_name}'")
        return

    print(f"Plotting {len(signals)} signals from '{msg_name}'")

    # Get timestamps
    timestamps = blf.get_time_series(msg_name)

    # Create subplots
    n_signals = len(signals)
    n_cols = min(2, n_signals)
    n_rows = (n_signals + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_signals == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each signal
    for idx, signal_name in enumerate(signals):
        signal_data = blf.get_signal_data(msg_name, signal_name)

        ax = axes[idx]
        ax.plot(timestamps, signal_data, linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(signal_name)
        ax.set_title(f"{signal_name}", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(signals), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"All Signals from Message: {msg_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save plot
    plot_filename = f"plot_{msg_name}_all_signals.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"\nPlot saved to: {plot_filename}")

    plt.show()


def compare_with_cantools(blf_file, channel_dbc_list):
    """Compare performance with cantools + python-can."""
    print("\n" + "=" * 60)
    print("Test 5: Performance Comparison with cantools")
    print("=" * 60)

    # Load DBC databases from channel_dbc_list
    dbc_dbs = [cantools.database.load_file(str(dbc)) for _, dbc in channel_dbc_list]

    print("\n[cantools + python-can]")
    print(f"Loading BLF file: {blf_file}")

    # Benchmark cantools approach
    start_time = time.perf_counter()

    # Storage for decoded data
    messages_data = {}

    # Read BLF file with python-can
    msg_count = 0
    decoded_count = 0
    seen_ids = set()
    channel_filtered = 0

    channels_seen = set()

    for msg in can.LogReader(str(blf_file)):
        msg_count += 1
        seen_ids.add(msg.arbitration_id)
        channels_seen.add(msg.channel)

        # Skip channel filtering for cantools to get a working comparison
        # (Our C++ implementation correctly handles channel 4)
        # if channel >= 0 and msg.channel != channel:
        #     channel_filtered += 1
        #     continue

        # Try to decode with each DBC database
        decoded = None
        msg_name = None
        for db in dbc_dbs:
            try:
                # Allow truncated messages to handle variable-length CAN data
                decoded = db.decode_message(msg.arbitration_id, msg.data, allow_truncated=True)
                msg_name = db.get_message_by_frame_id(msg.arbitration_id).name
                decoded_count += 1
                break
            except (KeyError, ValueError):
                continue

        if decoded and msg_name:
            # Initialize message storage
            if msg_name not in messages_data:
                messages_data[msg_name] = {"Time": []}

            # Store timestamp
            messages_data[msg_name]["Time"].append(msg.timestamp)

            # Store signals (create list if needed for new signals)
            for sig_name, sig_value in decoded.items():
                if sig_name not in messages_data[msg_name]:
                    # Pad with None for previous messages that didn't have this signal
                    messages_data[msg_name][sig_name] = [None] * (len(messages_data[msg_name]["Time"]) - 1)
                messages_data[msg_name][sig_name].append(sig_value)

    print(f"Total CAN messages read: {msg_count}")
    print(f"Channels seen: {sorted(channels_seen)}")
    print(f"Unique CAN IDs seen: {len(seen_ids)}")
    print(f"Messages successfully decoded: {decoded_count}")

    # Convert lists to numpy arrays
    for msg_name in messages_data:
        for sig_name in messages_data[msg_name]:
            messages_data[msg_name][sig_name] = np.array(messages_data[msg_name][sig_name])

    cantools_parse_time = time.perf_counter() - start_time

    print(f"[BENCHMARK] Parsing time: {cantools_parse_time:.3f} seconds")
    print(f"Messages decoded: {len(messages_data)}")

    # Now benchmark our implementation
    print("\n[blf_python (our implementation)]")
    print(f"Loading BLF file: {blf_file}")

    start_time = time.perf_counter()
    blf = BLF(blf_file, channel_dbc_list)
    our_parse_time = time.perf_counter() - start_time

    print(f"[BENCHMARK] Parsing time: {our_parse_time:.3f} seconds")
    print(f"Messages decoded: {len(blf.messages)}")

    # Calculate speedup
    speedup = cantools_parse_time / our_parse_time
    print(f"\n{'=' * 60}")
    print(f"PERFORMANCE SUMMARY:")
    print(f"  cantools + python-can: {cantools_parse_time:.3f} seconds")
    print(f"  blf_python (ours):     {our_parse_time:.3f} seconds")
    print(f"  Speedup:               {speedup:.2f}x faster")
    print(f"{'=' * 60}")

    # Benchmark signal retrieval
    if messages_data and blf.messages:
        msg_name = list(messages_data.keys())[0]
        sig_name = [s for s in messages_data[msg_name].keys() if s != "Time"][0]

        # cantools retrieval (already in memory as numpy array)
        start_time = time.perf_counter_ns()
        _ = messages_data[msg_name][sig_name]
        cantools_retrieval = time.perf_counter_ns() - start_time

        # Our implementation retrieval
        start_time = time.perf_counter_ns()
        _ = blf[msg_name][sig_name]
        our_retrieval = time.perf_counter_ns() - start_time

        print(f"\nSignal retrieval benchmark ({msg_name}.{sig_name}):")
        print(f"  cantools:     {cantools_retrieval:,} ns")
        print(f"  blf_python:   {our_retrieval:,} ns")

    # Create consolidated comparison plot
    print(f"\n{'=' * 60}")
    print("Creating comparison plot...")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot Distance.Distance
    if "Distance" in messages_data and "Distance" in messages_data["Distance"]:
        # Row 1, Col 1: cantools Distance
        ax = axes[0, 0]
        ax.plot(messages_data["Distance"]["Time"], messages_data["Distance"]["Distance"], linewidth=0.8, alpha=0.8, color="tab:blue")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Distance", fontsize=10)
        ax.set_title("Distance.Distance (cantools + python-can)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"Parse time: {cantools_parse_time:.3f}s", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if "Distance" in blf and "Distance" in blf["Distance"]:
        # Row 2, Col 1: ours Distance
        ax = axes[1, 0]
        ax.plot(blf["Distance"]["Time"], blf["Distance"]["Distance"], linewidth=0.8, alpha=0.8, color="tab:orange")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Distance", fontsize=10)
        ax.set_title("Distance.Distance (blf_python - ours)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"Parse time: {our_parse_time:.3f}s", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    # Plot AccelVehicle.AccelX
    if "AccelVehicle" in messages_data and "AccelX" in messages_data["AccelVehicle"]:
        # Row 1, Col 2: cantools AccelX
        ax = axes[0, 1]
        ax.plot(messages_data["AccelVehicle"]["Time"], messages_data["AccelVehicle"]["AccelX"], linewidth=0.8, alpha=0.8, color="tab:blue")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("AccelX", fontsize=10)
        ax.set_title("AccelVehicle.AccelX (cantools + python-can)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"Parse time: {cantools_parse_time:.3f}s", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if "AccelVehicle" in blf and "AccelX" in blf["AccelVehicle"]:
        # Row 2, Col 2: ours AccelX
        ax = axes[1, 1]
        ax.plot(blf["AccelVehicle"]["Time"], blf["AccelVehicle"]["AccelX"], linewidth=0.8, alpha=0.8, color="tab:orange")
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("AccelX", fontsize=10)
        ax.set_title("AccelVehicle.AccelX (blf_python - ours)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"Parse time: {our_parse_time:.3f}s", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    plt.suptitle(f"Performance Comparison: cantools vs blf_python (Speedup: {speedup:.2f}x)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_filename = "comparison_cantools_vs_blf_python.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"\nComparison plot saved to: {plot_filename}")
    plt.show()

    return blf, messages_data


def main():
    """Run all tests."""
    # Hardcoded test parameters
    blf_file = Path("example/RT3003_0223.blf")

    # List of (channel, dbc_file) tuples
    # Each tuple specifies which DBC to use for which channel
    # Use -1 for wildcard (matches all channels)
    #
    # Examples:
    #   Single channel: [(4, Path("vehicle.dbc"))]
    #   Multiple channels: [(1, Path("powertrain.dbc")), (2, Path("chassis.dbc"))]
    #   Wildcard: [(-1, Path("all_messages.dbc"))]
    channel_dbc_list = [
        (-1, Path("example/RT3003_240223dbc.dbc")),
    ]

    print("\n" + "=" * 60)
    print("BLF Python Module Test Suite")
    print("=" * 60)

    # Test 1: Load BLF file
    blf = test_basic_loading(blf_file, channel_dbc_list)

    if blf is None:
        print("\nTests aborted due to loading failure.")
        return

    # Test 2: Access data
    test_data_access(blf)

    # Test 3: Performance comparison with cantools (includes consolidated plot)
    compare_with_cantools(blf_file, channel_dbc_list)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
