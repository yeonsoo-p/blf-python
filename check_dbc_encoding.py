#!/usr/bin/env python3
"""
DBC File Encoding Checker

This script scans DBC files for non-UTF-8 characters and reports their locations.
Useful for diagnosing encoding issues in DBC database files.

Usage:
    python check_dbc_encoding.py <dbc_file>
    python check_dbc_encoding.py <directory>  # scans all .dbc files
"""

import sys
from pathlib import Path


def check_utf8_encoding(file_path: Path) -> list[tuple[int, str, bytes]]:
    """
    Check a file for non-UTF-8 characters.

    Args:
        file_path: Path to the file to check

    Returns:
        list of tuples (line_number, line_content, problematic_bytes)
    """
    issues = []

    try:
        with open(file_path, 'rb') as f:
            for line_num, line_bytes in enumerate(f, start=1):
                try:
                    # Try to decode as UTF-8
                    line_str = line_bytes.decode('utf-8')
                except UnicodeDecodeError as e:
                    # Found non-UTF-8 bytes
                    # Get the problematic portion
                    problem_start = max(0, e.start - 10)
                    problem_end = min(len(line_bytes), e.end + 10)
                    context = line_bytes[problem_start:problem_end]

                    # Try to decode what we can for display
                    try:
                        line_str = line_bytes.decode('utf-8', errors='replace')
                    except:
                        line_str = str(line_bytes)

                    issues.append((line_num, line_str.strip(), context))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return issues


def format_bytes_hex(data: bytes) -> str:
    """Format bytes as hex string."""
    return ' '.join(f'0x{b:02x}' for b in data)


def check_dbc_file(dbc_path: Path) -> None:
    """
    Check a single DBC file for encoding issues.

    Args:
        dbc_path: Path to the DBC file
    """
    print(f"\nChecking: {dbc_path}")
    print("=" * 80)

    issues = check_utf8_encoding(dbc_path)

    if not issues:
        print("[OK] No encoding issues found - file is valid UTF-8")
        return

    print(f"[FAIL] Found {len(issues)} line(s) with non-UTF-8 characters:\n")

    for line_num, line_content, problem_bytes in issues:
        print(f"Line {line_num}:")
        print(f"  Content: {line_content[:100]}")
        if len(line_content) > 100:
            print(f"           ... (truncated)")
        print(f"  Problem bytes: {format_bytes_hex(problem_bytes)}")

        # Try to decode with common encodings
        decodings = []
        for encoding in ['latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']:
            try:
                decoded = problem_bytes.decode(encoding)
                decodings.append(f"{encoding}: '{decoded}'")
            except:
                pass

        if decodings:
            print(f"  Possible decodings:")
            for dec in decodings:
                print(f"    - {dec}")
        print()


def find_dbc_files(path: Path) -> list[Path]:
    """
    Find all DBC files in a path.

    Args:
        path: File or directory path

    Returns:
        list of DBC file paths
    """
    if path.is_file():
        if path.suffix.lower() == '.dbc':
            return [path]
        else:
            print(f"Error: {path} is not a .dbc file")
            return []
    elif path.is_dir():
        dbc_files = list(path.glob('**/*.dbc'))
        if not dbc_files:
            print(f"No .dbc files found in {path}")
        return dbc_files
    else:
        print(f"Error: {path} does not exist")
        return []


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide a DBC file or directory path")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    dbc_files = find_dbc_files(input_path)

    if not dbc_files:
        sys.exit(1)

    print(f"\nFound {len(dbc_files)} DBC file(s) to check")

    total_issues = 0
    files_with_issues = 0
    for dbc_file in sorted(dbc_files):
        check_dbc_file(dbc_file)
        # Quick count
        issues = check_utf8_encoding(dbc_file)
        if issues:
            files_with_issues += 1
            total_issues += len(issues)

    print("\n" + "=" * 80)
    print(f"Summary: Checked {len(dbc_files)} file(s)")
    if files_with_issues > 0:
        print(f"[FAIL] Found issues in {files_with_issues} file(s)")
        print(f"Total lines with encoding issues: {total_issues}")
    else:
        print("[OK] All files are valid UTF-8")


if __name__ == '__main__':
    main()
