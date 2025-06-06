from pathlib import Path


def with_suffix(filepath: Path, suffix: str) -> Path:
    """Replace the suffix of the filepath, and with support for multiple suffixes (unlike `Path.with_suffix` method).

    Args:
        filepath: Path to the file.
        suffix: New suffix to use.

    Returns:
        Path to the file with the new suffix.
    """
    for _ in filepath.suffixes:
        filepath.with_suffix("")
    return filepath.with_suffix(suffix)


def append(filepath: Path, suffix: str) -> Path:
    """Append a suffix to the filepath, before the suffixes, i.e. file extensions.

    Args:
        filepath: Path to the file.
        suffix: Suffix to append.

    Returns:
        Path to the file with the suffix appended.
    """
    return filepath.parent / "".join([with_suffix(filepath, "").stem, suffix, *filepath.suffixes])
