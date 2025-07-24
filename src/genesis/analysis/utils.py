from pathlib import Path


def find_graph_file(
    input_file: Path,
    *,
    search_dirs: list[Path] | None = None,
    pattern: str = "*{id}*.json",
) -> Path:
    """Find a unique graph JSON file based on the input file.

    Resolve either:
        - a direct file path (if input_file.exists()), or
        - a patient ID (zero-padded to 4 digits) to any JSON matching `pattern`.

    Args:
        input_file: either a Path to an existing file or a Path whose stem is a patient ID.
        search_dirs: list of directories to search under; defaults to standard PERSEVERE/raw locations.
        pattern: a glob pattern containing '{id}' which will be replaced by the zero-padded ID.
                 e.g. "*{id}*_enriched_graph.json" or the default "*{id}*.json"

    Returns:
        The unique matching JSON Path.

    Raises:
        FileNotFoundError if no match, or
        FileNotFoundError if more than one unique match is found.
    """
    # 1) If they've passed a real file, just use it
    if input_file.is_file():
        return input_file.resolve()

    # 2) Otherwise interpret the stem as an ID
    patient_id = input_file.stem.zfill(4)
    pattern = pattern.format(id=patient_id)

    # default search locations
    if search_dirs is None:
        search_dirs = [
            Path("data/PERSEVERE/raw"),
            Path("../../../data/PERSEVERE/raw"),
        ]

    found = []
    for d in search_dirs:
        if d.is_dir():
            found.extend(d.rglob(pattern))

    unique = {p.resolve() for p in found}
    if not unique:
        raise FileNotFoundError(f"No graph JSON found for ID='{patient_id}' (pattern='{pattern}').")
    if len(unique) > 1:
        raise FileNotFoundError(f"Multiple matches for ID='{patient_id}' (pattern='{pattern}'): {list(unique)}")
    return unique.pop()
