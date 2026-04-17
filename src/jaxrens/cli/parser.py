"""Raw key=value reader for jaxnest-style ns.inp files.

Only ``parse_input_file`` is public.  The old dataclass-construction helpers
(``raw_to_configs``, ``load_config``) have been removed; use
``migrate_ns_inp`` + ``RootConfig.model_validate`` instead.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse_input_file(path: Path | str) -> dict[str, str]:
    """Parse a key=value input file into a raw dict.

    Handles:
    - Comments (#)
    - Blank lines
    - Whitespace around = sign
    - Inline comments

    Args:
        path: Path to ns.inp file.

    Returns:
        Dict of string key-value pairs.
    """
    path = Path(path)
    result: dict[str, str] = {}
    pattern = re.compile(r"^\s*(\S+)\s*=\s*(.*\S)\s*$")

    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                result[m.group(1)] = m.group(2)

    return result
