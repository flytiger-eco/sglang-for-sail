import json
import re
import sys
from pathlib import Path


def verify(path):
    results = []
    pending = None
    summary = None
    for raw in Path(path).open(errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        match = re.fullmatch(r"Ran (\d+) tests? in [0-9.]+s", line)
        if match:
            if pending is not None:
                raise ValueError("Incomplete unittest result")
            pending = int(match[1])
        elif pending is not None and re.fullmatch(r"OK(?: \(.*\))?", line):
            skipped = re.search(r"skipped=(\d+)", line)
            executed = pending - (int(skipped[1]) if skipped else 0)
            if executed <= 0:
                raise ValueError("A nightly file executed no tests")
            results.append(executed)
            pending = None
        elif re.match(r"FAILED(?: \(|$)", line):
            raise ValueError("A nightly file failed")
        match = re.search(r"Test Summary: (\d+)/(\d+) passed$", line)
        if match:
            summary = (int(match[1]), int(match[2]))
    if (
        pending is not None
        or not summary
        or summary[0] != summary[1]
        or summary[1] == 0
        or len(results) != summary[1]
    ):
        raise ValueError(
            "Missing, empty, failed, or incomplete nightly execution evidence"
        )
    return {"files": len(results), "executed": sum(results), "success": True}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(sys.argv[1])))
    except (OSError, ValueError) as error:
        sys.exit(str(error))
