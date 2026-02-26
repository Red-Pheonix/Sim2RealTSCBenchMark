import json
from pathlib import Path
from typing import List, Tuple, Union

def parse_pairs(text: str) -> List[Tuple[int, int]]:
    pairs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 2:
            continue
        t = int(parts[0])
        ph = int(parts[1])
        pairs.append((t, ph))
    pairs.sort(key=lambda x: x[0])
    return pairs

def merge_same_phase(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not pairs:
        return []
    merged = [pairs[0]]
    for t, ph in pairs[1:]:
        if ph == merged[-1][1]:
            continue
        merged.append((t, ph))
    return merged

def to_single_arrows(pairs: List[Tuple[int, int]]) -> dict:
    arrows = []
    for (t0, p0), (t1, p1) in zip(pairs, pairs[1:]):
        arrows.append({
            "from_phase": p0 + 1,
            "from_time": t0,
            "to_phase": p1 + 1,
            "to_time": t1,
        })
    return {"single_arrows": arrows}

def convert(source: Union[str, Path], *, merge_phases: bool = True, source_is_file: bool = True) -> dict:
    """
    source_is_file=True  -> treat `source` as a filename/path and read it
    source_is_file=False -> treat `source` as raw CSV-like text
    """
    if source_is_file:
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)

    pairs = parse_pairs(text)
    if merge_phases:
        pairs = merge_same_phase(pairs)
    return to_single_arrows(pairs)

def convert_file(in_path: str, out_path: str = "single_arrows.json", *, merge_phases: bool = True) -> None:
    data = convert(in_path, merge_phases=merge_phases, source_is_file=True)
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(data['single_arrows'])} arrows)")

# ---- usage ----
convert_file("actions_log.csv", "single_arrows.json", merge_phases=True)