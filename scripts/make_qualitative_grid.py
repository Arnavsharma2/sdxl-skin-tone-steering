#!/usr/bin/env python3
"""Build a deterministic, non-cherry-picked confirmatory example grid."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.study.analysis import load_jsonl_results

METHODS = ("prompt_only", "stepwise_unmasked", "stepwise_masked")
METHOD_LABELS = {
    "prompt_only": "Prompt-only",
    "stepwise_unmasked": "Stepwise, unmasked",
    "stepwise_masked": "Stepwise, masked",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Clean confirmatory run directory")
    parser.add_argument("analysis", type=Path, help="Frozen analysis directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/confirmatory_qualitative.png"),
    )
    parser.add_argument("--tile-size", type=int, default=320)
    return parser.parse_args()


def select_seed(matched: pd.DataFrame) -> tuple[int, pd.DataFrame, dict]:
    target = float(matched["target_change"].min())
    target_rows = matched[np.isclose(matched["target_change"], target)].copy()
    target_rows["match_complete"] = target_rows["match_complete"].fillna(False).astype(bool)
    complete = target_rows[target_rows["match_complete"]]
    coverage = complete.groupby("seed").size()
    if coverage.empty:
        raise ValueError("No seed has a complete matched-change condition")
    best_coverage = int(coverage.max())
    candidates = sorted(int(seed) for seed in coverage[coverage == best_coverage].index)
    candidate_rows = target_rows[target_rows["seed"].isin(candidates)]
    ita_by_seed = candidate_rows.groupby("seed")["original_ita"].median()
    median_ita = float(ita_by_seed.median())
    selected_seed = min(candidates, key=lambda seed: (abs(float(ita_by_seed[seed]) - median_ita), seed))
    selected = target_rows[target_rows["seed"] == selected_seed].copy()
    audit = {
        "selection_rule": (
            "Maximize the number of complete prespecified matched-change rows; "
            "among ties choose the seed closest to the median base ITA; then choose the lower seed."
        ),
        "target_change": target,
        "maximum_complete_rows": best_coverage,
        "candidate_seeds": candidates,
        "median_candidate_base_ita": median_ita,
        "selected_seed": selected_seed,
    }
    return selected_seed, selected, audit


def _load_tile(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_grid(
    run_dir: Path,
    selected_seed: int,
    selected: pd.DataFrame,
    *,
    tile_size: int,
) -> Image.Image:
    header_font = _font(18)
    footer_font = _font(16)
    columns = ("base", *METHODS)
    directions = (-1, 1)
    header_height = 30
    footer_height = 50
    row_height = tile_size + footer_height
    canvas = Image.new(
        "RGB",
        (tile_size * len(columns), header_height + row_height * len(directions)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Base", *(METHOD_LABELS[item] for item in METHODS))):
        draw.text((column * tile_size + 8, 5), label, fill="black", font=header_font)

    base_path = run_dir / "images" / f"seed_{selected_seed}" / "base.png"
    for row_index, direction in enumerate(directions):
        y = header_height + row_index * row_height
        base = _load_tile(base_path, tile_size)
        canvas.paste(base, (0, y))
        draw.text(
            (8, y + tile_size + 6),
            "Base (same seed)",
            fill="black",
            font=footer_font,
        )
        for column, method in enumerate(METHODS, start=1):
            match = selected[
                (selected["method"] == method) & (selected["direction"] == direction)
            ]
            if match.empty:
                continue
            record = match.iloc[0]
            image = _load_tile(run_dir / str(record["image_path"]), tile_size)
            canvas.paste(image, (column * tile_size, y))
            direction_label = "lighter" if direction < 0 else "darker"
            status = "matched" if bool(record["match_complete"]) else "outside tolerance"
            label = (
                f"{direction_label}; alpha={float(record['alpha']):g}; "
                f"change={float(record['skin_tone_change']):+.1f} ITA; {status}"
            )
            draw.multiline_text(
                (column * tile_size + 8, y + tile_size + 6),
                "\n".join(textwrap.wrap(label, width=38)),
                fill="black",
                font=footer_font,
                spacing=2,
            )
    return canvas


def main() -> None:
    args = parse_args()
    matched = pd.read_csv(args.analysis / "matched_conditions.csv")
    results = load_jsonl_results(args.run)
    if results.empty:
        raise SystemExit(f"No results.jsonl found below {args.run}")
    selected_seed, selected, audit = select_seed(matched)
    grid = make_grid(
        args.run,
        selected_seed,
        selected,
        tile_size=args.tile_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    audit["config_fingerprints"] = sorted(set(results["config_fingerprint"]))
    audit["selected_rows"] = selected[
        [
            "seed",
            "method",
            "direction",
            "target_change",
            "alpha",
            "skin_tone_change",
            "match_distance",
            "match_complete",
            "image_path",
        ]
    ].to_dict("records")
    audit_path = args.output.with_suffix(".selection.json")
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Selected seed {selected_seed}; wrote {args.output} and {audit_path}")


if __name__ == "__main__":
    main()
