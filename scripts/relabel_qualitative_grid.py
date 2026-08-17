#!/usr/bin/env python3
"""Rebuild labels on the retained qualitative grid without altering image tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from .make_qualitative_grid import (
        FOOTER_FONT_SIZE,
        FOOTER_HEIGHT,
        HEADER_FONT_SIZE,
        HEADER_HEIGHT,
        METHOD_LABELS,
        METHODS,
        _font,
    )
except ImportError:  # Direct execution: python scripts/relabel_qualitative_grid.py
    from make_qualitative_grid import (
        FOOTER_FONT_SIZE,
        FOOTER_HEIGHT,
        HEADER_FONT_SIZE,
        HEADER_HEIGHT,
        METHOD_LABELS,
        METHODS,
        _font,
    )

LEGACY_HEADER_HEIGHT = 30
LEGACY_FOOTER_HEIGHT = 50
PREVIOUS_HEADER_HEIGHT = 48
PREVIOUS_FOOTER_HEIGHT = 78
TILE_SIZE = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def relabel(grid: Image.Image, selection: dict) -> Image.Image:
    layouts = (
        (LEGACY_HEADER_HEIGHT, LEGACY_FOOTER_HEIGHT),
        (PREVIOUS_HEADER_HEIGHT, PREVIOUS_FOOTER_HEIGHT),
    )
    matching_layouts = [
        layout
        for layout in layouts
        if grid.size == (TILE_SIZE * 4, layout[0] + 2 * (TILE_SIZE + layout[1]))
    ]
    if not matching_layouts:
        expected_sizes = [
            (TILE_SIZE * 4, header + 2 * (TILE_SIZE + footer))
            for header, footer in layouts
        ]
        raise ValueError(f"Expected retained grid size in {expected_sizes}, got {grid.size}")
    source_header_height, source_footer_height = matching_layouts[0]

    rows = {
        (str(item["method"]), int(item["direction"])): item
        for item in selection["selected_rows"]
    }
    canvas = Image.new(
        "RGB",
        (TILE_SIZE * 4, HEADER_HEIGHT + 2 * (TILE_SIZE + FOOTER_HEIGHT)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = _font(HEADER_FONT_SIZE)
    footer_font = _font(FOOTER_FONT_SIZE)

    for column, label in enumerate(("Base", *(METHOD_LABELS[item] for item in METHODS))):
        draw.text((column * TILE_SIZE + 8, 7), label, fill="black", font=header_font)

    for row_index, direction in enumerate((-1, 1)):
        source_y = source_header_height + row_index * (TILE_SIZE + source_footer_height)
        target_y = HEADER_HEIGHT + row_index * (TILE_SIZE + FOOTER_HEIGHT)
        for column in range(4):
            tile = grid.crop(
                (
                    column * TILE_SIZE,
                    source_y,
                    (column + 1) * TILE_SIZE,
                    source_y + TILE_SIZE,
                )
            )
            canvas.paste(tile, (column * TILE_SIZE, target_y))

        draw.multiline_text(
            (8, target_y + TILE_SIZE + 6),
            "Base\nsame seed",
            fill="black",
            font=footer_font,
            spacing=4,
        )
        for column, method in enumerate(METHODS, start=1):
            record = rows[(method, direction)]
            direction_label = "lighter" if direction < 0 else "darker"
            status = "matched" if bool(record["match_complete"]) else "outside tolerance"
            label = (
                f"{direction_label}; alpha={float(record['alpha']):g}\n"
                f"delta ITA={float(record['skin_tone_change']):+.1f}; {status}"
            )
            draw.multiline_text(
                (column * TILE_SIZE + 8, target_y + TILE_SIZE + 6),
                label,
                fill="black",
                font=footer_font,
                spacing=4,
            )
    return canvas


def main() -> None:
    args = parse_args()
    with args.grid.open("rb") as source:
        grid = Image.open(source).convert("RGB")
        grid.load()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    output = relabel(grid, selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Wrote relabeled grid to {args.output}")


if __name__ == "__main__":
    main()
