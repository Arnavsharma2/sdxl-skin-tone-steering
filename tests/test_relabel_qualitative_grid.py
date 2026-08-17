from PIL import Image

from scripts.make_qualitative_grid import FOOTER_HEIGHT, HEADER_HEIGHT
from scripts.relabel_qualitative_grid import TILE_SIZE, relabel


def test_relabel_preserves_every_image_tile() -> None:
    legacy = Image.new("RGB", (TILE_SIZE * 4, 770), "white")
    colors = []
    for row in range(2):
        row_colors = []
        for column in range(4):
            color = (25 + row * 80, 35 + column * 40, 45 + row * 30 + column * 10)
            row_colors.append(color)
            legacy.paste(
                color,
                (
                    column * TILE_SIZE,
                    30 + row * 370,
                    (column + 1) * TILE_SIZE,
                    30 + row * 370 + TILE_SIZE,
                ),
            )
        colors.append(row_colors)

    selected_rows = []
    for direction in (-1, 1):
        for method in ("prompt_only", "stepwise_unmasked", "stepwise_masked"):
            selected_rows.append(
                {
                    "method": method,
                    "direction": direction,
                    "alpha": 0.5,
                    "skin_tone_change": 3.1 * direction,
                    "match_complete": True,
                }
            )

    output = relabel(legacy, {"selected_rows": selected_rows})

    assert output.size == (TILE_SIZE * 4, HEADER_HEIGHT + 2 * (TILE_SIZE + FOOTER_HEIGHT))
    for row in range(2):
        y = HEADER_HEIGHT + row * (TILE_SIZE + FOOTER_HEIGHT) + TILE_SIZE // 2
        for column in range(4):
            x = column * TILE_SIZE + TILE_SIZE // 2
            assert output.getpixel((x, y)) == colors[row][column]
