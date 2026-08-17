"""
Counterfactual grid generation and visualization.

This module creates visual grids for presenting counterfactual results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image, ImageDraw, ImageFont


class CounterfactualGridGenerator:
    """
    Makes nice grids to show off the results.

    Features:
    - Grids with labels
    - Shows metrics right on the image
    - Side-by-side comparisons
    - Can even make interactive HTML sliders

    Example:
        >>> generator = CounterfactualGridGenerator()
        >>> grid = generator.generate_grid(
        ...     original, counterfactuals,
        ...     labels=["α=-2", "α=-1", "α=0", "α=+1", "α=+2"],
        ...     metrics=results
        ... )
        >>> grid.save("counterfactual_grid.png")
    """

    def __init__(
        self,
        font_size: int = 20,
        padding: int = 10,
        border_width: int = 2,
    ):
        """
        Initialize grid generator.

        Args:
            font_size: Font size for labels
            padding: Padding between images
            border_width: Border width around images
        """
        self.font_size = font_size
        self.padding = padding
        self.border_width = border_width

        # Try to load a nice font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except OSError:
            self.font = ImageFont.load_default()

    def generate_grid(
        self,
        original: Image.Image,
        counterfactuals: List[Image.Image],
        labels: Optional[List[str]] = None,
        metrics: Optional[List[Dict]] = None,
        title: Optional[str] = None,
        columns: Optional[int] = None,
    ) -> Image.Image:
        """
        Builds the main grid with the original image and the modified versions.

        Layout:
        ┌───────────┬───────────┬───────────┬───────────┐
        │ Original  │ CF α=-2   │ CF α=-1   │ CF α=0    │
        │ (label)   │ (label)   │ (label)   │ (label)   │
        │           │ Metrics   │ Metrics   │ Metrics   │
        └───────────┴───────────┴───────────┴───────────┘

        Args:
            original: The starting image
            counterfactuals: The generated images
            labels: Text to put under each image
            metrics: Optional dicts of scores to show
            title: Big title at the top
            columns: How many columns (defaults to auto)

        Returns:
            The final grid image
        """
        all_images = [original] + counterfactuals
        n_images = len(all_images)

        # Auto-compute columns
        if columns is None:
            columns = min(5, n_images)

        rows = (n_images + columns - 1) // columns

        # Resize all images to same size
        target_size = (512, 512)
        all_images = [img.resize(target_size, Image.LANCZOS) for img in all_images]

        # Create labels
        if labels is None:
            labels = ["Original"] + [f"CF {i+1}" for i in range(len(counterfactuals))]
        else:
            labels = ["Original"] + labels

        # Calculate grid size
        img_width, img_height = target_size
        label_height = 40
        metric_height = 80 if metrics else 0
        title_height = 60 if title else 0

        cell_width = img_width + 2 * self.border_width
        cell_height = img_height + label_height + metric_height + 2 * self.border_width

        grid_width = columns * cell_width + (columns + 1) * self.padding
        grid_height = (
            rows * cell_height + (rows + 1) * self.padding + title_height
        )

        # Create grid canvas
        grid = Image.new("RGB", (grid_width, grid_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(grid)

        # Add title
        if title:
            title_y = self.padding + title_height // 2
            bbox = draw.textbbox((0, 0), title, font=self.font)
            title_width = bbox[2] - bbox[0]
            title_x = (grid_width - title_width) // 2
            draw.text((title_x, title_y), title, fill=(0, 0, 0), font=self.font, anchor="mm")

        # Place images
        for idx, (img, label) in enumerate(zip(all_images, labels)):
            row = idx // columns
            col = idx % columns

            # Calculate position
            x = col * cell_width + (col + 1) * self.padding
            y = (
                row * cell_height
                + (row + 1) * self.padding
                + title_height
            )

            # Draw border
            border_color = (0, 100, 200) if idx == 0 else (200, 200, 200)
            draw.rectangle(
                [x, y, x + cell_width, y + cell_height],
                outline=border_color,
                width=self.border_width,
            )

            # Paste image
            img_x = x + self.border_width
            img_y = y + self.border_width
            grid.paste(img, (img_x, img_y))

            # Draw label
            label_y = img_y + img_height + 10
            bbox = draw.textbbox((0, 0), label, font=self.font)
            label_width = bbox[2] - bbox[0]
            label_x = img_x + (img_width - label_width) // 2
            draw.text((label_x, label_y), label, fill=(0, 0, 0), font=self.font)

            # Draw metrics (if counterfactual)
            if metrics and idx > 0:
                metric_dict = metrics[idx - 1]
                metric_y = label_y + 25

                # Format metrics
                metric_lines = []
                if "face_similarity" in metric_dict and metric_dict["face_similarity"] is not None:
                    metric_lines.append(f"Face Sim: {metric_dict['face_similarity']:.2f}")
                if "landmark_rmse" in metric_dict and metric_dict["landmark_rmse"] is not None:
                    metric_lines.append(f"RMSE: {metric_dict['landmark_rmse']:.1f}px")
                if "is_disentangled" in metric_dict:
                    status = "YES" if metric_dict["is_disentangled"] else "NO"
                    metric_lines.append(f"{status} Disentangled")

                # Draw metric lines
                for i, line in enumerate(metric_lines):
                    bbox = draw.textbbox((0, 0), line, font=self.font)
                    line_width = bbox[2] - bbox[0]
                    line_x = img_x + (img_width - line_width) // 2
                    draw.text(
                        (line_x, metric_y + i * 20),
                        line,
                        fill=(60, 60, 60),
                        font=self.font,
                    )

        return grid

    def generate_comparison(
        self,
        img1: Image.Image,
        img2: Image.Image,
        label1: str = "Original",
        label2: str = "Counterfactual",
        metrics: Optional[Dict] = None,
    ) -> Image.Image:
        """
        Create side-by-side comparison.

        Args:
            img1: First image
            img2: Second image
            label1: Label for first image
            label2: Label for second image
            metrics: Optional metrics dict

        Returns:
            Comparison image
        """
        # Resize to same size
        target_size = (512, 512)
        img1 = img1.resize(target_size, Image.LANCZOS)
        img2 = img2.resize(target_size, Image.LANCZOS)

        # Calculate canvas size
        label_height = 40
        metric_height = 100 if metrics else 0
        canvas_width = target_size[0] * 2 + self.padding * 3
        canvas_height = target_size[1] + label_height + metric_height + self.padding * 2

        # Create canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(canvas)

        # Paste images
        x1 = self.padding
        x2 = target_size[0] + self.padding * 2
        y = self.padding

        canvas.paste(img1, (x1, y))
        canvas.paste(img2, (x2, y))

        # Draw labels
        label_y = y + target_size[1] + 10

        bbox1 = draw.textbbox((0, 0), label1, font=self.font)
        label1_width = bbox1[2] - bbox1[0]
        label1_x = x1 + (target_size[0] - label1_width) // 2
        draw.text((label1_x, label_y), label1, fill=(0, 0, 0), font=self.font)

        bbox2 = draw.textbbox((0, 0), label2, font=self.font)
        label2_width = bbox2[2] - bbox2[0]
        label2_x = x2 + (target_size[0] - label2_width) // 2
        draw.text((label2_x, label_y), label2, fill=(0, 0, 0), font=self.font)

        # Draw metrics
        if metrics:
            metric_y = label_y + 30
            metric_x = canvas_width // 2

            metric_lines = []
            if "face_similarity" in metrics:
                metric_lines.append(f"Face Similarity: {metrics['face_similarity']:.3f}")
            if "landmark_rmse" in metrics:
                metric_lines.append(f"Landmark RMSE: {metrics['landmark_rmse']:.2f} px")
            if "background_ssim" in metrics:
                metric_lines.append(f"Background SSIM: {metrics['background_ssim']:.3f}")
            if "is_disentangled" in metrics:
                status = "Disentangled" if metrics["is_disentangled"] else "Entangled"
                metric_lines.append(status)

            for i, line in enumerate(metric_lines):
                bbox = draw.textbbox((0, 0), line, font=self.font)
                line_width = bbox[2] - bbox[0]
                draw.text(
                    (metric_x - line_width // 2, metric_y + i * 22),
                    line,
                    fill=(60, 60, 60),
                    font=self.font,
                )

        return canvas

    def generate_slider_html(
        self,
        images: List[Image.Image],
        alphas: List[float],
        output_dir: Union[str, Path],
        subject_id: str = "subject",
    ) -> str:
        """
        Create interactive HTML slider for continuous exploration.

        Args:
            images: List of counterfactual images
            alphas: Corresponding alpha values
            output_dir: Directory to save images
            subject_id: Subject identifier

        Returns:
            HTML string
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save images
        image_paths = []
        for i, (img, alpha) in enumerate(zip(images, alphas)):
            filename = f"{subject_id}_alpha_{alpha:.2f}.png"
            filepath = output_dir / filename
            img.save(filepath)
            image_paths.append(filename)

        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Counterfactual Explorer - {subject_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        #image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        #image {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        .controls {{
            margin: 20px 0;
        }}
        #slider {{
            width: 100%;
            margin: 10px 0;
        }}
        .info {{
            text-align: center;
            font-size: 18px;
            color: #666;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Counterfactual Explorer</h1>
        <p style="text-align: center; color: #888;">Subject: {subject_id}</p>

        <div id="image-container">
            <img id="image" src="{image_paths[0]}" alt="Counterfactual">
        </div>

        <div class="controls">
            <div class="info">
                <strong>Alpha:</strong> <span id="alpha-value">{alphas[0]:.2f}</span>
            </div>
            <input type="range" id="slider" min="0" max="{len(images)-1}" value="0" step="1">
            <div class="info" style="font-size: 14px; margin-top: 10px;">
                <span style="float: left;">Lighter ({alphas[0]:.1f})</span>
                <span>Original (0.0)</span>
                <span style="float: right;">Darker ({alphas[-1]:.1f})</span>
            </div>
        </div>

        <div style="clear: both; margin-top: 20px; text-align: center; color: #999; font-size: 12px;">
            Use the slider to explore different racial attributes while preserving identity.
        </div>
    </div>

    <script>
        const images = {image_paths};
        const alphas = {alphas};
        const slider = document.getElementById('slider');
        const imageElement = document.getElementById('image');
        const alphaValue = document.getElementById('alpha-value');

        slider.addEventListener('input', function() {{
            const index = parseInt(this.value);
            imageElement.src = images[index];
            alphaValue.textContent = alphas[index].toFixed(2);
        }});
    </script>
</body>
</html>
"""

        # Save HTML
        html_path = output_dir / f"{subject_id}_slider.html"
        with open(html_path, "w") as f:
            f.write(html)

        print(f"Created interactive slider: {html_path}")

        return html

    def create_annotated_image(
        self,
        img: Image.Image,
        annotations: List[str],
        title: Optional[str] = None,
    ) -> Image.Image:
        """
        Add text annotations to image.

        Args:
            img: Input image
            annotations: List of text annotations
            title: Optional title

        Returns:
            Annotated image
        """
        # Calculate canvas size
        annotation_height = len(annotations) * 25 + 20
        title_height = 40 if title else 0
        canvas_width = img.width
        canvas_height = img.height + annotation_height + title_height

        # Create canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Add title
        if title:
            bbox = draw.textbbox((0, 0), title, font=self.font)
            title_width = bbox[2] - bbox[0]
            title_x = (canvas_width - title_width) // 2
            draw.text((title_x, 10), title, fill=(0, 0, 0), font=self.font)

        # Paste image
        img_y = title_height
        canvas.paste(img, (0, img_y))

        # Add annotations
        annotation_y = img_y + img.height + 10
        for i, text in enumerate(annotations):
            draw.text(
                (10, annotation_y + i * 25),
                text,
                fill=(60, 60, 60),
                font=self.font,
            )

        return canvas
