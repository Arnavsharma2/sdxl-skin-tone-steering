"""
Stable Diffusion wrapper with latent space access.

This module provides a clean interface for:
- Encoding images to latent space
- Decoding latents to images
- Generating images from prompts
- Caching latent codes
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)
from PIL import Image


class StableDiffusionWrapper:
    """
    Wrapper for Stable Diffusion XL that gives us access to the latent space.

    What it does:
    - Encodes images into latents
    - Decodes latents back into images
    - Generates images from prompts (while letting us mess with the latents)
    - Handles memory efficiently

    Example:
        >>> sd = StableDiffusionWrapper("cuda", dtype=torch.float16)
        >>> image = Image.open("portrait.jpg")
        >>> latent = sd.encode_image(image)
        >>> reconstructed = sd.decode_latent(latent)
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        revision: Optional[str] = None,
        enable_xformers: bool = True,
        enable_cpu_offload: bool = False,
    ):
        """
        Sets up the Stable Diffusion pipeline.

        Args:
            device: Where to run it ('cuda' or 'cpu')
            dtype: Precision (torch.float16 or torch.float32)
            model_id: Which HuggingFace model to use
            revision: Exact model revision requested from Hugging Face
            enable_xformers: Whether to use memory-efficient attention (saves VRAM)
            enable_cpu_offload: Whether to offload models to CPU when idle
        """
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.requested_revision = revision

        print(f"Loading Stable Diffusion XL from {model_id}...")

        # Load VAE separately for encoding/decoding
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=dtype,
            revision=revision,
        )
        self.vae.to(device)
        self.vae.eval()

        # Load full pipeline for generation
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            vae=self.vae,
            revision=revision,
        )

        # DPM++ 2M Karras — significantly better quality per step than DDIM.
        # Achieves DDIM-50-step quality in ~20 steps.
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

        # Memory optimizations — always enable VAE slicing (zero quality cost)
        self.pipe.enable_vae_slicing()

        if enable_xformers and device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory-efficient attention")
            except Exception as e:
                print(f"WARNING: Could not enable xformers: {e}")
                self.pipe.enable_attention_slicing()
        else:
            # Attention slicing helps on MPS and CPU
            self.pipe.enable_attention_slicing()

        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload")
        else:
            self.pipe.to(device)

        self.resolved_revision = (
            getattr(self.pipe.config, "_commit_hash", None)
            or getattr(self.vae.config, "_commit_hash", None)
            or revision
        )

        print("Stable Diffusion loaded successfully.")

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        size: Tuple[int, int] = (512, 512),
    ) -> torch.Tensor:
        """
        Prepares an image for the encoder.

        Args:
            image: PIL Image or numpy array
            size: Target size (width, height)

        Returns:
            Tensor ready for the model (1, 3, H, W) in range [-1, 1]
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize
        image = image.resize(size, Image.LANCZOS)

        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Normalize to [-1, 1]
        image = (image * 2.0) - 1.0

        return image.to(self.device).to(self.dtype)

    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL Image.

        Args:
            tensor: Image tensor (1, 3, H, W) in range [-1, 1]

        Returns:
            PIL Image
        """
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor / 2.0 + 0.5).clamp(0, 1)

        # Convert to numpy
        image = tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)

        return Image.fromarray(image)

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray],
        size: Tuple[int, int] = (512, 512),
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: Input image
            size: Size to resize to before encoding.
            sample_posterior: Sample the VAE posterior. The deterministic
                posterior mode is the default for reproducible experiments.
            generator: Optional generator used for posterior sampling.

        Returns:
            Latent code tensor (1, 4, H//8, W//8)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image, size)

        # Encode with VAE
        latent_dist = self.vae.encode(image_tensor).latent_dist
        latent = (
            latent_dist.sample(generator=generator)
            if sample_posterior
            else latent_dist.mode()
        )

        # Scale (SDXL uses scaling factor)
        latent = latent * self.vae.config.scaling_factor

        return latent

    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
        return_tensor: bool = False,
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Decode latent code to image.

        Args:
            latent: Latent code (1, 4, H//8, W//8)
            return_tensor: Return tensor instead of PIL Image

        Returns:
            Decoded image
        """
        # Unscale
        latent = latent / self.vae.config.scaling_factor

        # Decode
        image = self.vae.decode(latent).sample

        if return_tensor:
            return image
        else:
            return self.postprocess_image(image)

    @torch.no_grad()
    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        return_latent: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Image.Image, Optional[torch.Tensor]]:
        """
        Generate image from text prompt.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Generated image height
            width: Generated image width
            seed: Random seed for reproducibility
            return_latent: Also return final latent code
            **kwargs: Additional arguments for pipeline

        Returns:
            (generated_image, latent_code) if return_latent=True
            (generated_image, None) otherwise
        """
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        size_kwargs = {
            key: value
            for key, value in {"height": height, "width": width}.items()
            if value is not None
        }

        # Generate
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil" if not return_latent else "latent",
            **size_kwargs,
            **kwargs,
        )

        if return_latent:
            # Safely extract latent from output
            if hasattr(output, 'images'):
                latent = output.images
            elif isinstance(output, torch.Tensor):
                latent = output
            elif hasattr(output, '__getitem__'):
                latent = output[0]
            else:
                raise ValueError(f"Unexpected output type from pipeline: {type(output)}")

            # Decode to get image
            image = self.decode_latent(latent)
            return image, latent
        else:
            image = output.images[0]
            return image, None

    def generate_steered(
        self,
        prompt: str,
        race_vector: torch.Tensor,
        alpha: float,
        seed: int,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Generate an image steered by a skin-tone direction during denoising.

        Instead of adding the direction to the final latent (which pushes it
        out-of-distribution), this method injects the vector at each denoising
        step. The diffusion process then naturally produces a coherent image in
        the steered direction.

        Args:
            prompt: Text prompt (same as base image)
            race_vector: Skin-tone direction tensor. The parameter retains its
                original name for API compatibility.
            alpha: Steering magnitude. Positive = darker, Negative = lighter.
                   Values in [-5, 5] are a reasonable starting range.
            seed: RNG seed — use the same seed as the base image for consistency
            negative_prompt: Negative guidance prompt
            num_inference_steps: Denoising steps (25 with DPM++ 2M ≈ DDIM 50)
            guidance_scale: CFG scale
            height: Generated image height
            width: Generated image width

        Returns:
            (steered_image, final_latent)
        """
        import torch.nn.functional as F

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Per-step injection amount. We spread the total alpha * direction
        # uniformly across all steps. At early (noisy) steps the injected delta
        # is tiny relative to the noise magnitude so it has little effect;
        # at late (clean) steps the latent scale matches the direction scale
        # and the injection meaningfully steers the color/tone. This naturally
        # concentrates the effect where it matters.
        alpha_per_step = alpha / num_inference_steps

        def steering_callback(_pipe, _step_idx, _timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            rv = race_vector.to(dtype=latents.dtype, device=latents.device)

            # Resize the direction if its spatial dims do not match the latent.
            if rv.shape != latents.shape:
                # Ensure 4-D for interpolate
                if rv.dim() == 3:
                    rv = rv.unsqueeze(0)
                if rv.shape[2:] != latents.shape[2:]:
                    rv = F.interpolate(
                        rv,
                        size=latents.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                # Drop batch dim if latents also has no batch dim
                if latents.dim() == 3:
                    rv = rv.squeeze(0)

            latents = latents + alpha_per_step * rv
            return {"latents": latents}

        size_kwargs = {
            key: value
            for key, value in {"height": height, "width": width}.items()
            if value is not None
        }
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            callback_on_step_end=steering_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            **size_kwargs,
        )

        latent = output.images
        image = self.decode_latent(latent)
        return image, latent

    def save_latent(self, latent: torch.Tensor, path: Union[str, Path]):
        """Save latent code to disk."""
        torch.save(latent.cpu(), path)

    def load_latent(self, path: Union[str, Path]) -> torch.Tensor:
        """Load latent code from disk."""
        latent = torch.load(path, map_location=self.device, weights_only=True)
        return latent.to(self.dtype)

    def reconstruct_image(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Encode and decode image (reconstruction test).

        Args:
            image: Input image

        Returns:
            (reconstructed_image, latent_code)
        """
        latent = self.encode_image(image)
        reconstructed = self.decode_latent(latent)
        return reconstructed, latent

    def interpolate_latents(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        num_steps: int = 5,
    ) -> List[Image.Image]:
        """
        Linear interpolation between two latent codes.

        Args:
            latent1: First latent code
            latent2: Second latent code
            num_steps: Number of interpolation steps

        Returns:
            List of interpolated images
        """
        alphas = torch.linspace(0, 1, num_steps)
        images = []

        for alpha in alphas:
            interpolated = (1 - alpha) * latent1 + alpha * latent2
            image = self.decode_latent(interpolated)
            images.append(image)

        return images


class LatentCache:
    """
    Simple cache for latent codes to avoid re-encoding.

    Example:
        >>> cache = LatentCache("data/embeddings")
        >>> cache.save("image_001", latent)
        >>> latent = cache.load("image_001")
    """

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cached latents
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, image_id: str) -> Path:
        """Get path for cached latent."""
        return self.cache_dir / f"{image_id}.pt"

    def exists(self, image_id: str) -> bool:
        """Check if latent is cached."""
        return self.get_path(image_id).exists()

    def save(self, image_id: str, latent: torch.Tensor):
        """Save latent to cache."""
        path = self.get_path(image_id)
        torch.save(latent.cpu(), path)

    def load(self, image_id: str, device: str = "cuda") -> torch.Tensor:
        """Load latent from cache."""
        path = self.get_path(image_id)
        return torch.load(path, map_location=device, weights_only=True)

    def clear(self):
        """Clear all cached latents."""
        for file in self.cache_dir.glob("*.pt"):
            file.unlink()
