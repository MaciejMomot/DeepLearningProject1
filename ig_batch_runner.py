
from __future__ import annotations

import csv
import math
import random
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from captum.attr import IntegratedGradients


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
DEFAULT_MEAN = [0.47889522, 0.47227842, 0.43047404]
DEFAULT_STD = [0.24205776, 0.23828046, 0.25874835]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))
    return name.strip("._") or "item"


def build_transform(mean: Sequence[float], std: Sequence[float], image_size: Optional[Tuple[int, int]] = None):
    ops: List[Any] = []
    if image_size is not None:
        ops.append(transforms.Resize(image_size))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(ops)


def denormalize_tensor(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    img = (x * std_t + mean_t).clamp(0, 1)
    return img[0].permute(1, 2, 0).detach().cpu().numpy()


# -----------------------------
# Images
# -----------------------------
def list_images(image_dir: str | Path, img_exts: Optional[set] = None) -> List[Path]:
    img_exts = img_exts or IMG_EXTS
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in img_exts)
    if not files:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return files


def resolve_label_from_path(image_path: str | Path, class_names: Sequence[str]) -> Optional[str]:
    image_path = Path(image_path)
    parent_name = image_path.parent.name
    return parent_name if parent_name in class_names else None


def select_images(
    image_dir: Optional[str | Path] = None,
    image_paths: Optional[Sequence[str | Path]] = None,
    image_indices: Optional[Sequence[int]] = None,
    max_images: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Path]:
    if image_paths:
        selected = [Path(p) for p in image_paths]
        missing = [str(p) for p in selected if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing image files: {missing}")
    else:
        if image_dir is None:
            raise ValueError("Provide image_dir or image_paths")
        files = list_images(image_dir)
        if image_indices is not None:
            selected = []
            for idx in image_indices:
                if idx < 0 or idx >= len(files):
                    raise IndexError(f"image_index={idx}, number of images={len(files)}")
                selected.append(files[idx])
        else:
            selected = files

    if shuffle:
        rng = random.Random(seed)
        selected = selected[:]
        rng.shuffle(selected)

    if max_images is not None:
        selected = selected[:max_images]

    if not selected:
        raise ValueError("No images selected")
    return selected


def load_image_tensor(
    image_path: str | Path,
    transform,
    device: torch.device = DEVICE,
) -> Tuple[Image.Image, torch.Tensor]:
    pil_img = Image.open(image_path).convert("RGB")
    x = transform(pil_img).unsqueeze(0).to(device)
    return pil_img, x


# -----------------------------
# Model registry
# -----------------------------
def get_mobilenetv2(num_classes: int, dropout_rate: float = 0.0, first_stride: int = 1) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=first_stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
    )
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def get_resnet18(num_classes: int, cifar_style: bool = True) -> nn.Module:
    model = models.resnet18(weights=None)
    if cifar_style:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "mobilenet_v2": get_mobilenetv2,
    "resnet18": get_resnet18,
    "efficientnet_b0": get_efficientnet_b0,
}


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            tensor_values = sum(torch.is_tensor(v) for v in checkpoint.values())
            if tensor_values > 0:
                state_dict = checkpoint
            else:
                raise ValueError("Could not extract a valid state_dict from checkpoint")
    else:
        raise TypeError("Checkpoint must be a dict-like object")

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def load_checkpoint_flexible(model: nn.Module, weights_path: str | Path, device: torch.device = DEVICE) -> nn.Module:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        raise RuntimeError(
            "Checkpoint loaded only partially.\n"
            f"Missing keys: {missing[:10]}\n"
            f"Unexpected keys: {unexpected[:10]}\n"
            f"Original error: {exc}"
        )

    model.to(device)
    model.eval()
    return model


def build_model_from_cfg(model_cfg: Dict[str, Any], device: torch.device = DEVICE) -> nn.Module:
    builder_name = model_cfg["builder"]
    if builder_name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown builder: {builder_name}. Available: {list(MODEL_BUILDERS)}")

    builder = MODEL_BUILDERS[builder_name]
    builder_kwargs = dict(model_cfg.get("builder_kwargs", {}))
    num_classes = int(model_cfg.get("num_classes", len(DEFAULT_CLASS_NAMES)))
    model = builder(num_classes=num_classes, **builder_kwargs)
    model = load_checkpoint_flexible(model, model_cfg["weights_path"], device=device)
    return model


# -----------------------------
# Prediction and targets
# -----------------------------
@torch.no_grad()
def predict_with_probs(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1)
    pred_conf = probs.gather(1, pred_class.unsqueeze(1)).squeeze(1)
    return {
        "logits": logits,
        "probs": probs,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
    }


def resolve_targets(
    probs: torch.Tensor,
    label_name: Optional[str],
    class_names: Sequence[str],
    target_mode: str = "pred",
    target_class: Optional[int] = None,
    target_classes: Optional[Sequence[int]] = None,
    topk: int = 5,
    class_order: str = "index",
) -> List[Tuple[int, str]]:
    probs_1d = probs[0]
    num_classes = probs_1d.numel()
    pred_class = int(probs_1d.argmax().item())

    if target_mode == "pred":
        return [(pred_class, f"pred: {class_names[pred_class]}")]

    if target_mode == "label":
        if label_name is None:
            raise ValueError("Could not infer true label from image folder name")
        cls = class_names.index(label_name)
        return [(cls, f"label: {label_name}")]

    if target_mode == "class":
        if target_class is None:
            raise ValueError("Set target_class when target_mode='class'")
        cls = int(target_class)
        return [(cls, f"class: {class_names[cls]}")]

    if target_mode == "class_list":
        if not target_classes:
            raise ValueError("Set target_classes when target_mode='class_list'")
        return [(int(cls), f"class_list: {class_names[int(cls)]}") for cls in target_classes]

    if target_mode == "topk":
        topk = max(1, min(int(topk), num_classes))
        indices = probs_1d.topk(topk).indices.detach().cpu().tolist()
        return [(int(cls), f"topk: {class_names[int(cls)]}") for cls in indices]

    if target_mode == "all_classes":
        indices = list(range(num_classes))
        if class_order == "prob_desc":
            indices = sorted(indices, key=lambda idx: float(probs_1d[idx].item()), reverse=True)
        return [(int(cls), f"all_classes: {class_names[int(cls)]}") for cls in indices]

    raise ValueError(
        f"Unknown target_mode: {target_mode}. "
        "Available: pred, label, class, class_list, topk, all_classes"
    )


# -----------------------------
# Integrated Gradients
# -----------------------------
def make_black_baseline_normalized(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    baseline = (0.0 - mean_t) / std_t
    return baseline.expand_as(x)


def gaussian_kernel2d(kernel_size: int = 31, sigma: float = 5.0, device: torch.device = DEVICE, dtype=torch.float32):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d.view(1, 1, kernel_size, kernel_size)


def blur_map(x: torch.Tensor, kernel_size: int = 31, sigma: float = 5.0) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if x.dim() == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x4 = x.unsqueeze(1)
    else:
        raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(x.shape)}")
    kernel = gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma, device=x4.device, dtype=x4.dtype)
    pad = kernel.shape[-1] // 2
    out = F.conv2d(x4, kernel, padding=pad)
    if x.dim() == 2:
        return out[0, 0]
    return out[:, 0]


def robust_normalize_map(
    x: torch.Tensor,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    flat = x.flatten()
    lo = torch.quantile(flat, low_percentile / 100.0)
    hi = torch.quantile(flat, high_percentile / 100.0)
    denom = torch.clamp(hi - lo, min=eps)
    x = (x - lo) / denom
    return x.clamp(0, 1)


def build_score_forward(
    model: nn.Module,
    score_mode: str = "logit",
):
    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        logits = model(x)
        if score_mode == "logit":
            return logits
        if score_mode == "prob":
            return F.softmax(logits, dim=1)
        if score_mode == "logprob":
            return F.log_softmax(logits, dim=1)
        if score_mode == "margin":
            num_classes = logits.shape[1]
            others_mean = (logits.sum(dim=1, keepdim=True) - logits) / max(1, num_classes - 1)
            return logits - others_mean
        raise ValueError(f"Unknown score_mode: {score_mode}. Available: logit, prob, logprob, margin")
    return forward_fn


def integrated_gradients_for_targets(
    model: nn.Module,
    x: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    targets: Sequence[int],
    n_steps: int = 50,
    internal_batch_size: int = 16,
    score_mode: str = "logit",
    device: torch.device = DEVICE,
) -> List[Dict[str, Any]]:
    model.eval()
    x = x.to(device)

    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)

    baseline = make_black_baseline_normalized(x, mean=mean, std=std)
    ig = IntegratedGradients(build_score_forward(model, score_mode=score_mode))

    outputs: List[Dict[str, Any]] = []
    for target in targets:
        attributions, delta = ig.attribute(
            inputs=x,
            baselines=baseline,
            target=int(target),
            n_steps=n_steps,
            method="gausslegendre",
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
        outputs.append({
            "target_class": int(target),
            "attributions": attributions[0].detach(),
            "delta": float(delta[0].item()) if torch.is_tensor(delta) else float(delta),
        })
    return outputs


# -----------------------------
# Visual maps
# -----------------------------
def build_signed_map(
    attributions_single: torch.Tensor,
    clip_percentile: float = 99.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    signed_map = attributions_single.mean(dim=0)
    vmax_signed = torch.quantile(signed_map.abs().flatten(), clip_percentile / 100.0)
    vmax_signed = torch.clamp(vmax_signed, min=eps)
    signed_map = torch.clamp(signed_map, min=-vmax_signed, max=vmax_signed)
    return signed_map / vmax_signed


def build_heatmap_base(
    attributions_single: torch.Tensor,
    mode: str = "mean_rgb",
    blur_kernel: int = 1,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    if attributions_single.dim() != 3:
        raise ValueError(f"Expected [3, H, W], got {tuple(attributions_single.shape)}")

    if mode == "mean_rgb":
        base = attributions_single.mean(dim=0)
    elif mode == "positive":
        base = attributions_single.clamp(min=0).sum(dim=0)
    elif mode == "abs":
        base = attributions_single.abs().sum(dim=0)
    elif mode == "signed_positive_mean":
        base = attributions_single.mean(dim=0).clamp(min=0)
    else:
        raise ValueError(f"Unknown heatmap mode: {mode}")

    if blur_kernel > 1:
        base = blur_map(base, kernel_size=blur_kernel, sigma=blur_sigma)

    return base


def normalize_heatmap_independent(
    base_map: torch.Tensor,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    lo = base_map.min()
    hi = base_map.max()
    heat = (base_map - lo) / torch.clamp(hi - lo, min=eps)
    if gamma != 1.0:
        heat = heat.clamp(0, 1) ** gamma
    return heat.clamp(0, 1)


def normalize_heatmaps_shared(
    base_maps: Sequence[torch.Tensor],
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
    gamma: float = 1.0,
    mass_scale_power: float = 1.0,
    suppress_below_mass_ratio: float = 0.0,
    eps: float = 1e-8,
) -> List[Dict[str, Any]]:
    if not base_maps:
        return []

    flat_all = torch.cat([bm.flatten() for bm in base_maps])
    global_hi = torch.quantile(flat_all, high_percentile / 100.0)
    global_hi = torch.clamp(global_hi, min=eps)

    masses = [float(bm.sum().item()) for bm in base_maps]
    max_mass = max(max(masses), eps)

    outputs: List[Dict[str, Any]] = []
    for base_map, mass in zip(base_maps, masses):
        lo = torch.quantile(base_map.flatten(), low_percentile / 100.0) if low_percentile > 0 else torch.tensor(0.0, device=base_map.device, dtype=base_map.dtype)
        denom = torch.clamp(global_hi - lo, min=eps)
        heat = ((base_map - lo) / denom).clamp(0, 1)

        relative_mass = float(mass / max_mass)
        if relative_mass <= suppress_below_mass_ratio:
            heat = torch.zeros_like(heat)
        else:
            strength = relative_mass ** mass_scale_power
            heat = heat * strength

        if gamma != 1.0:
            heat = heat.clamp(0, 1) ** gamma

        outputs.append({
            "heatmap": heat.clamp(0, 1),
            "raw_mass": mass,
            "relative_mass": relative_mass,
        })
    return outputs


def build_overlay(
    img_np: np.ndarray,
    heat_np: np.ndarray,
    cmap: str = "turbo",
    max_alpha: float = 0.80,
    min_alpha: float = 0.05,
    threshold: float = 0.08,
) -> np.ndarray:
    heat_np = np.clip(heat_np, 0, 1)
    colored = plt.get_cmap(cmap)(heat_np)[..., :3]

    alpha = np.clip((heat_np - threshold) / max(1e-8, (1.0 - threshold)), 0, 1)
    alpha = min_alpha + (max_alpha - min_alpha) * alpha
    alpha = np.where(heat_np <= threshold, 0.0, alpha)

    blended = img_np * (1.0 - alpha[..., None]) + colored * alpha[..., None]
    return np.clip(blended, 0, 1)


def build_signed_overlay(
    img_np: np.ndarray,
    signed_np: np.ndarray,
    cmap: str = "coolwarm",
    max_alpha: float = 0.80,
    min_alpha: float = 0.05,
    threshold: float = 0.10,
) -> np.ndarray:
    signed_np = np.clip(signed_np, -1, 1)
    heat_01 = 0.5 + 0.5 * signed_np
    colored = plt.get_cmap(cmap)(heat_01)[..., :3]

    strength = np.abs(signed_np)
    alpha = np.clip((strength - threshold) / max(1e-8, (1.0 - threshold)), 0, 1)
    alpha = min_alpha + (max_alpha - min_alpha) * alpha
    alpha = np.where(strength <= threshold, 0.0, alpha)

    blended = img_np * (1.0 - alpha[..., None]) + colored * alpha[..., None]
    return np.clip(blended, 0, 1)


def add_overlay_colorbar(fig, axes, cmap: str = "inferno", label: str = "Normalized 2D IG map [0, 1]", pad: float = 0.03, width: float = 0.018) -> None:
    import matplotlib as mpl

    axes = list(axes)
    if len(axes) == 0:
        return

    x0 = min(ax.get_position().x0 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)

    cax = fig.add_axes([min(x1 + pad, 0.965 - width), y0, width, max(0.12, y1 - y0)])
    sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label, fontsize=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])


def add_method_footer(
    fig,
    *,
    score_mode: str = "logit",
    n_steps: int = 50,
    heatmap_mode: str = "mean_rgb",
    legend_title: str = "How to read this figure",
    y: float = 0.02,
) -> None:
    reduce_txt = "mean over RGB channels" if heatmap_mode == "mean_rgb" else heatmap_mode
    body = (
        f"{legend_title}: IG baseline = black image; target = class {score_mode}; integration steps = {n_steps}; "
        f"3×H×W attributions → 2D map via {reduce_txt}; each class map is normalized independently to [0, 1]. "
        f"Tile text: p = softmax probability, signed mass = relative signed attribution mass. "
        f"Overlay colors: dark = lower normalized value, bright = higher normalized value within the same class tile. "
        f"Colors are not signed here, so compare location patterns across tiles; use signed mass for overall signed support."
    )
    wrapped = textwrap.fill(body, width=145)
    fig.text(
        0.03,
        y,
        wrapped,
        ha="left",
        va="bottom",
        fontsize=9.0,
        family="sans-serif",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.96},
    )

def build_explanation_maps(
    attributions_single: torch.Tensor,
    signed_clip_percentile: float = 99.0,
    heatmap_mode: str = "mean_rgb",
    blur_kernel: int = 1,
    blur_sigma: float = 0.0,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
    gamma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    signed_map = build_signed_map(attributions_single, clip_percentile=signed_clip_percentile)
    base_map = build_heatmap_base(
        attributions_single,
        mode=heatmap_mode,
        blur_kernel=blur_kernel,
        blur_sigma=blur_sigma,
    )
    cam_like_map = normalize_heatmap_independent(
        base_map=base_map,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        gamma=gamma,
    )
    return {
        "signed_map": signed_map,
        "base_map": base_map,
        "cam_like_map": cam_like_map,
        "raw_mass": float(base_map.sum().item()),
    }


# -----------------------------
# Plotting
# -----------------------------
def _short_tile_title(
    class_name: str,
    p: float,
    signed_mass: Optional[float] = None,
    is_pred: bool = False,
    is_label: bool = False,
    width: int = 18,
) -> str:
    tag_bits = []
    if is_pred:
        tag_bits.append("pred")
    if is_label:
        tag_bits.append("label")
    tag = f" [{' | '.join(tag_bits)}]" if tag_bits else ""
    line1 = textwrap.fill(f"{class_name}{tag}", width=width, break_long_words=False)
    line2 = f"p = {p:.3f}"
    line3 = f"signed mass = {signed_mass:.2f}" if signed_mass is not None else None
    return "\n".join([x for x in (line1, line2, line3) if x])


def _reserve_figure_margins(
    fig,
    *,
    top: float,
    bottom: float,
    left: float = 0.03,
    right: float = 0.93,
    wspace: float = 0.18,
    hspace: float = 0.30,
):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)


def plot_single_explanation(
    x: torch.Tensor,
    class_names: Sequence[str],
    pred_class: int,
    pred_conf: float,
    target_class: int,
    target_desc: str,
    target_prob: float,
    mean: Sequence[float],
    std: Sequence[float],
    signed_map: torch.Tensor,
    heatmap: torch.Tensor,
    delta: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 5.8),
    signed_cmap: str = "seismic",
    overlay_cmap: str = "inferno",
    overlay_mode: str = "unsigned",
    overlay_alpha: float = 0.80,
    min_overlay_alpha: float = 0.05,
    threshold: float = 0.08,
    colorbar_pad: float = 0.03,
    add_reading_legend: bool = True,
    score_mode: str = "logit",
    n_steps: int = 50,
    heatmap_mode: str = "mean_rgb",
    show: bool = False,
    save_path: Optional[str | Path] = None,
    dpi: int = 180,
):
    img_np = denormalize_tensor(x, mean=mean, std=std)
    signed_np = signed_map.detach().cpu().numpy()
    heat_np = heatmap.detach().cpu().numpy()
    overlay_np = build_overlay(
        img_np=img_np,
        heat_np=heat_np,
        cmap=overlay_cmap,
        max_alpha=overlay_alpha,
        min_alpha=min_overlay_alpha,
        threshold=threshold,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _reserve_figure_margins(fig, top=0.84, bottom=0.14, left=0.03, right=0.96, wspace=0.22)

    axes[0].imshow(img_np)
    axes[0].set_title(
        f"Input\npred = {class_names[pred_class]} ({pred_conf:.3f})",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[0].axis("off")

    im1 = axes[1].imshow(signed_np, cmap=signed_cmap, vmin=-1, vmax=1, interpolation="bilinear")
    axes[1].set_title("2D IG map\nmean over RGB channels", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_ticks([-1, -0.5, 0, 0.5, 1])

    axes[2].imshow(overlay_np)
    axes[2].set_title(
        f"Normalized IG overlay\nclass = {class_names[target_class]} | p = {target_prob:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    axes[2].axis("off")

    title = f"Integrated Gradients | target = {target_desc}"
    if delta is not None:
        title += f" | delta = {delta:.6f}"
    fig.suptitle(title, y=0.96, fontsize=14, fontweight="bold")

    if add_reading_legend:
        add_method_footer(
            fig,
            score_mode=score_mode,
            n_steps=n_steps,
            heatmap_mode=heatmap_mode,
            legend_title="How to read this figure",
            y=0.03,
        )

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_multi_class_grid(
    x: torch.Tensor,
    class_names: Sequence[str],
    probs: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    explanations: Sequence[Dict[str, Any]],
    n_cols: int = 4,
    overlay_cmap: str = "inferno",
    overlay_mode: str = "unsigned",
    overlay_alpha: float = 0.80,
    min_overlay_alpha: float = 0.05,
    threshold: float = 0.08,
    colorbar_pad: float = 0.03,
    figsize_per_cell: Tuple[float, float] = (4.0, 4.0),
    sort_by: str = "index",
    title: Optional[str] = None,
    add_reading_legend: bool = False,
    legend_title: str = "How to read this grid",
    score_mode: str = "logit",
    n_steps: int = 50,
    heatmap_mode: str = "mean_rgb",
    show: bool = False,
    save_path: Optional[str | Path] = None,
    dpi: int = 180,
):
    img_np = denormalize_tensor(x, mean=mean, std=std)
    probs_np = probs[0].detach().cpu().numpy()

    items = list(explanations)
    if sort_by == "prob_desc":
        items = sorted(items, key=lambda d: float(d["target_prob"]), reverse=True)
    elif sort_by != "index":
        raise ValueError(f"Unknown sort_by: {sort_by}")

    n_panels = len(items) + 1
    n_cols = max(1, min(int(n_cols), n_panels))
    n_rows = math.ceil(n_panels / n_cols)

    fig_w = figsize_per_cell[0] * n_cols + 0.8
    fig_h = figsize_per_cell[1] * n_rows + (1.0 if add_reading_legend else 0.0) + 0.35
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    top_margin = 0.88 if title else 0.94
    bottom_margin = 0.14 if add_reading_legend else 0.06
    _reserve_figure_margins(fig, top=top_margin, bottom=bottom_margin, left=0.03, right=0.87, wspace=0.16, hspace=0.34)

    pred_idx = int(probs_np.argmax())

    top3 = probs_np.argsort()[::-1][:3]
    top3_lines = [f"{class_names[i]}: {probs_np[i]:.3f}" for i in top3]
    input_title = "Input\n" + f"Top-1: {class_names[pred_idx]} ({probs_np[pred_idx]:.3f})\n" + "\n".join(top3_lines)
    axes[0].imshow(img_np)
    axes[0].set_title(input_title, fontsize=10.5, fontweight="bold", pad=7)
    axes[0].axis("off")

    for ax, item in zip(axes[1:], items):
        heat_np = item["heatmap"].detach().cpu().numpy()
        signed_np = item["signed_map"].detach().cpu().numpy()
        if overlay_mode == "signed_centered":
            overlay_np = build_signed_overlay(
                img_np=img_np,
                signed_np=signed_np,
                cmap=overlay_cmap,
                max_alpha=overlay_alpha,
                min_alpha=min_overlay_alpha,
                threshold=threshold,
            )
        else:
            overlay_np = build_overlay(
                img_np=img_np,
                heat_np=heat_np,
                cmap=overlay_cmap,
                max_alpha=overlay_alpha,
                min_alpha=min_overlay_alpha,
                threshold=threshold,
            )
        cls = int(item["target_class"])
        p = float(item["target_prob"])
        rel_mass = item.get("relative_mass")
        ax.imshow(overlay_np)
        ax.set_title(
            _short_tile_title(
                class_name=class_names[cls],
                p=p,
                signed_mass=rel_mass,
                is_pred=(cls == pred_idx),
                is_label=(item.get("label_name") is not None and class_names[cls] == item["label_name"]),
                width=18,
            ),
            fontsize=10.2,
            fontweight="bold",
            pad=6,
        )
        ax.axis("off")

    for ax in axes[n_panels:]:
        ax.axis("off")

    cbar_label = "Signed 2D IG overlay mapped to [0, 1] (0.5 = neutral)" if overlay_mode == "signed_centered" else "Normalized 2D IG overlay [0, 1]"
    add_overlay_colorbar(fig, axes[:n_panels], cmap=overlay_cmap, label=cbar_label, pad=colorbar_pad)

    if title:
        title_wrapped = textwrap.fill(title, width=max(36, 18 * n_cols))
        fig.suptitle(title_wrapped, y=0.965, fontsize=15, fontweight="bold")

    # if add_reading_legend:
    #     add_method_footer(
    #         fig,
    #         score_mode=score_mode,
    #         n_steps=n_steps,
    #         heatmap_mode=heatmap_mode,
    #         legend_title=legend_title,
    #         y=0.03,
    #     )

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
# -----------------------------
# Explain image
# -----------------------------
def explain_image_targets(
    model: nn.Module,
    image_path: str | Path,
    transform,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    target_mode: str = "pred",
    target_class: Optional[int] = None,
    target_classes: Optional[Sequence[int]] = None,
    topk: int = 5,
    class_order: str = "index",
    n_steps: int = 50,
    internal_batch_size: int = 16,
    score_mode: str = "logit",
    signed_clip_percentile: float = 99.0,
    heatmap_mode: str = "mean_rgb",
    blur_kernel: int = 1,
    blur_sigma: float = 0.0,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
    gamma: float = 1.0,
    shared_normalization: bool = False,
    mass_scale_power: float = 1.0,
    suppress_below_mass_ratio: float = 0.0,
    save_single_class_images: bool = True,
    single_output_dir: Optional[str | Path] = None,
    single_figsize: Tuple[float, float] = (16, 5),
    save_class_grid: bool = True,
    class_grid_path: Optional[str | Path] = None,
    class_grid_cols: int = 4,
    class_grid_sort_by: str = "index",
    overlay_cmap: str = "turbo",
    overlay_mode: str = "unsigned",
    overlay_alpha: float = 0.80,
    min_overlay_alpha: float = 0.05,
    threshold: float = 0.08,
    colorbar_pad: float = 0.03,
    show: bool = False,
    dpi: int = 180,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    _, x = load_image_tensor(image_path, transform=transform, device=device)
    label_name = resolve_label_from_path(image_path, class_names=class_names)
    pred = predict_with_probs(model, x)

    pred_class = int(pred["pred_class"][0].item())
    pred_conf = float(pred["pred_conf"][0].item())
    probs = pred["probs"]

    targets_with_desc = resolve_targets(
        probs=probs,
        label_name=label_name,
        class_names=class_names,
        target_mode=target_mode,
        target_class=target_class,
        target_classes=target_classes,
        topk=topk,
        class_order=class_order,
    )
    targets = [t[0] for t in targets_with_desc]

    ig_outputs = integrated_gradients_for_targets(
        model=model,
        x=x,
        mean=mean,
        std=std,
        targets=targets,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        score_mode=score_mode,
        device=device,
    )

    desc_by_target = {cls: desc for cls, desc in targets_with_desc}
    image_name = Path(image_path).stem
    image_name_safe = sanitize_filename(image_name)
    per_class_results: List[Dict[str, Any]] = []

    map_bundle_list: List[Dict[str, Any]] = []
    for out in ig_outputs:
        target_cls = int(out["target_class"])
        maps = build_explanation_maps(
            attributions_single=out["attributions"],
            signed_clip_percentile=signed_clip_percentile,
            heatmap_mode=heatmap_mode,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            gamma=gamma,
        )
        maps["target_class"] = target_cls
        maps["delta"] = float(out["delta"])
        map_bundle_list.append(maps)

    if shared_normalization:
        shared = normalize_heatmaps_shared(
            [m["base_map"] for m in map_bundle_list],
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            gamma=gamma,
            mass_scale_power=mass_scale_power,
            suppress_below_mass_ratio=suppress_below_mass_ratio,
        )
        for m, s in zip(map_bundle_list, shared):
            m["cam_like_map"] = s["heatmap"]
            m["raw_mass"] = s["raw_mass"]
            m["relative_mass"] = s["relative_mass"]
    else:
        max_mass = max(max(float(m["raw_mass"]) for m in map_bundle_list), 1e-8)
        for m in map_bundle_list:
            m["relative_mass"] = float(m["raw_mass"]) / max_mass

    for maps in map_bundle_list:
        target_cls = int(maps["target_class"])
        target_prob = float(probs[0, target_cls].item())
        single_path = None
        if save_single_class_images and single_output_dir is not None:
            single_path = Path(single_output_dir) / f"{image_name_safe}__class_{target_cls:02d}_{sanitize_filename(class_names[target_cls])}.png"
            plot_single_explanation(
                x=x,
                class_names=class_names,
                pred_class=pred_class,
                pred_conf=pred_conf,
                target_class=target_cls,
                target_desc=desc_by_target[target_cls],
                target_prob=target_prob,
                mean=mean,
                std=std,
                signed_map=maps["signed_map"],
                heatmap=maps["cam_like_map"],
                delta=maps["delta"],
                figsize=single_figsize,
                overlay_cmap=overlay_cmap,
                overlay_alpha=overlay_alpha,
                min_overlay_alpha=min_overlay_alpha,
                threshold=threshold,
                add_reading_legend=False,
                score_mode=score_mode,
                n_steps=n_steps,
                heatmap_mode=heatmap_mode,
                show=show,
                save_path=single_path,
                dpi=dpi,
            )

        per_class_results.append({
            "image_path": str(image_path),
            "image_name": Path(image_path).name,
            "label_name": label_name,
            "pred_class": pred_class,
            "pred_label": class_names[pred_class],
            "pred_conf": pred_conf,
            "target_class": target_cls,
            "target_label": class_names[target_cls],
            "target_desc": desc_by_target[target_cls],
            "target_prob": target_prob,
            "delta": float(maps["delta"]),
            "saved_path": str(single_path) if single_path is not None else None,
            "raw_mass": float(maps["raw_mass"]),
            "relative_mass": float(maps.get("relative_mass", 1.0)),
            "signed_map": maps["signed_map"],
            "heatmap": maps["cam_like_map"],
        })

    grid_saved_path = None
    if save_class_grid and class_grid_path is not None:
        title = f"All classes | image = {Path(image_path).name} | pred = {class_names[pred_class]} ({pred_conf:.3f})"
        plot_multi_class_grid(
            x=x,
            class_names=class_names,
            probs=probs,
            mean=mean,
            std=std,
            explanations=per_class_results,
            n_cols=class_grid_cols,
            overlay_cmap=overlay_cmap,
            overlay_mode=overlay_mode,
            overlay_alpha=overlay_alpha,
            min_overlay_alpha=min_overlay_alpha,
            threshold=threshold,
            colorbar_pad=colorbar_pad,
            sort_by=class_grid_sort_by,
            title=title,
            add_reading_legend=False,
            legend_title="How to read this grid",
            score_mode=score_mode,
            n_steps=n_steps,
            heatmap_mode=heatmap_mode,
            show=show,
            save_path=class_grid_path,
            dpi=dpi,
        )
        grid_saved_path = str(class_grid_path)

    # Strip tensors from returned rows to keep CSV/export simple.
    serializable_rows = []
    for row in per_class_results:
        serializable_rows.append({
            k: v for k, v in row.items() if k not in {"signed_map", "heatmap"}
        })

    return {
        "rows": serializable_rows,
        "grid_saved_path": grid_saved_path,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "label_name": label_name,
        "probs": probs[0].detach().cpu(),
    }


# -----------------------------
# Collages and summaries
# -----------------------------
def create_contact_sheet(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    thumb_size: Tuple[int, int] = (700, 320),
    n_cols: int = 2,
    padding: int = 20,
    bg_color: Tuple[int, int, int] = (18, 18, 18),
    add_captions: bool = True,
) -> Optional[Path]:
    image_paths = [Path(p) for p in image_paths if p and Path(p).exists()]
    if not image_paths:
        return None

    thumbs: List[Tuple[Image.Image, str]] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", thumb_size, color=bg_color)
        x = (thumb_size[0] - img.width) // 2
        y = (thumb_size[1] - img.height) // 2
        canvas.paste(img, (x, y))
        thumbs.append((canvas, path.stem))

    n_cols = max(1, min(int(n_cols), len(thumbs)))
    n_rows = math.ceil(len(thumbs) / n_cols)

    caption_h = 30 if add_captions else 0
    sheet_w = padding + n_cols * (thumb_size[0] + padding)
    sheet_h = padding + n_rows * (thumb_size[1] + caption_h + padding)
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg_color)
    draw = ImageDraw.Draw(sheet)

    for idx, (thumb, caption) in enumerate(thumbs):
        row = idx // n_cols
        col = idx % n_cols
        x = padding + col * (thumb_size[0] + padding)
        y = padding + row * (thumb_size[1] + caption_h + padding)
        sheet.paste(thumb, (x, y))
        if add_captions:
            draw.text((x, y + thumb_size[1] + 5), caption[:80], fill=(235, 235, 235))

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    sheet.save(output_path)
    return output_path


def save_summary_csv(rows: Sequence[Dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if not rows:
        raise ValueError("No rows to save")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def create_group_contact_sheets(
    rows: Sequence[Dict[str, Any]],
    output_root: str | Path,
    prefer_grid_paths: bool = True,
) -> Dict[str, List[Path]]:
    output_root = ensure_dir(output_root)
    by_model_dir = ensure_dir(output_root / "by_model")
    by_image_dir = ensure_dir(output_root / "by_image")

    created: Dict[str, List[Path]] = {"by_model": [], "by_image": []}
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    rows_by_image: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        rows_by_model.setdefault(str(row["model_name"]), []).append(row)
        rows_by_image.setdefault(str(row["source_image_name"]), []).append(row)

    def pick_paths(group_rows: Sequence[Dict[str, Any]]) -> List[str]:
        if prefer_grid_paths:
            unique_grid_paths = []
            seen = set()
            for r in group_rows:
                gp = r.get("grid_saved_path")
                if gp and gp not in seen and Path(gp).exists():
                    unique_grid_paths.append(gp)
                    seen.add(gp)
            if unique_grid_paths:
                return unique_grid_paths
        return [r["saved_path"] for r in group_rows if r.get("saved_path") and Path(r["saved_path"]).exists()]

    for model_name, model_rows in rows_by_model.items():
        img_paths = pick_paths(model_rows)
        out_path = by_model_dir / f"{sanitize_filename(model_name)}__contact_sheet.png"
        created_path = create_contact_sheet(img_paths, out_path, n_cols=min(2, max(1, len(img_paths))))
        if created_path:
            created["by_model"].append(created_path)

    for image_name, image_rows in rows_by_image.items():
        img_paths = pick_paths(image_rows)
        out_path = by_image_dir / f"{sanitize_filename(image_name)}__all_models.png"
        created_path = create_contact_sheet(img_paths, out_path, n_cols=min(3, max(1, len(img_paths))))
        if created_path:
            created["by_image"].append(created_path)

    return created


# -----------------------------
# Main runner
# -----------------------------
# Thesis-faithful variant: black baseline, 50 IG steps, RGB-channel averaging, per-map [0,1] normalization.
def run_experiments(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    set_seed(int(config.get("seed", 42)))

    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)
    mean = config.get("mean", DEFAULT_MEAN)
    std = config.get("std", DEFAULT_STD)
    image_size = config.get("image_size")
    transform = build_transform(mean=mean, std=std, image_size=image_size)

    images_cfg = config["images"]
    selected_images = select_images(
        image_dir=images_cfg.get("image_dir"),
        image_paths=images_cfg.get("image_paths"),
        image_indices=images_cfg.get("image_indices"),
        max_images=images_cfg.get("max_images"),
        shuffle=images_cfg.get("shuffle", False),
        seed=config.get("seed", 42),
    )

    output_root = ensure_dir(config.get("output_dir", "./ig_outputs_v3"))
    show_inline = bool(config.get("show_inline", False))
    plot_cfg = dict(config.get("plot", {}))
    ig_cfg = dict(config.get("ig", {}))
    export_cfg = dict(config.get("export", {}))

    rows: List[Dict[str, Any]] = []

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        print(f"\n[MODEL] {model_name}")
        model = build_model_from_cfg(model_cfg, device=DEVICE)
        model_out_dir = ensure_dir(output_root / sanitize_filename(model_name))
        single_dir = ensure_dir(model_out_dir / "single_classes")
        grid_dir = ensure_dir(model_out_dir / "class_grids")

        for i, image_path in enumerate(selected_images, start=1):
            print(f"  [{i:03d}/{len(selected_images):03d}] {image_path}")
            safe_image_name = sanitize_filename(Path(image_path).stem)
            class_grid_path = grid_dir / f"{safe_image_name}__class_grid.png"

            explained = explain_image_targets(
                model=model,
                image_path=image_path,
                transform=transform,
                class_names=class_names,
                mean=mean,
                std=std,
                target_mode=ig_cfg.get("target_mode", "pred"),
                target_class=ig_cfg.get("target_class"),
                target_classes=ig_cfg.get("target_classes"),
                topk=int(ig_cfg.get("topk", 5)),
                class_order=ig_cfg.get("class_order", "index"),
                n_steps=int(ig_cfg.get("n_steps", 50)),
                internal_batch_size=int(ig_cfg.get("internal_batch_size", 16)),
                score_mode=ig_cfg.get("score_mode", "logit"),
                signed_clip_percentile=float(plot_cfg.get("signed_clip_percentile", 99.0)),
                heatmap_mode=plot_cfg.get("heatmap_mode", "mean_rgb"),
                blur_kernel=int(plot_cfg.get("blur_kernel", 1)),
                blur_sigma=float(plot_cfg.get("blur_sigma", 0.0)),
                low_percentile=float(plot_cfg.get("low_percentile", 0.0)),
                high_percentile=float(plot_cfg.get("high_percentile", 100.0)),
                gamma=float(plot_cfg.get("gamma", 1.0)),
                shared_normalization=bool(plot_cfg.get("shared_normalization", False)),
                mass_scale_power=float(plot_cfg.get("mass_scale_power", 1.0)),
                suppress_below_mass_ratio=float(plot_cfg.get("suppress_below_mass_ratio", 0.0)),
                save_single_class_images=bool(export_cfg.get("save_single_class_images", True)),
                single_output_dir=single_dir,
                single_figsize=tuple(plot_cfg.get("single_figsize", (16, 5))),
                save_class_grid=bool(export_cfg.get("save_class_grid", True)),
                class_grid_path=class_grid_path,
                class_grid_cols=int(plot_cfg.get("class_grid_cols", 4)),
                class_grid_sort_by=plot_cfg.get("class_grid_sort_by", "index"),
                overlay_cmap=plot_cfg.get("overlay_cmap", "inferno"),
                overlay_alpha=float(plot_cfg.get("overlay_alpha", 0.80)),
                min_overlay_alpha=float(plot_cfg.get("min_overlay_alpha", 0.05)),
                threshold=float(plot_cfg.get("threshold", 0.08)),
                show=show_inline,
                dpi=int(plot_cfg.get("dpi", 180)),
                device=DEVICE,
            )

            for row in explained["rows"]:
                row["model_name"] = model_name
                row["source_image_name"] = Path(image_path).name
                row["grid_saved_path"] = explained["grid_saved_path"]
                rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = save_summary_csv(rows, output_root / "summary.csv")
    print(f"\nSaved summary: {summary_path}")

    if config.get("make_contact_sheets", True):
        created = create_group_contact_sheets(rows, output_root / "contact_sheets", prefer_grid_paths=True)
        print(f"Contact sheets by model: {len(created['by_model'])}")
        print(f"Contact sheets by image: {len(created['by_image'])}")

    return rows
