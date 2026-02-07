#!/usr/bin/env python3
"""
Timelapse generator — turns a folder of photos into an MP4 video.

Usage:
    python3 timelapse.py /path/to/photos
    python3 timelapse.py /path/to/photos --output my_timelapse.mp4 --fps 12

Requires: Pillow, opencv-python, numpy, ffmpeg (must be on PATH)
    pip install Pillow opencv-python numpy

Optional (for deep-learning alignment):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install git+https://github.com/cvg/LightGlue.git
"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ExifTags, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV and NumPy are required: pip install opencv-python numpy")
    sys.exit(1)

# Optional deep-learning matcher
try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    HAS_LIGHTGLUE = True
    # Use GPU if available (ROCm for AMD, CUDA for NVIDIA)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_LIGHTGLUE = False
    DEVICE = None


WIDTH = 1920
HEIGHT = 1080
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

# Tree-trunk detection parameters
WORK_WIDTH = 960
WORK_HEIGHT = 540


def _to_degrees(value) -> float:
    """Convert EXIF GPS rational values to decimal degrees."""
    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


def get_exif_gps(path: str) -> tuple[float, float] | None:
    """Extract GPS lat/lon from EXIF data."""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        gps_info = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, "")
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = ExifTags.GPSTAGS.get(gps_tag_id, "")
                    gps_info[gps_tag] = gps_value
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = _to_degrees(gps_info["GPSLatitude"])
            lon = _to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLatitudeRef", "N") == "S":
                lat = -lat
            if gps_info.get("GPSLongitudeRef", "E") == "W":
                lon = -lon
            return (lat, lon)
    except Exception:
        pass
    return None


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    import math
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_exif_date(path: str) -> datetime | None:
    """Extract the date taken from EXIF data."""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, "")
            if tag == "DateTimeOriginal":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def get_file_date(path: str) -> datetime:
    """Fallback: use file modification time."""
    return datetime.fromtimestamp(os.path.getmtime(path))


def get_date(path: str) -> datetime:
    """Get photo date from EXIF, falling back to file mod time."""
    return get_exif_date(path) or get_file_date(path)


def resize_and_crop(img: Image.Image) -> Image.Image:
    """Resize and center-crop to exactly WIDTHxHEIGHT."""
    # Handle EXIF orientation
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    img_ratio = img.width / img.height
    target_ratio = WIDTH / HEIGHT

    if img_ratio > target_ratio:
        # Image is wider — fit height, crop width
        new_height = HEIGHT
        new_width = int(HEIGHT * img_ratio)
    else:
        # Image is taller — fit width, crop height
        new_width = WIDTH
        new_height = int(WIDTH / img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Center crop
    left = (new_width - WIDTH) // 2
    top = (new_height - HEIGHT) // 2
    img = img.crop((left, top, left + WIDTH, top + HEIGHT))

    return img


def add_date_overlay(img: Image.Image, date: datetime) -> Image.Image:
    """Add date text to the bottom-right corner."""
    draw = ImageDraw.Draw(img)
    text = date.strftime("%B %-d, %Y")

    # Try to load a good font, fall back to default
    font = None
    font_size = 36
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Position: bottom-right with padding
    padding = 20
    x = WIDTH - tw - padding
    y = HEIGHT - th - padding

    # Draw shadow then text
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 200), font=font)
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    return img


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def detect_trunks(image: np.ndarray) -> list[dict] | None:
    """Detect two tree trunks at the left and right edges of the frame.

    Uses column brightness profiling: the trunks are dark regions at the
    left and right edges with a brighter gap between them.

    Returns [left_trunk, right_trunk] dicts with centroid, inner_edge,
    bottom_center, bbox, area — or None if the pattern isn't found.
    """
    work = cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Use the upper-middle band where the sky gap is most visible
    roi_top = int(WORK_HEIGHT * 0.1)
    roi_bottom = int(WORK_HEIGHT * 0.6)
    roi = gray[roi_top:roi_bottom, :]

    # Mean brightness per column
    col_brightness = np.mean(roi, axis=0).astype(np.float64)

    # Smooth heavily to get the broad dark-bright-dark pattern
    kernel_size = WORK_WIDTH // 20
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(col_brightness, np.ones(kernel_size) / kernel_size, mode="same")

    # Threshold at the midpoint between min and max brightness
    threshold = (np.min(smoothed) + np.max(smoothed)) / 2

    # Scan from left to find where brightness first exceeds threshold
    left_edge = None
    for x in range(len(smoothed)):
        if smoothed[x] > threshold:
            left_edge = x
            break

    # Scan from right to find where brightness first exceeds threshold
    right_edge = None
    for x in range(len(smoothed) - 1, -1, -1):
        if smoothed[x] > threshold:
            right_edge = x
            break

    if left_edge is None or right_edge is None:
        return None

    # Validate the pattern
    # Left edge (inner edge of left trunk) should be in left portion
    if left_edge > WORK_WIDTH * 0.45:
        return None
    # Right edge (inner edge of right trunk) should be in right portion
    if right_edge < WORK_WIDTH * 0.55:
        return None
    # Must have a meaningful gap between the trunks
    if (right_edge - left_edge) < WORK_WIDTH * 0.2:
        return None
    # The bright region between trunks must be significantly brighter than edges
    gap_brightness = np.mean(smoothed[left_edge:right_edge + 1])
    left_brightness = np.mean(smoothed[:max(left_edge, 1)])
    right_brightness = np.mean(smoothed[min(right_edge, WORK_WIDTH - 2):])
    edge_brightness = (left_brightness + right_brightness) / 2
    if gap_brightness < edge_brightness * 1.3:
        return None

    left_trunk = {
        "centroid": (left_edge / 2, WORK_HEIGHT / 2),
        "bbox": (0, 0, left_edge, WORK_HEIGHT),
        "bottom_center": (left_edge / 2, WORK_HEIGHT),
        "inner_edge": left_edge,
        "area": left_edge * WORK_HEIGHT,
    }
    right_trunk = {
        "centroid": ((right_edge + WORK_WIDTH) / 2, WORK_HEIGHT / 2),
        "bbox": (right_edge, 0, WORK_WIDTH - right_edge, WORK_HEIGHT),
        "bottom_center": ((right_edge + WORK_WIDTH) / 2, WORK_HEIGHT),
        "inner_edge": right_edge,
        "area": (WORK_WIDTH - right_edge) * WORK_HEIGHT,
    }

    return [left_trunk, right_trunk]


def detect_wall_y(image: np.ndarray) -> float | None:
    """Detect the horizontal wall position using Sobel edge detection.

    Looks for the strongest horizontal edge in the center strip (between trunks)
    in the y=180-350 range of the 960x540 working resolution.

    Returns wall y-position in WORK_HEIGHT coords, or None if not found.
    """
    work = cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Center strip between trunks
    x1 = int(WORK_WIDTH * 0.25)
    x2 = int(WORK_WIDTH * 0.75)
    center = gray[:, x1:x2]

    # Horizontal edge detection
    sobel_y = cv2.Sobel(center, cv2.CV_64F, 0, 1, ksize=5)
    edge_profile = np.mean(np.abs(sobel_y), axis=1)

    # Smooth to avoid noise spikes
    kernel = np.ones(11) / 11
    edge_smooth = np.convolve(edge_profile, kernel, mode="same")

    # Search in the expected wall region
    search_start = int(WORK_HEIGHT * 0.33)
    search_end = int(WORK_HEIGHT * 0.65)
    if search_end <= search_start:
        return None

    region = edge_smooth[search_start:search_end]
    best_y = search_start + np.argmax(region)
    strength = edge_smooth[best_y]

    # Reject if edge is too weak (no wall visible)
    if strength < 800:
        return None

    return float(best_y)


def compute_trunk_alignment(trunks: list[dict], ref_trunks: list[dict],
                            frame_shape: tuple[int, int]) -> np.ndarray | None:
    """Compute affine transform to align frame trunks to reference trunks.

    Uses trunk bottom-centers and centroids as control points.
    frame_shape is (height, width) of the full-resolution frame.
    """
    sx = frame_shape[1] / WORK_WIDTH
    sy = frame_shape[0] / WORK_HEIGHT

    def scale_pt(pt):
        return (pt[0] * sx, pt[1] * sy)

    src_pts = np.float32([
        scale_pt(trunks[0]["bottom_center"]),
        scale_pt(trunks[0]["centroid"]),
        scale_pt(trunks[1]["bottom_center"]),
        scale_pt(trunks[1]["centroid"]),
    ])
    dst_pts = np.float32([
        scale_pt(ref_trunks[0]["bottom_center"]),
        scale_pt(ref_trunks[0]["centroid"]),
        scale_pt(ref_trunks[1]["bottom_center"]),
        scale_pt(ref_trunks[1]["centroid"]),
    ])

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return None

    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    if scale < 0.7 or scale > 1.4:
        return None

    angle = np.abs(np.arctan2(M[0, 1], M[0, 0]))
    if angle > np.radians(15):
        return None

    return M


def align_frame_by_trunks(frame: np.ndarray, M: np.ndarray,
                           output_size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    """Apply affine transform to align frame."""
    return cv2.warpAffine(frame, M, output_size,
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def align_sequential(prev_gray: np.ndarray, curr_frame: np.ndarray,
                     detector, center_mask: np.ndarray,
                     flann: cv2.FlannBasedMatcher | None = None,
                     lg_extractor=None, lg_matcher=None) -> tuple[np.ndarray | None, np.ndarray]:
    """Align curr_frame to prev_gray using feature matching in the center region,
    refined with ECC (Enhanced Correlation Coefficient) optimization.

    Uses LightGlue (deep learning) if available, falls back to SIFT+FLANN.
    Returns (affine_matrix_or_None, curr_gray).
    """
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Try LightGlue first (much better cross-season matching)
    if lg_extractor is not None and lg_matcher is not None:
        M = _lightglue_match(prev_gray, curr_gray, lg_extractor, lg_matcher,
                             center_mask)
        if M is not None:
            # Refine with ECC
            M_refined = _ecc_refine(prev_gray, curr_gray, M, center_mask)
            if M_refined is not None:
                M = M_refined
            return M, curr_gray

    # Fall back to SIFT + FLANN
    kp1, desc1 = detector.detectAndCompute(prev_gray, center_mask)
    kp2, desc2 = detector.detectAndCompute(curr_gray, center_mask)

    if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return _ecc_fallback(prev_gray, curr_gray, center_mask)

    if flann is not None:
        matches = flann.knnMatch(desc2, desc1, k=2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(desc2, desc1, k=2)

    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < 8:
        return _ecc_fallback(prev_gray, curr_gray, center_mask)

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                        ransacReprojThreshold=3.0)
    if M is None:
        return _ecc_fallback(prev_gray, curr_gray, center_mask)

    # Reject if transform is too large
    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    tx, ty = M[0, 2], M[1, 2]
    if abs(scale - 1.0) > 0.15 or abs(tx) > 80 or abs(ty) > 80:
        return _ecc_fallback(prev_gray, curr_gray, center_mask)

    # Refine with ECC using the SIFT result as initial guess
    M_refined = _ecc_refine(prev_gray, curr_gray, M, center_mask)
    if M_refined is not None:
        M = M_refined

    return M, curr_gray


def _ecc_fallback(prev_gray: np.ndarray, curr_gray: np.ndarray,
                  center_mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """Try ECC alignment when feature matching fails."""
    M = np.eye(2, 3, dtype=np.float64)
    M_result = _ecc_refine(prev_gray, curr_gray, M, center_mask)
    return M_result, curr_gray


def _ecc_refine(prev_gray: np.ndarray, curr_gray: np.ndarray,
                init_warp: np.ndarray, center_mask: np.ndarray) -> np.ndarray | None:
    """Refine alignment using ECC (Enhanced Correlation Coefficient).

    Works on downscaled images for speed, using the center region.
    """
    # Downscale for speed
    scale = 0.5
    h, w = prev_gray.shape
    small_prev = cv2.resize(prev_gray, (int(w * scale), int(h * scale)))
    small_curr = cv2.resize(curr_gray, (int(w * scale), int(h * scale)))
    small_mask = cv2.resize(center_mask, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_NEAREST)

    # Scale the initial warp translation
    warp = init_warp.copy().astype(np.float32)
    warp[0, 2] *= scale
    warp[1, 2] *= scale

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp = cv2.findTransformECC(small_prev, small_curr, warp,
                                        cv2.MOTION_EUCLIDEAN, criteria,
                                        small_mask, 5)
        # Scale translation back
        warp[0, 2] /= scale
        warp[1, 2] /= scale

        # Sanity check
        s = np.sqrt(warp[0, 0] ** 2 + warp[0, 1] ** 2)
        tx, ty = warp[0, 2], warp[1, 2]
        if abs(s - 1.0) > 0.15 or abs(tx) > 80 or abs(ty) > 80:
            return None
        return warp.astype(np.float64)
    except cv2.error:
        return None


def _lightglue_match(prev_gray: np.ndarray, curr_gray: np.ndarray,
                     lg_extractor, lg_matcher,
                     center_mask: np.ndarray) -> np.ndarray | None:
    """Match features using SuperPoint + LightGlue (deep learning).

    Returns affine matrix or None. Uses center_mask to filter matches.
    """
    h, w = prev_gray.shape

    # LightGlue expects float32 tensor in [0, 1], shape (1, 1, H, W)
    t_prev = torch.from_numpy(prev_gray).float()[None, None].to(DEVICE) / 255.0
    t_curr = torch.from_numpy(curr_gray).float()[None, None].to(DEVICE) / 255.0

    with torch.no_grad():
        feats0 = lg_extractor.extract(t_prev)
        feats1 = lg_extractor.extract(t_curr)
        matches01 = lg_matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0 = feats0["keypoints"].cpu().numpy()
    kpts1 = feats1["keypoints"].cpu().numpy()
    match_indices = matches01["matches"].cpu().numpy()

    # Filter: only use matches where both points are in center mask
    good_src = []
    good_dst = []
    for idx0, idx1 in match_indices:
        pt0 = kpts0[idx0]
        pt1 = kpts1[idx1]
        # Check both points are in the mask region
        y0, x0 = int(pt0[1]), int(pt0[0])
        y1, x1_coord = int(pt1[1]), int(pt1[0])
        if (0 <= y0 < h and 0 <= x0 < w and center_mask[y0, x0] > 0 and
                0 <= y1 < h and 0 <= x1_coord < w and center_mask[y1, x1_coord] > 0):
            good_dst.append(pt0)  # prev = destination
            good_src.append(pt1)  # curr = source

    if len(good_src) < 8:
        return None

    src_pts = np.float32(good_src).reshape(-1, 1, 2)
    dst_pts = np.float32(good_dst).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
    if M is None:
        return None

    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    tx, ty = M[0, 2], M[1, 2]
    if abs(scale - 1.0) > 0.15 or abs(tx) > 80 or abs(ty) > 80:
        return None

    return M


def make_center_mask(width: int, height: int) -> np.ndarray:
    """Create a mask focused on the foreground trees and stone wall,
    excluding big trunks, sky, and bare ground."""
    mask = np.zeros((height, width), dtype=np.uint8)
    # Focus on center where small trees and stone wall are
    x1 = int(width * 0.25)
    x2 = int(width * 0.75)
    y1 = int(height * 0.30)
    y2 = int(height * 0.75)
    mask[y1:y2, x1:x2] = 255
    return mask


# Target positions for trunk inner edges in output frame (fraction of WIDTH)
TARGET_LEFT_EDGE = 0.15   # left trunk inner edge at 15% from left
TARGET_RIGHT_EDGE = 0.85  # right trunk inner edge at 85% from left
TARGET_WALL_Y = 0.45      # wall at 45% down from top of output frame


def compute_crop_box(img_size: tuple[int, int],
                     trunks: list[dict],
                     wall_y: float | None = None) -> tuple[float, float, float, float]:
    """Compute crop box that locks trunk inner edges and wall to fixed output positions.

    Returns (left, top, crop_w, crop_h) in original image coords.
    """
    orig_w, orig_h = img_size
    sx = orig_w / WORK_WIDTH
    sy = orig_h / WORK_HEIGHT

    # Trunk inner edges in original image coords
    left_edge_orig = trunks[0]["inner_edge"] * sx
    right_edge_orig = trunks[1]["inner_edge"] * sx
    trunk_span_orig = right_edge_orig - left_edge_orig

    # Target span in output pixels
    target_span = (TARGET_RIGHT_EDGE - TARGET_LEFT_EDGE) * WIDTH

    # Scale: how many original pixels per output pixel
    scale = trunk_span_orig / target_span

    # Crop dimensions in original image coords
    crop_w = WIDTH * scale
    crop_h = HEIGHT * scale

    # Clamp to image bounds
    crop_w = min(crop_w, orig_w)
    crop_h = min(crop_h, orig_h)

    # Ensure aspect ratio
    if crop_w / crop_h > WIDTH / HEIGHT:
        crop_h = crop_w * (HEIGHT / WIDTH)
    else:
        crop_w = crop_h * (WIDTH / HEIGHT)

    # Position: left trunk inner edge should map to TARGET_LEFT_EDGE
    scale = crop_w / WIDTH
    left = left_edge_orig - TARGET_LEFT_EDGE * crop_w

    # Vertical: anchor on wall if detected, otherwise center
    if wall_y is not None:
        wall_y_orig = wall_y * sy
        top = wall_y_orig - TARGET_WALL_Y * crop_h
    else:
        top = (orig_h - crop_h) / 2

    left = max(0, min(left, orig_w - crop_w))
    top = max(0, min(top, orig_h - crop_h))

    return left, top, crop_w, crop_h


def apply_crop(img: Image.Image, left: float, top: float,
               crop_w: float, crop_h: float,
               offset_x: float = 0, offset_y: float = 0) -> Image.Image:
    """Apply a crop box with optional pixel offset, clamped to image bounds."""
    orig_w, orig_h = img.size
    left = left + offset_x
    top = top + offset_y
    left = max(0, min(left, orig_w - crop_w))
    top = max(0, min(top, orig_h - crop_h))
    img = img.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    return img


def crop_around_trunks(img: Image.Image, trunks: list[dict],
                       wall_y: float | None = None) -> Image.Image:
    """Crop image so the two trunks are centered in the frame at WIDTHxHEIGHT."""
    left, top, crop_w, crop_h = compute_crop_box(img.size, trunks, wall_y)
    return apply_crop(img, left, top, crop_w, crop_h)


def pick_reference_with_trunks(photos: list[tuple[datetime, str]]) -> tuple[np.ndarray, list[dict]] | None:
    """Find a reference frame with two detectable trunks."""
    mid = len(photos) // 2
    indices = sorted(range(len(photos)), key=lambda i: abs(i - mid))

    for idx in indices[:20]:
        _, path = photos[idx]
        img = Image.open(path).convert("RGB")
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        cv_img = pil_to_cv(img)
        trunks = detect_trunks(cv_img)
        if trunks is not None:
            img = crop_around_trunks(img, trunks)
            ref_cv = pil_to_cv(img)
            ref_trunks = detect_trunks(ref_cv)
            if ref_trunks is not None:
                print(f"  Reference frame: {os.path.basename(path)} (index {idx})")
                return (ref_cv, ref_trunks)

    return None


def debug_draw_trunks(image: np.ndarray, trunks: list[dict] | None,
                      path: str, output_dir: str) -> None:
    """Save a debug image with detected trunk inner edges highlighted."""
    vis = cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT))
    if trunks is not None:
        # Draw inner edge lines
        left_x = trunks[0]["inner_edge"]
        right_x = trunks[1]["inner_edge"]
        cv2.line(vis, (left_x, 0), (left_x, WORK_HEIGHT), (0, 255, 0), 2)
        cv2.line(vis, (right_x, 0), (right_x, WORK_HEIGHT), (0, 0, 255), 2)
        # Draw centroids
        for i, t in enumerate(trunks):
            cx, cy = int(t["centroid"][0]), int(t["centroid"][1])
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.circle(vis, (cx, cy), 5, color, -1)
    else:
        cv2.putText(vis, "NO TRUNKS DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    name = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(output_dir, f"debug_{name}.jpg"), vis)


def pick_reference(photos: list[tuple[datetime, str]]) -> np.ndarray:
    """Pick the reference frame — use the middle photo (likely stable framing)."""
    mid = len(photos) // 2
    img = Image.open(photos[mid][1]).convert("RGB")
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = resize_and_crop(img)
    return pil_to_cv(img)


def align_to_reference(frame: np.ndarray, ref_gray: np.ndarray,
                       ref_kp: list, ref_desc: np.ndarray,
                       detector: cv2.ORB) -> np.ndarray | None:
    """Align a frame to the reference using feature matching + homography.
    Returns the warped frame, or None if alignment fails."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = detector.detectAndCompute(gray, None)

    if desc is None or len(kp) < 10:
        return None

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(desc, ref_desc, k=2)

    # Lowe's ratio test
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 15:
        return None

    # Compute homography
    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    # Check that the homography is reasonable (not wildly distorting)
    det = np.linalg.det(H[:2, :2])
    if det < 0.5 or det > 2.0:
        return None

    h, w = ref_gray.shape
    return cv2.warpPerspective(frame, H, (w, h), borderMode=cv2.BORDER_REFLECT)


def find_photos(folder: str, filter_gps: bool = True, radius: int = 50) -> list[tuple[datetime, str]]:
    """Find all photos, optionally filter by GPS to the most common location, and sort by date."""
    raw: list[tuple[datetime, str, tuple[float, float] | None]] = []
    for entry in Path(folder).rglob("*"):
        if entry.suffix.lower() in EXTENSIONS and entry.is_file():
            path = str(entry)
            date = get_date(path)
            gps = get_exif_gps(path) if filter_gps else None
            raw.append((date, path, gps))

    if not raw:
        return []

    if filter_gps:
        # Find the most common GPS cluster
        gps_photos = [(d, p, g) for d, p, g in raw if g is not None]
        no_gps = [(d, p, g) for d, p, g in raw if g is None]

        if len(gps_photos) < 2:
            print("  Few photos have GPS data — skipping location filter")
            return sorted([(d, p) for d, p, _ in raw], key=lambda x: x[0])

        # Use median lat/lon as the center (robust against outliers)
        lats = sorted([g[0] for _, _, g in gps_photos])
        lons = sorted([g[1] for _, _, g in gps_photos])
        center_lat = lats[len(lats) // 2]
        center_lon = lons[len(lons) // 2]

        kept = []
        dropped = 0
        for date, path, gps in gps_photos:
            dist = haversine(center_lat, center_lon, gps[0], gps[1])
            if dist <= radius:
                kept.append((date, path))
            else:
                dropped += 1

        if no_gps:
            print(f"  {len(no_gps)} photos have no GPS data — including them")
            for date, path, _ in no_gps:
                kept.append((date, path))

        if dropped:
            print(f"  Filtered out {dropped} photos taken elsewhere (>{radius}m from main spot)")

        kept.sort(key=lambda x: x[0])
        return kept

    return sorted([(d, p) for d, p, _ in raw], key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description="Generate a timelapse MP4 from photos")
    parser.add_argument("folder", help="Path to folder containing photos")
    parser.add_argument("--output", "-o", default="timelapse.mp4", help="Output MP4 path (default: timelapse.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("--no-date", action="store_true", help="Disable date overlay")
    parser.add_argument("--no-gps-filter", action="store_true", help="Disable GPS location filtering")
    parser.add_argument("--gps-radius", type=int, default=30, help="GPS filter radius in meters (default: 30)")
    parser.add_argument("--no-align", action="store_true", help="Disable feature-based image alignment")
    parser.add_argument("--tree-detect", action="store_true", help="Detect two tree trunks for alignment and filtering")
    parser.add_argument("--tree-debug", action="store_true", help="Save debug images showing detected trunks")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of photos to process (0 = all)")
    args = parser.parse_args()

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: ffmpeg not found. Install it first.")
        print("  Fedora: sudo dnf install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  macOS:  brew install ffmpeg")
        sys.exit(1)

    # Find and sort photos
    print(f"Scanning {args.folder} ...")
    photos = find_photos(args.folder, filter_gps=not args.no_gps_filter, radius=args.gps_radius)
    if not photos:
        print("No photos found.")
        sys.exit(1)

    if args.limit > 0:
        photos = photos[:args.limit]
        print(f"Limited to first {len(photos)} photos")

    print(f"Found {len(photos)} photos from {photos[0][0].strftime('%Y-%m-%d')} to {photos[-1][0].strftime('%Y-%m-%d')}")
    duration = len(photos) / args.fps
    print(f"Output: {duration:.1f}s at {args.fps}fps")

    # Set up alignment
    tree_mode = args.tree_detect
    align = not args.no_align
    ref_trunks = None

    if tree_mode:
        print("Setting up tree-trunk detection...")
        detector = cv2.SIFT_create(nfeatures=3000)
        flann_index = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        flann_search = dict(checks=50)
        flann = cv2.FlannBasedMatcher(flann_index, flann_search)
        center_mask = make_center_mask(WIDTH, HEIGHT)

        # Initialize LightGlue if available
        lg_extractor = None
        lg_matcher = None
        if HAS_LIGHTGLUE:
            device_name = "GPU" if DEVICE.type == "cuda" else "CPU"
            print(f"  LightGlue available — using deep learning alignment ({device_name})")
            lg_extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
            lg_matcher = LightGlue(features="superpoint").eval().to(DEVICE)
        else:
            print("  LightGlue not available — using SIFT+FLANN")
        if args.tree_debug:
            debug_dir = os.path.join(os.path.dirname(args.output) or ".", "tree_debug")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"  Debug images will be saved to {debug_dir}/")

        # Pass 1: detect trunks in all photos
        print("Pass 1: Detecting trunks...")
        trunk_data = []  # (index, date, path, trunks)
        tree_discards = 0
        for i, (date, path) in enumerate(photos):
            try:
                img = Image.open(path).convert("RGB")
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                cv_img = pil_to_cv(img)
                trunks = detect_trunks(cv_img)

                if args.tree_debug:
                    debug_draw_trunks(cv_img, trunks, path, debug_dir)

                if trunks is None:
                    tree_discards += 1
                else:
                    wall_y = detect_wall_y(cv_img)
                    trunk_data.append((i, date, path, trunks, wall_y))
            except Exception as e:
                print(f"  Skipping {path}: {e}")
                tree_discards += 1

            if (i + 1) % 100 == 0 or i == len(photos) - 1:
                print(f"  {i + 1}/{len(photos)}")

        print(f"  Kept {len(trunk_data)}, discarded {tree_discards}")

        if not trunk_data:
            print("Error: No photos with detectable trunks.")
            sys.exit(1)

        # Smooth trunk positions and wall y with a large rolling window
        smooth_window = max(7, len(trunk_data) // 10)
        if smooth_window % 2 == 0:
            smooth_window += 1
        left_edges = np.array([t[3][0]["inner_edge"] for t in trunk_data], dtype=np.float64)
        right_edges = np.array([t[3][1]["inner_edge"] for t in trunk_data], dtype=np.float64)

        # Wall y: fill missing detections with interpolation
        raw_wall_y = [t[4] for t in trunk_data]
        wall_detected = sum(1 for w in raw_wall_y if w is not None)
        if wall_detected > len(trunk_data) * 0.5:
            # Enough wall detections — interpolate missing values
            median_wall = np.median([w for w in raw_wall_y if w is not None])
            wall_y_arr = np.array([w if w is not None else median_wall for w in raw_wall_y],
                                  dtype=np.float64)
            use_wall = True
            print(f"  Wall detected in {wall_detected}/{len(trunk_data)} frames")
        else:
            wall_y_arr = None
            use_wall = False
            print(f"  Wall detected in only {wall_detected}/{len(trunk_data)} frames — not using")

        kernel = np.ones(smooth_window) / smooth_window
        left_smooth = np.convolve(left_edges, kernel, mode="same")
        right_smooth = np.convolve(right_edges, kernel, mode="same")
        if use_wall:
            wall_smooth = np.convolve(wall_y_arr, kernel, mode="same")
        # Fix edges of convolution
        half = smooth_window // 2
        for j in range(half):
            w = j + half + 1
            left_smooth[j] = np.mean(left_edges[:w])
            right_smooth[j] = np.mean(right_edges[:w])
            left_smooth[-(j + 1)] = np.mean(left_edges[-w:])
            right_smooth[-(j + 1)] = np.mean(right_edges[-w:])
            if use_wall:
                wall_smooth[j] = np.mean(wall_y_arr[:w])
                wall_smooth[-(j + 1)] = np.mean(wall_y_arr[-w:])

        print(f"  Smoothed positions (window={smooth_window})")

        # Pick a reference frame for alignment (use middle of kept frames)
        ref_idx = len(trunk_data) // 2
        ref_entry = trunk_data[ref_idx]
        ref_img = Image.open(ref_entry[2]).convert("RGB")
        try:
            from PIL import ImageOps
            ref_img = ImageOps.exif_transpose(ref_img)
        except Exception:
            pass
        ref_smooth_trunks = [
            {**ref_entry[3][0], "inner_edge": int(left_smooth[ref_idx]),
             "centroid": (left_smooth[ref_idx] / 2, WORK_HEIGHT / 2),
             "bottom_center": (left_smooth[ref_idx] / 2, WORK_HEIGHT)},
            {**ref_entry[3][1], "inner_edge": int(right_smooth[ref_idx]),
             "centroid": ((right_smooth[ref_idx] + WORK_WIDTH) / 2, WORK_HEIGHT / 2),
             "bottom_center": ((right_smooth[ref_idx] + WORK_WIDTH) / 2, WORK_HEIGHT)},
        ]
        ref_wall_y = wall_smooth[ref_idx] if use_wall else None
        ref_img = crop_around_trunks(ref_img, ref_smooth_trunks, ref_wall_y)
        ref_gray = cv2.cvtColor(pil_to_cv(ref_img), cv2.COLOR_BGR2GRAY)
        print(f"  Alignment reference: {os.path.basename(ref_entry[2])}")

        # Pass 2: rough crop, sequential alignment, accumulate offsets
        print("Pass 2: Finding alignment offsets (sequential)...")
        align_ok = 0
        align_fail = 0
        prev_crop_gray = ref_gray  # start chain from reference
        cum_ox, cum_oy = 0.0, 0.0
        # Store (crop_box, offset_x, offset_y, date, path) per frame
        offset_data = []
    elif align:
        print("Setting up alignment (using middle photo as reference)...")
        ref_cv = pick_reference(photos)
        ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB_create(nfeatures=3000)
        ref_kp, ref_desc = detector.detectAndCompute(ref_gray, None)
        align_failures = 0

    # Process frames into temp directory
    frame_count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        if tree_mode:
            # Pass 2: rough crop each frame, match to previous frame
            for j, (orig_i, date, path, trunks, _wall_y_raw) in enumerate(trunk_data):
                try:
                    img = Image.open(path).convert("RGB")
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass

                    smooth_trunks = [
                        {**trunks[0], "inner_edge": int(left_smooth[j]),
                         "centroid": (left_smooth[j] / 2, WORK_HEIGHT / 2),
                         "bottom_center": (left_smooth[j] / 2, WORK_HEIGHT)},
                        {**trunks[1], "inner_edge": int(right_smooth[j]),
                         "centroid": ((right_smooth[j] + WORK_WIDTH) / 2, WORK_HEIGHT / 2),
                         "bottom_center": ((right_smooth[j] + WORK_WIDTH) / 2, WORK_HEIGHT)},
                    ]

                    frame_wall_y = wall_smooth[j] if use_wall else None
                    crop_box = compute_crop_box(img.size, smooth_trunks, frame_wall_y)
                    rough_crop = apply_crop(img, *crop_box)
                    frame_cv = pil_to_cv(rough_crop)
                    curr_gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)

                    # Match to previous frame (sequential)
                    M, _ = align_sequential(prev_crop_gray, frame_cv,
                                            detector, center_mask, flann,
                                            lg_extractor, lg_matcher)
                    if M is not None:
                        # Accumulate the per-frame offset in original coords
                        scale_x = crop_box[2] / WIDTH
                        scale_y = crop_box[3] / HEIGHT
                        cum_ox += -M[0, 2] * scale_x
                        cum_oy += -M[1, 2] * scale_y
                        align_ok += 1
                    else:
                        align_fail += 1

                    offset_data.append((crop_box, cum_ox, cum_oy, date, path))
                    prev_crop_gray = curr_gray
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    continue

                if (j + 1) % 50 == 0 or j == len(trunk_data) - 1:
                    print(f"  {j + 1}/{len(trunk_data)}")

            print(f"  Discarded {tree_discards}/{len(photos)} photos (no trunks detected)")
            print(f"  Sequential alignment: {align_ok} OK, {align_fail} failed")

            # Smooth the offsets
            if offset_data:
                ox_raw = np.array([d[1] for d in offset_data])
                oy_raw = np.array([d[2] for d in offset_data])
                t_kernel = np.ones(smooth_window) / smooth_window
                ox_smooth = np.convolve(ox_raw, t_kernel, mode="same")
                oy_smooth = np.convolve(oy_raw, t_kernel, mode="same")
                for k in range(half):
                    w = k + half + 1
                    ox_smooth[k] = np.mean(ox_raw[:w])
                    oy_smooth[k] = np.mean(oy_raw[:w])
                    ox_smooth[-(k + 1)] = np.mean(ox_raw[-w:])
                    oy_smooth[-(k + 1)] = np.mean(oy_raw[-w:])
                print(f"  Smoothed crop offsets (window={smooth_window})")

            # Pass 3: re-crop with smoothed offsets baked in — no warping
            print("Pass 3: Rendering with corrected crops...")
            for j, (crop_box, _, _, date, path) in enumerate(offset_data):
                try:
                    img = Image.open(path).convert("RGB")
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass

                    img = apply_crop(img, crop_box[0], crop_box[1],
                                     crop_box[2], crop_box[3],
                                     ox_smooth[j], oy_smooth[j])

                    if not args.no_date:
                        img = add_date_overlay(img, date)
                    frame_path = os.path.join(tmpdir, f"frame_{frame_count:06d}.jpg")
                    img.save(frame_path, "JPEG", quality=95)
                    frame_count += 1
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    continue
        else:
            print("Processing frames...")
            for i, (date, path) in enumerate(photos):
                try:
                    img = Image.open(path).convert("RGB")
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass

                    img = resize_and_crop(img)
                    if align:
                        frame_cv = pil_to_cv(img)
                        aligned = align_to_reference(frame_cv, ref_gray,
                                                     ref_kp, ref_desc, detector)
                        if aligned is None:
                            align_failures += 1
                        else:
                            img = cv_to_pil(aligned)

                    if not args.no_date:
                        img = add_date_overlay(img, date)
                    frame_path = os.path.join(tmpdir, f"frame_{frame_count:06d}.jpg")
                    img.save(frame_path, "JPEG", quality=95)
                    frame_count += 1
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    continue

                if (i + 1) % 50 == 0 or i == len(photos) - 1:
                    print(f"  {i + 1}/{len(photos)}")

            if align and align_failures:
                print(f"  Alignment failed on {align_failures}/{len(photos)} frames (used unaligned)")

        # Stitch with ffmpeg
        print("Encoding video...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(tmpdir, "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            args.output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr}")
            sys.exit(1)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done! {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
