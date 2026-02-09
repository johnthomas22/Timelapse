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


def detect_foreground_trunks(image: np.ndarray, wall_y: float) -> tuple[float, float] | None:
    """Detect the two foreground tree trunks (J and K) at the wall line.

    Uses zone-based detection: J is expected at 58-65% and K at 65-72%
    of frame width. Finds the most prominent dark dip in each zone.

    Returns (left_trunk_x, right_trunk_x) in WORK_WIDTH coords, or None.
    """
    from scipy.signal import find_peaks

    work = cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Horizontal strip around the wall
    strip_half = 20
    wy = int(wall_y)
    wall_strip = gray[max(0, wy - strip_half):wy + strip_half, :]

    # Column brightness profile
    col_brightness = np.mean(wall_strip, axis=0).astype(np.float64)
    k = np.ones(5) / 5
    col_smooth = np.convolve(col_brightness, k, mode="same")

    # Search the full foreground region for all dark dips
    fg_start = int(WORK_WIDTH * 0.45)
    fg_end = int(WORK_WIDTH * 0.75)
    fg_region = col_smooth[fg_start:fg_end]

    inverted = fg_region.max() - fg_region
    peaks, props = find_peaks(inverted, height=5, distance=10, prominence=5)

    if len(peaks) == 0:
        return None

    # Convert peaks to absolute x positions
    abs_peaks = fg_start + peaks
    proms = props["prominences"]

    # Zone-based detection: find best peak in each zone
    # J zone: 58-65% of frame width
    j_lo = int(WORK_WIDTH * 0.58)
    j_hi = int(WORK_WIDTH * 0.65)
    # K zone: 65-72% of frame width
    k_lo = int(WORK_WIDTH * 0.65)
    k_hi = int(WORK_WIDTH * 0.72)

    j_best = None
    j_prom = 0
    k_best = None
    k_prom = 0

    for pi, (ax, pr) in enumerate(zip(abs_peaks, proms)):
        if j_lo <= ax <= j_hi and pr > j_prom:
            j_best = float(ax)
            j_prom = pr
        if k_lo <= ax <= k_hi and pr > k_prom:
            k_best = float(ax)
            k_prom = pr

    if j_best is None or k_best is None:
        return None

    # Both should have reasonable prominence (actual tree trunks, not noise)
    if j_prom < 8 or k_prom < 8:
        return None

    return (j_best, k_best)


def warp_to_landmarks(image: np.ndarray, left_edge: float, right_edge: float,
                      wall_y: float,
                      output_size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    """Warp image using full affine transform so detected landmarks map to
    fixed output positions.

    Uses 3 control points (left trunk at wall, right trunk at wall, midpoint
    above wall) to correct scale, translation, AND rotation — keeping the
    wall horizontal and both trunks at fixed positions every frame.
    """
    out_w, out_h = output_size
    h, w = image.shape[:2]
    sx = w / WORK_WIDTH
    sy = h / WORK_HEIGHT

    # Source points in original image coords
    wall_y_orig = wall_y * sy
    left_orig = left_edge * sx
    right_orig = right_edge * sx
    mid_x = (left_orig + right_orig) / 2

    # 3rd point above midpoint — vertical offset proportional to trunk span
    vert_offset = (right_orig - left_orig) * 0.5
    src_pts = np.float32([
        [left_orig, wall_y_orig],
        [right_orig, wall_y_orig],
        [mid_x, wall_y_orig - vert_offset],
    ])

    # Destination points at fixed output positions
    dst_left_x = TARGET_LEFT_EDGE * out_w
    dst_right_x = TARGET_RIGHT_EDGE * out_w
    dst_wall_y = TARGET_WALL_Y * out_h
    dst_mid_x = (dst_left_x + dst_right_x) / 2
    dst_vert_offset = (dst_right_x - dst_left_x) * 0.5

    dst_pts = np.float32([
        [dst_left_x, dst_wall_y],
        [dst_right_x, dst_wall_y],
        [dst_mid_x, dst_wall_y - dst_vert_offset],
    ])

    M = cv2.getAffineTransform(src_pts, dst_pts)

    return cv2.warpAffine(image, M, output_size,
                          flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


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


# Target positions for landmarks in output frame (fraction of WIDTH/HEIGHT)
# Big tree trunks at edges, wall as horizontal anchor
TARGET_LEFT_EDGE = 0.05   # left trunk inner edge at 5% from left
TARGET_RIGHT_EDGE = 0.95  # right trunk inner edge at 95% from left
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
        print("Setting up feature-based alignment...")
        center_mask = make_center_mask(WIDTH, HEIGHT)

        if not HAS_LIGHTGLUE:
            print("  Error: LightGlue is required for --tree-detect mode")
            sys.exit(1)

        device_name = "GPU" if DEVICE.type == "cuda" else "CPU"
        print(f"  LightGlue ({device_name})")
        lg_extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
        lg_matcher = LightGlue(features="superpoint").eval().to(DEVICE)

        # Reference frame (middle of sequence)
        ref_idx = len(photos) // 2
        ref_date, ref_path = photos[ref_idx]
        print(f"  Reference: {os.path.basename(ref_path)} ({ref_date.strftime('%Y-%m-%d')})")
        ref_pil = Image.open(ref_path).convert("RGB")
        try:
            from PIL import ImageOps
            ref_pil = ImageOps.exif_transpose(ref_pil)
        except Exception:
            pass
        ref_pil = resize_and_crop(ref_pil)
        ref_cv = pil_to_cv(ref_pil)
        ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY)
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
            # Align each frame to reference using LightGlue feature matching
            # Process outward from reference in both directions to minimize chain drift
            print("Aligning and rendering frames...")
            transforms = [None] * len(photos)
            transforms[ref_idx] = np.eye(2, 3, dtype=np.float64)

            def _chain_affine(M_prev, M_seq):
                """Chain affine transforms: M_seq (curr→prev) then M_prev (prev→ref)."""
                A1 = np.vstack([M_seq, [0, 0, 1]])
                A2 = np.vstack([M_prev, [0, 0, 1]])
                return (A2 @ A1)[:2, :]

            def _load_gray(path):
                img = Image.open(path).convert("RGB")
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                img = resize_and_crop(img)
                return cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2GRAY)

            direct_ok = 0
            chain_ok = 0
            failures = 0

            # Forward from reference (ref+1, ref+2, ...)
            prev_gray = ref_gray
            prev_M = transforms[ref_idx]
            for i in range(ref_idx + 1, len(photos)):
                date, path = photos[i]
                try:
                    frame_gray = _load_gray(path)
                    M = _lightglue_match(ref_gray, frame_gray,
                                         lg_extractor, lg_matcher, center_mask)
                    if M is not None:
                        transforms[i] = M
                        direct_ok += 1
                    else:
                        M_seq = _lightglue_match(prev_gray, frame_gray,
                                                 lg_extractor, lg_matcher, center_mask)
                        if M_seq is not None:
                            transforms[i] = _chain_affine(prev_M, M_seq)
                            chain_ok += 1
                        else:
                            failures += 1
                    if transforms[i] is not None:
                        prev_gray = frame_gray
                        prev_M = transforms[i]
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    failures += 1
                if (i - ref_idx) % 50 == 0:
                    print(f"  Forward: {i - ref_idx}/{len(photos) - ref_idx - 1}")

            # Backward from reference (ref-1, ref-2, ...)
            prev_gray = ref_gray
            prev_M = transforms[ref_idx]
            for i in range(ref_idx - 1, -1, -1):
                date, path = photos[i]
                try:
                    frame_gray = _load_gray(path)
                    M = _lightglue_match(ref_gray, frame_gray,
                                         lg_extractor, lg_matcher, center_mask)
                    if M is not None:
                        transforms[i] = M
                        direct_ok += 1
                    else:
                        M_seq = _lightglue_match(prev_gray, frame_gray,
                                                 lg_extractor, lg_matcher, center_mask)
                        if M_seq is not None:
                            transforms[i] = _chain_affine(prev_M, M_seq)
                            chain_ok += 1
                        else:
                            failures += 1
                    if transforms[i] is not None:
                        prev_gray = frame_gray
                        prev_M = transforms[i]
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    failures += 1
                if (ref_idx - i) % 50 == 0:
                    print(f"  Backward: {ref_idx - i}/{ref_idx}")

            print(f"  Alignment: {direct_ok} direct, {chain_ok} chained, {failures} failed")

            # Render aligned frames
            print("Rendering frames...")
            for i, (date, path) in enumerate(photos):
                M = transforms[i]
                if M is None:
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    img = resize_and_crop(img)
                    frame_cv = pil_to_cv(img)
                    aligned = cv2.warpAffine(frame_cv, M, (WIDTH, HEIGHT),
                                             flags=cv2.INTER_LANCZOS4,
                                             borderMode=cv2.BORDER_REFLECT)
                    img = cv_to_pil(aligned)

                    if not args.no_date:
                        img = add_date_overlay(img, date)
                    frame_path = os.path.join(tmpdir, f"frame_{frame_count:06d}.jpg")
                    img.save(frame_path, "JPEG", quality=95)
                    frame_count += 1
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    continue

                if (i + 1) % 100 == 0 or i == len(photos) - 1:
                    print(f"  {i + 1}/{len(photos)}")

            print(f"  Rendered {frame_count} frames ({failures} skipped)")
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
