#!/usr/bin/env python3
"""
Timelapse generator — turns a folder of photos into an MP4 video.

Usage:
    python3 timelapse.py /path/to/photos
    python3 timelapse.py /path/to/photos --output my_timelapse.mp4 --fps 12

Requires: Pillow, opencv-python, numpy, ffmpeg (must be on PATH)
    pip install Pillow opencv-python numpy
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


WIDTH = 1920
HEIGHT = 1080
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

# Tree-trunk detection parameters
WORK_WIDTH = 960
WORK_HEIGHT = 540
MIN_TRUNK_HEIGHT_FRAC = 0.25
MAX_TRUNK_WIDTH_FRAC = 0.35
TRUNK_BOTTOM_FRAC = 0.7
MIN_TRUNK_SEPARATION_FRAC = 0.2


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
    """Detect two tree trunks in an image.

    Returns [left_trunk, right_trunk] dicts with centroid, bbox, bottom_center,
    or None if two trunks cannot be reliably detected.
    """
    work = cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT))
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    # Find dark regions using two complementary methods
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=10
    )
    dark_mask = cv2.bitwise_and(otsu_mask, adaptive_mask)

    # Morphological ops to isolate vertical structures
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, close_kernel)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, erode_kernel)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    dark_mask = cv2.dilate(dark_mask, vert_kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < (WORK_WIDTH * WORK_HEIGHT * 0.005):
            continue
        if h < WORK_HEIGHT * MIN_TRUNK_HEIGHT_FRAC:
            continue
        if w > WORK_WIDTH * MAX_TRUNK_WIDTH_FRAC:
            continue
        if (y + h) < WORK_HEIGHT * TRUNK_BOTTOM_FRAC:
            continue
        if h / max(w, 1) < 1.0:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        candidates.append({
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "contour": contour,
            "area": area,
            "bottom_center": (x + w // 2, y + h),
        })

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda c: c["area"], reverse=True)

    best_pair = None
    for i in range(len(candidates)):
        for j in range(i + 1, min(len(candidates), 6)):
            ci, cj = candidates[i], candidates[j]
            if ci["centroid"][0] < cj["centroid"][0]:
                left, right = ci, cj
            else:
                left, right = cj, ci

            if left["centroid"][0] > WORK_WIDTH * 0.5:
                continue
            if right["centroid"][0] < WORK_WIDTH * 0.5:
                continue

            sep = right["centroid"][0] - left["centroid"][0]
            if sep < WORK_WIDTH * MIN_TRUNK_SEPARATION_FRAC:
                continue

            combined = left["area"] + right["area"]
            if best_pair is None or combined > best_pair[2]:
                best_pair = (left, right, combined)

    if best_pair is None:
        return None
    return [best_pair[0], best_pair[1]]


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


def crop_around_trunks(img: Image.Image, trunks: list[dict]) -> Image.Image:
    """Crop image so the two trunks are centered in the frame at WIDTHxHEIGHT."""
    orig_w, orig_h = img.size
    sx = orig_w / WORK_WIDTH
    sy = orig_h / WORK_HEIGHT

    left_cx = trunks[0]["centroid"][0] * sx
    right_cx = trunks[1]["centroid"][0] * sx
    mid_x = (left_cx + right_cx) / 2

    left_bot_y = trunks[0]["bottom_center"][1] * sy
    right_bot_y = trunks[1]["bottom_center"][1] * sy
    bottom_y = max(left_bot_y, right_bot_y)

    trunk_span = right_cx - left_cx
    crop_w = trunk_span / 0.65
    crop_h = crop_w * (HEIGHT / WIDTH)

    crop_w = min(crop_w, orig_w)
    crop_h = min(crop_h, orig_h)

    if crop_w / crop_h > WIDTH / HEIGHT:
        crop_h = crop_w * (HEIGHT / WIDTH)
    else:
        crop_w = crop_h * (WIDTH / HEIGHT)

    left = mid_x - crop_w / 2
    top = bottom_y - crop_h * 0.85

    left = max(0, min(left, orig_w - crop_w))
    top = max(0, min(top, orig_h - crop_h))

    img = img.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    return img


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
    """Save a debug image with detected trunks highlighted."""
    vis = cv2.resize(image, (WORK_WIDTH, WORK_HEIGHT))
    if trunks is not None:
        for i, t in enumerate(trunks):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            x, y, w, h = t["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cx, cy = int(t["centroid"][0]), int(t["centroid"][1])
            cv2.circle(vis, (cx, cy), 5, color, -1)
            bx, by = t["bottom_center"]
            cv2.circle(vis, (bx, by), 5, (255, 255, 0), -1)
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
    parser.add_argument("--gps-radius", type=int, default=50, help="GPS filter radius in meters (default: 50)")
    parser.add_argument("--no-align", action="store_true", help="Disable feature-based image alignment")
    parser.add_argument("--tree-detect", action="store_true", help="Detect two tree trunks for alignment and filtering")
    parser.add_argument("--tree-debug", action="store_true", help="Save debug images showing detected trunks")
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

    print(f"Found {len(photos)} photos from {photos[0][0].strftime('%Y-%m-%d')} to {photos[-1][0].strftime('%Y-%m-%d')}")
    duration = len(photos) / args.fps
    print(f"Output: {duration:.1f}s at {args.fps}fps")

    # Set up alignment
    tree_mode = args.tree_detect
    align = not args.no_align
    ref_trunks = None

    if tree_mode:
        print("Setting up tree-trunk detection...")
        result = pick_reference_with_trunks(photos)
        if result is None:
            print("Error: Could not find a reference frame with two detectable trunks.")
            print("Try running without --tree-detect or check your photos.")
            sys.exit(1)
        ref_cv, ref_trunks = result
        ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB_create(nfeatures=3000)
        ref_kp, ref_desc = detector.detectAndCompute(ref_gray, None)
        tree_discards = 0
        tree_align_failures = 0
        if args.tree_debug:
            debug_dir = os.path.join(os.path.dirname(args.output) or ".", "tree_debug")
            os.makedirs(debug_dir, exist_ok=True)
            print(f"  Debug images will be saved to {debug_dir}/")
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
        print("Processing frames...")
        for i, (date, path) in enumerate(photos):
            try:
                img = Image.open(path).convert("RGB")
                try:
                    from PIL import ImageOps
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass

                if tree_mode:
                    cv_img = pil_to_cv(img)
                    trunks = detect_trunks(cv_img)

                    if args.tree_debug:
                        debug_draw_trunks(cv_img, trunks, path,
                                          debug_dir)

                    if trunks is None:
                        tree_discards += 1
                        continue

                    img = crop_around_trunks(img, trunks)
                    frame_cv = pil_to_cv(img)
                    frame_trunks = detect_trunks(frame_cv)

                    if frame_trunks is not None:
                        M = compute_trunk_alignment(frame_trunks, ref_trunks,
                                                    (HEIGHT, WIDTH))
                        if M is not None:
                            aligned = align_frame_by_trunks(frame_cv, M)
                            img = cv_to_pil(aligned)
                        else:
                            tree_align_failures += 1
                    else:
                        tree_align_failures += 1
                        # Fall back to ORB alignment
                        aligned = align_to_reference(frame_cv, ref_gray,
                                                     ref_kp, ref_desc, detector)
                        if aligned is not None:
                            img = cv_to_pil(aligned)
                else:
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

        if tree_mode:
            print(f"  Discarded {tree_discards}/{len(photos)} photos (no trunks detected)")
            if tree_align_failures:
                print(f"  Trunk alignment failed on {tree_align_failures} frames (used fallback)")
        elif align and align_failures:
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
