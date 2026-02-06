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
    align = not args.no_align
    if align:
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
                img = resize_and_crop(img)

                if align:
                    frame_cv = pil_to_cv(img)
                    aligned = align_to_reference(frame_cv, ref_gray, ref_kp, ref_desc, detector)
                    if aligned is None:
                        align_failures += 1
                        # Use unaligned frame rather than skipping
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
