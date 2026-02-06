# Timelapse

Turns a folder of photos into an MP4 timelapse video.

Features:
- Automatic sorting by EXIF date (falls back to file modification time)
- GPS-based filtering to exclude photos taken at different locations
- Feature-based image alignment to stabilise the output
- Date overlay on each frame
- EXIF orientation handling

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH
- Python packages: `pip install Pillow opencv-python numpy`

## Usage

```
python3 timelapse.py /path/to/photos
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o`, `--output` | Output file path | `timelapse.mp4` |
| `--fps` | Frames per second | `10` |
| `--no-date` | Disable date overlay | |
| `--no-gps-filter` | Disable GPS location filtering | |
| `--gps-radius` | GPS filter radius in meters | `50` |
| `--no-align` | Disable image alignment | |

### Examples

```
python3 timelapse.py ~/garden-photos --output garden.mp4 --fps 12
python3 timelapse.py ./photos --no-align --no-date
python3 timelapse.py ./photos --gps-radius 100
```
