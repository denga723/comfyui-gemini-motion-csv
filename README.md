# ComfyUI Gemini Motion CSV

A visible, modular ComfyUI pipeline for splitting a video into timed clips, asking Gemini to describe the motion in each clip, and exporting the captions as CSV.

The included workflow keeps every stage on the canvas:

1. Video path and Gemini authentication
2. Exact 1.5-second MP4 segmentation
3. Segment manifest with timestamps, durations, and temporary file paths
4. One labeled thumbnail per segment
5. Independent Gemini motion analysis for every segment
6. Timestamped caption and JSON previews
7. Final CSV path and exact CSV-row preview

## Output format

The exporter writes one row per source video:

```csv
clip/media_id,video_path,video_caption
```

`video_caption` contains JSON shaped like:

```json
{
  "segment_captions": {
    "00:00.000": "...",
    "00:01.500": "..."
  }
}
```

The final partial segment is included. For example, an 8-second video produces five 1.5-second clips and one 0.5-second clip.

## Requirements

- ComfyUI
- Python packages in `requirements.txt`
- `ffmpeg` available on `PATH`, with H.264 encoding support
- Gemini access through either:
  - Google Cloud Application Default Credentials (Vertex AI), or
  - a Gemini API key from Google AI Studio

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/denga723/comfyui-gemini-motion-csv.git
cd comfyui-gemini-motion-csv
python -m pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Authentication

### Vertex AI / Application Default Credentials

In the `Gemini Motion Authentication` node:

- Set `project_id` to your Google Cloud project.
- Set `location` to a supported Gemini region, usually `global`.
- Leave `api_key` empty.
- Optionally provide a service-account JSON path. Otherwise the node uses Application Default Credentials.

### Google AI Studio API key

- Enter the key in `api_key`.
- `project_id` is not required on this path.

Avoid committing workflows that contain API keys or credential paths.

## Workflow

Open:

```text
examples/Gemini_1.5s_Video_Motion_to_CSV.json
```

Set the video path in `1. Split Video into Timed Segments`, configure authentication, then queue the workflow.

The default motion prompt asks Gemini to describe subject movement, direction and speed, camera movement, framing changes, and transitions. Each short clip is sent inline as MP4. Video sampling is increased above Gemini's default and capped at 12 FPS for more granular motion analysis.

## Nodes

### Gemini Motion Authentication

Creates an authentication configuration without making a network request.

### 1. Split Video into Timed Segments

Uses `ffmpeg` to create exact clips in ComfyUI's temporary directory. Outputs the segment bundle, timestamped thumbnails, and a readable manifest.

### 2. Analyze Segment Motion with Gemini

Sends up to four segments concurrently. Retries transient rate-limit and server errors, then assembles timestamped captions and the final `video_caption` JSON.

### 3. Export Motion Captions to CSV

Writes a UTF-8 CSV into the ComfyUI output directory and exposes the saved path plus the exact serialized row for preview.

## Privacy

Video segments are sent to the configured Google Gemini endpoint when the analysis node runs. Segments and thumbnails remain in ComfyUI's temporary directory; CSV files are written to the normal ComfyUI output directory.

## License

MIT
