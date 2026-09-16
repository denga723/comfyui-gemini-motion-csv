import asyncio
import base64
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import aiohttp
import cv2
import folder_paths
import google.auth
import numpy as np
import torch
from google.auth.transport.requests import Request


DEFAULT_MOTION_PROMPT = (
    "Describe only the visible motion in this video segment. Focus on subject and object actions, "
    "direction and speed of movement, camera movement, changes in framing, and any visible transition. "
    "Mention visual style or setting only when it helps identify the moving elements. Do not infer events "
    "outside this clip. Return one concise English paragraph of 1-3 sentences with no timestamp, label, "
    "bullet, markdown, or JSON."
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _format_timestamp(seconds):
    total_ms = max(0, round(seconds * 1000))
    minutes, total_ms = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(total_ms, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _resolve_video_path(video_path):
    source_path = Path(str(video_path).strip().strip('"')).expanduser().resolve()
    if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError(f"Unsupported video extension: {source_path.suffix}")
    if not source_path.is_file():
        raise RuntimeError(f"Video does not exist: {source_path}")
    return source_path


def _probe_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not determine video duration: {video_path}")
    return frame_count / fps, fps, width, height


def _split_video(video_path, segment_dir, duration, interval_sec):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    segments = []
    for index in range(int(math.ceil(duration / interval_sec))):
        start = index * interval_sec
        segment_duration = min(interval_sec, duration - start)
        if segment_duration <= 0.02:
            continue
        segment_path = segment_dir / f"segment_{index:04d}_{round(start * 1000):09d}ms.mp4"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(video_path),
            "-t",
            f"{segment_duration:.6f}",
            "-an",
            "-vf",
            "scale=1280:1280:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(segment_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"Failed to create segment {_format_timestamp(start)}: {detail}")
        segments.append(
            {
                "index": index,
                "start": start,
                "end": start + segment_duration,
                "duration": segment_duration,
                "timestamp": _format_timestamp(start),
                "path": segment_path,
            }
        )
    return segments


def _segment_thumbnail(segment):
    capture = cv2.VideoCapture(str(segment["path"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(segment["duration"] * fps * 0.5)))
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not create preview for segment {segment['timestamp']}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    label = f"{segment['timestamp']} - {_format_timestamp(segment['end'])}"
    cv2.rectangle(frame, (12, 12), (372, 58), (0, 0, 0), -1)
    cv2.putText(frame, label, (25, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return torch.from_numpy(frame.astype(np.float32) / 255.0)


def _clean_caption(text):
    caption = text.strip()
    if caption.startswith("```") and caption.endswith("```"):
        caption = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", caption, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(caption)
        if isinstance(parsed, str):
            caption = parsed
        elif isinstance(parsed, dict):
            for key in ("caption", "motion_caption", "description", "text"):
                if isinstance(parsed.get(key), str):
                    caption = parsed[key]
                    break
    except json.JSONDecodeError:
        pass
    return " ".join(caption.strip().strip('"').split())


def _extract_response_text(response):
    parts = []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text"):
                parts.append(part["text"])
    return _clean_caption("\n".join(parts))


def _resolve_auth(auth_config):
    auth_config = auth_config or {}
    service_account_path = str(auth_config.get("service_account_json_path", "")).strip()
    if service_account_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

    api_key = str(auth_config.get("api_key", "")).strip()
    project_id = str(auth_config.get("project_id", "")).strip()
    location = str(auth_config.get("location", "global")).strip() or "global"
    if api_key:
        return {
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "api_key": api_key,
            "token": None,
        }

    credentials, default_project = google.auth.default()
    credentials.refresh(Request())
    project_id = project_id or default_project
    if not project_id:
        raise RuntimeError("Gemini authentication did not provide a Google Cloud project ID")
    if location in {"us", "eu"}:
        hostname = f"aiplatform.{location}.rep.googleapis.com"
    elif location == "global":
        hostname = "aiplatform.googleapis.com"
    else:
        hostname = f"{location}-aiplatform.googleapis.com"
    return {
        "endpoint": f"https://{hostname}/v1beta1/projects/{project_id}/locations/{location}/publishers/google/models/{{model}}:generateContent",
        "api_key": None,
        "token": credentials.token,
    }


async def _request_caption(session, semaphore, auth, model_name, segment, prompt, video_fps):
    video_data = base64.b64encode(segment["path"].read_bytes()).decode("ascii")
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {"mimeType": "video/mp4", "data": video_data},
                        "videoMetadata": {"fps": min(12.0, max(4.0, video_fps / 2.0))},
                    },
                    {
                        "text": (
                            f"This clip covers source-video time {segment['timestamp']} through "
                            f"{_format_timestamp(segment['end'])}. {prompt}"
                        )
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
    }
    headers = {"Content-Type": "application/json"}
    endpoint = auth["endpoint"].format(model=model_name)
    if auth["api_key"]:
        headers["x-goog-api-key"] = auth["api_key"]
    else:
        headers["Authorization"] = f"Bearer {auth['token']}"

    async with semaphore:
        for attempt in range(4):
            async with session.post(endpoint, headers=headers, json=body) as response:
                response_text = await response.text()
                if response.status == 200:
                    caption = _extract_response_text(json.loads(response_text))
                    if not caption:
                        raise RuntimeError(f"Gemini returned no caption for {segment['timestamp']}")
                    return caption
                if response.status not in RETRYABLE_STATUSES or attempt == 3:
                    raise RuntimeError(
                        f"Gemini API error for {segment['timestamp']} (HTTP {response.status}): {response_text[:1200]}"
                    )
            await asyncio.sleep(2 ** attempt)


async def _analyze_segments(auth, model_name, segments, prompt, video_fps):
    timeout = aiohttp.ClientTimeout(total=240)
    connector = aiohttp.TCPConnector(force_close=True)
    semaphore = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            _request_caption(session, semaphore, auth, model_name, segment, prompt, video_fps)
            for segment in segments
        ]
        return await asyncio.gather(*tasks)


def _run_analysis(auth, model_name, segments, prompt, video_fps):
    result = []
    error = []

    def run():
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.extend(loop.run_until_complete(_analyze_segments(auth, model_name, segments, prompt, video_fps)))
        except Exception as exc:
            error.append(exc)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    if error:
        raise error[0]
    return result


class GeminiMotionAuthConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_id": ("STRING", {"default": "", "multiline": False}),
                "location": (["global", "us-central1", "us", "eu", "us-east1", "us-west1", "europe-west1"],),
                "service_account_json_path": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("GEMINI_MOTION_AUTH",)
    RETURN_NAMES = ("auth_config",)
    FUNCTION = "configure"
    CATEGORY = "Gemini Motion CSV/Config"

    def configure(self, project_id, location, service_account_json_path, api_key):
        return (
            {
                "project_id": project_id,
                "location": location,
                "service_account_json_path": service_account_json_path,
                "api_key": api_key,
            },
        )


class GeminiVideoMotionSegmenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "interval_sec": ("FLOAT", {"default": 1.5, "min": 0.25, "max": 30.0, "step": 0.25}),
            }
        }

    RETURN_TYPES = ("GEMINI_VIDEO_SEGMENTS", "IMAGE", "STRING")
    RETURN_NAMES = ("segments", "segment_thumbnails", "segment_manifest")
    FUNCTION = "segment"
    CATEGORY = "Gemini Motion CSV/Pipeline"

    def segment(self, video_path, interval_sec):
        source_path = _resolve_video_path(video_path)
        duration, source_fps, width, height = _probe_video(source_path)
        run_dir = Path(folder_paths.get_temp_directory()) / "gemini_motion_segments" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        segments = _split_video(source_path, run_dir, duration, float(interval_sec))
        thumbnails = torch.stack([_segment_thumbnail(segment) for segment in segments], dim=0)
        bundle = {
            "source_path": source_path,
            "duration": duration,
            "source_fps": source_fps,
            "width": width,
            "height": height,
            "interval_sec": float(interval_sec),
            "segments": segments,
        }
        manifest_lines = [
            f"SOURCE VIDEO\n{source_path}",
            f"{duration:.3f}s | {source_fps:.3f} fps | {width}x{height}",
            f"\nSEGMENT PLAN ({len(segments)} clips)",
        ]
        for segment in segments:
            manifest_lines.append(
                f"[{segment['index'] + 1}] {segment['timestamp']} - {_format_timestamp(segment['end'])} "
                f"({segment['duration']:.3f}s)\n{segment['path']}"
            )
        return bundle, thumbnails, "\n\n".join(manifest_lines)


class GeminiSegmentMotionAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("GEMINI_VIDEO_SEGMENTS",),
                "auth_config": ("GEMINI_MOTION_AUTH",),
                "model_name": (["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"], {"default": "gemini-2.5-flash"}),
                "motion_prompt": ("STRING", {"default": DEFAULT_MOTION_PROMPT, "multiline": True}),
            }
        }

    RETURN_TYPES = ("GEMINI_MOTION_ANALYSIS", "STRING", "STRING")
    RETURN_NAMES = ("analysis", "analysis_preview", "video_caption_json")
    FUNCTION = "analyze_segments"
    CATEGORY = "Gemini Motion CSV/Pipeline"

    def analyze_segments(self, segments, auth_config, model_name, motion_prompt):
        auth = _resolve_auth(auth_config)
        captions = _run_analysis(auth, str(model_name), segments["segments"], str(motion_prompt).strip(), segments["source_fps"])
        segment_captions = {
            segment["timestamp"]: caption for segment, caption in zip(segments["segments"], captions)
        }
        caption_json = json.dumps({"segment_captions": segment_captions}, ensure_ascii=False, indent=2)
        analysis = {**segments, "model_name": str(model_name), "captions": captions, "caption_json": caption_json}
        report_lines = [
            f"Video: {segments['source_path']}",
            f"Source: {segments['duration']:.3f}s, {segments['source_fps']:.3f} fps, {segments['width']}x{segments['height']}",
            f"Segments: {len(segments['segments'])} at {segments['interval_sec']:.3f}s intervals",
            f"Model: {model_name}",
            "",
        ]
        for segment, caption in zip(segments["segments"], captions):
            report_lines.extend([f"{segment['timestamp']} - {_format_timestamp(segment['end'])}", caption, ""])
        return analysis, "\n".join(report_lines), caption_json


class GeminiMotionCSVExporter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis": ("GEMINI_MOTION_ANALYSIS",),
                "output_prefix": ("STRING", {"default": "gemini_motion", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("csv_path", "csv_row_preview")
    FUNCTION = "export"
    CATEGORY = "Gemini Motion CSV/Pipeline"
    OUTPUT_NODE = True

    def export(self, analysis, output_prefix):
        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(output_prefix).strip()).strip("._") or "gemini_motion"
        source_path = analysis["source_path"]
        output_path = Path(folder_paths.get_output_directory()) / f"{safe_prefix}_{source_path.stem}_captions.csv"
        row = {
            "clip/media_id": source_path.stem,
            "video_path": str(source_path),
            "video_caption": analysis["caption_json"],
        }
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clip/media_id", "video_path", "video_caption"])
            writer.writeheader()
            writer.writerow(row)

        preview_buffer = io.StringIO(newline="")
        preview_writer = csv.DictWriter(preview_buffer, fieldnames=["clip/media_id", "video_path", "video_caption"])
        preview_writer.writeheader()
        preview_writer.writerow(row)
        message = f"CSV written successfully:\n{output_path}"
        return {"ui": {"text": [message]}, "result": (str(output_path), preview_buffer.getvalue())}


NODE_CLASS_MAPPINGS = {
    "GeminiMotionAuthConfig": GeminiMotionAuthConfig,
    "GeminiVideoMotionSegmenter": GeminiVideoMotionSegmenter,
    "GeminiSegmentMotionAnalyzer": GeminiSegmentMotionAnalyzer,
    "GeminiMotionCSVExporter": GeminiMotionCSVExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiMotionAuthConfig": "Gemini Motion Authentication",
    "GeminiVideoMotionSegmenter": "1. Split Video into Timed Segments",
    "GeminiSegmentMotionAnalyzer": "2. Analyze Segment Motion with Gemini",
    "GeminiMotionCSVExporter": "3. Export Motion Captions to CSV",
}
