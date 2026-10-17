"""MJPEG proxy to the simulation bridge, with detector boxes on the drone cameras.

The FloodUAV bridge (sim/flooduav/bridge.py) exposes:
  GET  /stream/follow     third-person camera following the selected drone
  GET  /stream/overview   scene overview camera
  GET  /stream/cam<i>     camera of drone i
  POST /view              {"drone": i} points the follow camera at a drone

This module turns those into the three endpoints the operator UI shows (plan 11.1):
  GET /video/follow.mjpg
  GET /video/overview.mjpg
  GET /video/camera/<i>.mjpg      the same as cam<i>, with the latest detector boxes drawn

Boxes come from Orchestrator.latest_detections(i) when the orchestrator keeps them
Nothing here is required for a mission to
run: without a bridge every endpoint answers 404 straight away.
"""

import asyncio
import contextlib
import logging
from typing import AsyncIterator

import httpx

log = logging.getLogger("sar.video")

BOUNDARY = "sarframe"


def stream_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Connection": "close",
        "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        "X-Accel-Buffering": "no",
    }


async def iter_mjpeg(url: str, client: httpx.AsyncClient | None = None) -> AsyncIterator[bytes]:
    """Yields a well-formed multipart MJPEG body from the bridge's own MJPEG stream.

    The bridge already emits multipart/x-mixed-replace frames; we copy its boundary bytes
    through unchanged so the browser sees a normal MJPEG stream.
    """
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
    try:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise httpx.HTTPStatusError("bridge stream refused", request=resp.request, response=resp)
            async for chunk in resp.aiter_bytes():          # as it arrives: a chunk size waits to fill it
                if chunk:
                    yield chunk
    finally:
        if own:
            await client.aclose()


async def iter_jpegs(url: str, client: httpx.AsyncClient | None = None) -> AsyncIterator[bytes]:
    """Yields raw JPEG frames from a multipart stream, skipping the boundary lines."""
    buf = b""
    async for chunk in iter_mjpeg(url, client):
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9")
            if start < 0 or end < 0 or end < start:
                break
            yield buf[start : end + 2]
            buf = buf[end + 2 :]


def part(jpg: bytes) -> bytes:
    """One multipart frame under our own boundary."""
    head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
    return head + jpg + b"\r\n"


async def frames(url: str) -> AsyncIterator[bytes]:
    """Re-frame a bridge MJPEG stream under BOUNDARY: the bridges use their own boundary
    string, and the response header (stream_headers) announces ours."""
    try:
        async for jpg in iter_jpegs(url):
            yield part(jpg)
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        log.info("stream %s ended: %r", url, exc)


def draw_boxes(jpeg: bytes, boxes: list, frame_w: int, frame_h: int) -> bytes:
    """Draw detector boxes (x0, y0, x1, y1, score) on a JPEG, scaled from full-frame pixels."""
    if not boxes:
        return jpeg
    try:
        import io

        from PIL import Image, ImageDraw
    except ImportError:  # Pillow is a hard dependency elsewhere, but never break the stream
        return jpeg
    try:
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception:
        return jpeg
    w, h = img.size
    if not frame_w or not frame_h:
        frame_w, frame_h = w, h
    sx, sy = w / float(frame_w), h / float(frame_h)
    draw = ImageDraw.Draw(img)
    for b in boxes:
        try:
            x0, y0, x1, y1 = (float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy)
            score = float(b[4]) if len(b) > 4 else None
        except (TypeError, ValueError, IndexError):
            continue
        color = (255, 220, 60) if (score is None or score >= 0.8) else (255, 140, 40)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, w // 320))
        if score is not None:
            label = f"{score:.2f}"
            draw.text((x0 + 3, max(0, y0 - 12)), label, fill=color)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


async def set_view(bridge: str, drone: int, client: httpx.AsyncClient) -> None:
    """Tell the bridge which drone the follow camera tracks. Failures are logged, not raised."""
    try:
        await client.post(f"{bridge}/view", json={"drone": int(drone)}, timeout=3.0)
    except Exception as exc:
        log.debug("bridge /view failed: %r", exc)


async def frames_with_boxes(bridge: str, drone: int, orch, client: httpx.AsyncClient | None = None) -> AsyncIterator[bytes]:
    """Stream the drone camera as a fresh multipart body, drawing boxes on each frame."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
    try:
        async for jpg in iter_jpegs(f"{bridge}/stream/cam{int(drone)}", client):
            latest = None
            getter = getattr(orch, "latest_detections", None)
            if callable(getter):
                with contextlib.suppress(Exception):
                    latest = getter(int(drone))
            if isinstance(latest, dict):
                boxes = latest.get("boxes") or []
                jpg = draw_boxes(jpg, boxes, int(latest.get("w") or 0), int(latest.get("h") or 0))
            elif isinstance(latest, (list, tuple)) and latest:
                jpg = draw_boxes(jpg, list(latest), 0, 0)
            yield part(jpg)
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        log.info("camera stream for drone %s ended: %r", drone, exc)
    finally:
        if own:
            await client.aclose()