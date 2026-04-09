import json
import logging
import time
from typing import Optional, Dict, Any

import httpx


logger = logging.getLogger(__name__)


class HeyGenClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("HEYGEN_API_KEY is missing. Set HEYGEN_APIKEY in .env")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-KEY": self.api_key,
        }

    def upload_audio_asset(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
        """Uploads raw audio bytes to HeyGen and returns audio_asset_id."""
        url = "https://upload.heygen.com/v1/asset"
        headers = {
            **self._headers(),
            "Content-Type": mime_type,
        }

        resp = httpx.post(url, headers=headers, content=audio_bytes, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        # Common patterns: {"data": {"id": "..."}} or {"data": {"asset_id": "..."}}
        data = payload.get("data") or {}
        asset_id = data.get("id") or data.get("asset_id")
        if not asset_id:
            raise RuntimeError(f"HeyGen upload response missing asset id: {json.dumps(payload)[:1000]}")
        return asset_id

    def create_avatar_video_v2(
        self,
        *,
        avatar_id: str,
        avatar_pose_id: str,
        audio_asset_id: Optional[str] = None,
        input_text: Optional[str] = None,
        voice_id: Optional[str] = None,
        avatar_style: str = "normal",
        title: Optional[str] = None,
        dimension: Optional[Dict[str, int]] = None,
    ) -> str:
        """Starts a video render job and returns video_id."""
        # HeyGen API: Create a WebM Video
        # This endpoint is broadly available and supports text+voice or uploaded audio.
        # Docs: https://docs.heygen.com/reference/create-a-webm-video
        url = "https://api.heygen.com/v1/video.webm"
        headers = {
            **self._headers(),
            "Content-Type": "application/json",
        }

        if not avatar_pose_id:
            raise ValueError("avatar_pose_id is required for /v1/video.webm endpoint")

        if audio_asset_id and (input_text or voice_id):
            raise ValueError("Provide either audio_asset_id OR (input_text + voice_id), not both")

        body: Dict[str, Any] = {
            "avatar_id": avatar_id,
            "avatar_style": avatar_style,
            "avatar_pose_id": avatar_pose_id,
        }

        if audio_asset_id:
            body["input_audio"] = audio_asset_id
        else:
            if not input_text or not voice_id:
                raise ValueError("Missing voice input: provide audio_asset_id or (input_text and voice_id)")
            body["input_text"] = input_text
            body["voice_id"] = voice_id

        # title is not a documented field for this endpoint; ignore if provided
        if dimension:
            body["dimension"] = dimension

        resp = httpx.post(url, headers=headers, json=body, timeout=60)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HeyGen create video failed ({resp.status_code}): {resp.text[:2000]}"
            ) from e
        payload = resp.json()
        data = payload.get("data") or {}
        video_id = data.get("video_id") or data.get("id")
        if not video_id:
            raise RuntimeError(f"HeyGen create video response missing video_id: {json.dumps(payload)[:1000]}")
        return video_id

    def get_video_status(self, video_id: str) -> Dict[str, Any]:
        url = "https://api.heygen.com/v1/video_status.get"
        headers = {
            **self._headers(),
        }
        resp = httpx.get(url, headers=headers, params={"video_id": video_id}, timeout=60)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HeyGen video status failed ({resp.status_code}): {resp.text[:2000]}"
            ) from e
        payload = resp.json()
        data = payload.get("data") or {}
        return data

    def wait_for_video_url(
        self,
        video_id: str,
        *,
        timeout_s: int = 180,
        poll_interval_s: float = 2.0,
    ) -> str:
        """Polls HeyGen until completed/failed/timeout. Returns video_url."""
        deadline = time.time() + timeout_s
        last_status = None

        while time.time() < deadline:
            data = self.get_video_status(video_id)
            status = data.get("status")
            if status != last_status:
                logger.info(f"HeyGen video {video_id} status: {status}")
                last_status = status

            if status == "completed":
                video_url = data.get("video_url")
                if not video_url:
                    raise RuntimeError(f"HeyGen completed but missing video_url: {json.dumps(data)[:1000]}")
                return video_url

            if status == "failed":
                raise RuntimeError(f"HeyGen video generation failed: {json.dumps(data)[:1000]}")

            time.sleep(poll_interval_s)

        raise TimeoutError(f"Timed out waiting for HeyGen video {video_id}")
