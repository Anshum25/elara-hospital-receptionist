"""D-ID Live Streaming Client for real-time avatar video."""
import json
import httpx
import base64
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class DIDClient:
    """D-ID API client for live streaming avatar with WebRTC."""

    def __init__(self, api_key: str, base_url: str = "https://api.d-id.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.stream_id: Optional[str] = None
        self.ice_servers: list = []
        self.sdp_offer: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        # D-ID API key format: "email:secret" - needs base64 encoding for Basic auth
        if ':' in self.api_key:
            encoded = base64.b64encode(self.api_key.encode()).decode()
        else:
            encoded = self.api_key

        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    def close_all_sessions(self) -> bool:
        """Close all existing streaming sessions to free up limit."""
        try:
            url = f"{self.base_url}/talks/streams"
            resp = httpx.get(url, headers=self._headers(), timeout=30)
            if resp.status_code == 200:
                sessions = resp.json()
                if isinstance(sessions, list):
                    for session in sessions:
                        session_id = session.get("id")
                        if session_id:
                            delete_url = f"{self.base_url}/talks/streams/{session_id}"
                            try:
                                httpx.delete(delete_url, headers=self._headers(), timeout=30)
                                logger.info(f"Closed existing D-ID session: {session_id}")
                            except Exception:
                                pass
            return True
        except Exception as e:
            logger.warning(f"Could not close existing sessions: {e}")
            return False

    def create_stream(self, source_url: str) -> Dict[str, Any]:
        """Create a new streaming session and return WebRTC config."""
        self.close_all_sessions()
        url = f"{self.base_url}/talks/streams"
        body = {"source_url": source_url}

        logger.info(f"Creating D-ID stream with source_url: {source_url[:50]}...")
        headers = self._headers()
        resp = httpx.post(url, headers=headers, json=body, timeout=60)

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_text = resp.text[:1000]
            logger.error(f"D-ID create stream failed ({resp.status_code}): {error_text}")
            raise RuntimeError(f"D-ID create stream failed ({resp.status_code}): {error_text}") from e

        data = resp.json()
        self.stream_id = data.get("id")
        # Handle both snake_case and camelCase from D-ID
        self.session_id = data.get("session_id") or data.get("sessionId")
        self.ice_servers = data.get("ice_servers", [])
        self.sdp_offer = data.get("offer", {}).get("sdp") if data.get("offer") else data.get("sdp")

        logger.info(f"✅ D-ID stream created: stream_id={self.stream_id}, session_id={self.session_id}")
        return {
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "sdp_offer": self.sdp_offer,
            "ice_servers": self.ice_servers,
        }

    def submit_sdp_answer(self, sdp_answer: str) -> bool:
        """Submit the WebRTC SDP answer to complete handshake."""
        if not self.session_id or not self.stream_id:
            raise RuntimeError("No active stream. Call create_stream() first.")

        url = f"{self.base_url}/talks/streams/{self.stream_id}/sdp"
        # D-ID expects the answer to be a JSEP object
        body = {
            "session_id": self.session_id, 
            "answer": {
                "type": "answer",
                "sdp": sdp_answer
            }
        }

        resp = httpx.post(url, headers=self._headers(), json=body, timeout=30)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ D-ID SDP submit failed ({resp.status_code}): {resp.text}")
            raise
        logger.info("✅ D-ID WebRTC handshake complete")
        return True

    def submit_ice_candidate(self, candidate: str, sdp_mid: str, sdp_mline_index: int) -> bool:
        """Submit ICE candidate for NAT traversal."""
        if not self.session_id or not self.stream_id:
            raise RuntimeError("No active stream. Call create_stream() first.")

        url = f"{self.base_url}/talks/streams/{self.stream_id}/ice"
        body = {
            "session_id": self.session_id,
            "candidate": candidate,
            "sdpMid": sdp_mid,
            "sdpMLineIndex": sdp_mline_index
        }

        resp = httpx.post(url, headers=self._headers(), json=body, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"D-ID ICE candidate submit failed: {e}")
            return False
        return True

    def send_text(self, text: str) -> bool:
        """Send text to be spoken by the avatar in real-time."""
        if not self.session_id or not self.stream_id:
            raise RuntimeError("No active stream. Call create_stream() first.")

        url = f"{self.base_url}/talks/streams/{self.stream_id}"
        body = {
            "script": {
                "type": "text",
                "input": text,
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-JennyNeural"
                }
            },
            "session_id": self.session_id,
            "config": {"stitch": True, "fluent": True}
        }

        resp = httpx.post(url, headers=self._headers(), json=body, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"D-ID send text failed: {e}")
            return False

        logger.info(f"🎙️ Sent text to D-ID: {text[:50]}...")
        return True

    def delete_stream(self) -> bool:
        """Close and cleanup the streaming session."""
        if not self.session_id or not self.stream_id:
            return True
        
        url = f"{self.base_url}/talks/streams/{self.stream_id}"
        try:
            httpx.delete(url, headers=self._headers(), timeout=30)
            logger.info(f"🗑️ Deleted D-ID stream session: {self.session_id}")
            self.session_id = None
            self.stream_id = None
            return True
        except Exception:
            return False
