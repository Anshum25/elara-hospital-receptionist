"""
Voice Integration: Speech to Text (STT)

=== WHAT IS THIS FILE? ===
This module handles listening to the patient. It takes an audio file
and uses OpenAI's Whisper model to transcribe it into text that the LangChain 
Agent can read.

=== WHY WHISPER? ===
Whisper is an incredibly accurate, open-source model. The "base" model is small 
(under 150MB) and runs very fast on Mac CPUs while still maintaining great accuracy.
"""

import logging
import warnings
import os

# Suppress annoying warnings from PyTorch/Whisper about CPU usage
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self):
        """
        Initialize the Whisper model.
        The first time this runs, it will download the 'base' model to your computer.
        """
        logger.info("🎤 Initializing Speech-to-Text module...")
        try:
            import whisper
            logger.info("📦 Loading Whisper 'base' model (first time takes ~30s)...")
            
            # Load the model directly. "base" strikes a great balance for a Mac without GPU.
            self.model = whisper.load_model("base")
            logger.info("✅ Whisper model ready to listen!")
            self.ready = True
        except ImportError:
            logger.error("❌ Whisper is not installed. Did you run pip install openai-whisper?")
            self.ready = False
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
            self.ready = False

    def transcribe(self, audio_filepath: str) -> str:
        """
        Takes the path to an audio file (.wav, .mp3, etc.) and returns the text.
        """
        if not self.ready:
            return "Error: Speech recognition is currently unavailable."
            
        if not os.path.exists(audio_filepath):
            logger.error(f"Audio file not found: {audio_filepath}")
            return ""

        try:
            logger.info(f"🎙️ Transcribing audio from {audio_filepath}...")
            
            # The heart of the STT - model reads the audio
            result = self.model.transcribe(audio_filepath)
            
            text = result["text"].strip()
            logger.info(f"📝 Heard: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return ""
