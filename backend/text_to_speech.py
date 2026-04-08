"""
Voice Integration: Text to Speech (TTS)

=== WHAT IS THIS FILE? ===
This module handles speaking back to the patient. It takes the text generated
by the LangChain Agent, and creates an audio file that the patient can hear.

=== WHY COQUI TTS? ===
Coqui TTS is a powerful open-source library that gives access to high-quality
voices. It sounds much more natural and empathetic than standard robot voices,
which is critical for a pleasant Hospital Receptionist experience!
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self):
        """
        Initialize the TTS model.
        We're using a VITS model which generates high-quality speech fast enough for a Mac.
        """
        logger.info("🔊 Initializing Text-to-Speech module...")
        
        try:
            # We redirect stdout temporarily to hide the massive warning text Coqui TTS usually prints
            from TTS.api import TTS
            import torch
            
            logger.info("📦 Loading Coqui TTS model (first time takes ~2 mins)...")
            
            # The VCTK dataset has over 100 voices. p225 is commonly used as a clean female voice.
            # model_name = "tts_models/en/vctk/vits"
            
            # Use CPU on Mac
            self.device = "cpu"
            
            # We will use the built-in macOS speech synthesizer if TTS loading fails or is too heavy.
            # But we try Coqui first for premium quality.
            self.tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False).to(self.device)
            self.ready = True
            logger.info("✅ Premium TTS model ready to speak!")
            
        except ImportError:
            logger.error("❌ Coqui TTS is not installed. Did you run pip install TTS?")
            self.ready = False
        except Exception as e:
            logger.error(f"❌ Failed to load Coqui TTS: {e}")
            logger.info("⚠️ Falling back to basic system voice...")
            self.ready = False

    def synthesize(self, text: str, output_filepath: str) -> bool:
        """
        Takes text, turns it into a voice, and saves it to a file.
        Returns True if successful.
        """
        if not text:
            return False
            
        logger.info(f"🗣️ Synthesizing speech: '{text[:50]}...'")
        
        try:
            if self.ready:
                # Use premium Coqui TTS
                # speaker="p225" is standard female. p326 is deep male.
                self.tts.tts_to_file(text=text, speaker="p225", file_path=output_filepath)
                logger.info(f"💾 Audio saved: {output_filepath}")
                return True
            else:
                # Fallback to macOS native speech if Coqui isn't ready
                # This works purely on the command line!
                os.system(f'say -v "Samantha" -o "{output_filepath}" --data-format=LEF32@32000 "{text}"')
                logger.info(f"💾 Basic audio saved: {output_filepath}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Speech synthesis failed: {e}")
            return False
