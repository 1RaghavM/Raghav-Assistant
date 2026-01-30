"""Menzi Voice Assistant - Gemini API Version."""

import re
import sys
import os
import queue
import threading
import subprocess
import platform
import time
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    VOICE_EMBEDDINGS_DIR,
    FACE_ENCODINGS_DIR,
    MAX_MESSAGES,
    VISION_CONTEXT_EXPIRY_MINUTES,
    ADMIN_USER,
)
from core.tools import get_gemini_tools
from core.executor import ToolExecutor
from memory.user_memory import UserMemory
from memory.user_registry import UserRegistry
from memory.routines import RoutineTracker
from identity.resolver import IdentityResolver
from vision.camera import Camera
from vision.vlm import VLM


class TTS:
    """Text-to-speech."""

    def __init__(self):
        self.system = platform.system()
        self.queue = queue.Queue()
        self.running = True
        self.proc = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _clean(self, text: str) -> str:
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[\*#`]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _worker(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    break
                text = self._clean(text)
                if text and len(text) > 1:
                    if self.system == "Darwin":
                        self.proc = subprocess.Popen(
                            ["say", "-r", "190", text],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    elif self.system == "Linux":
                        self.proc = subprocess.Popen(
                            ["espeak", "-s", "150", text],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    if self.proc:
                        self.proc.wait()
                        self.proc = None
                self.queue.task_done()
            except queue.Empty:
                continue

    def say(self, text: str):
        self.queue.put(text)

    def say_sync(self, text: str):
        self.say(text)
        self.queue.join()

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=2)
        if self.proc:
            self.proc.terminate()


class STT:
    """Speech-to-text."""

    def __init__(self):
        import speech_recognition as sr
        self.sr = sr
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 300  # Fixed threshold
        self.rec.dynamic_energy_threshold = False  # Don't auto-adjust
        self.rec.pause_threshold = 1.5
        self.rec.phrase_threshold = 0.2
        self.rec.non_speaking_duration = 0.3
        self.last_audio = None

    def listen(self, prompt: str = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
        if prompt:
            print(f"\n[{prompt}]")
        else:
            print("\n[Listening...]")

        try:
            with self.sr.Microphone(sample_rate=16000) as source:
                print("[Speak now...]")
                audio = self.rec.listen(source, timeout=10, phrase_time_limit=20)

            print("[Processing...]")
            audio_data = np.frombuffer(
                audio.get_raw_data(), dtype=np.int16
            ).astype(np.float32) / 32768.0
            self.last_audio = audio_data

            text = self.rec.recognize_google(audio)
            return text, audio_data

        except self.sr.WaitTimeoutError:
            print("[Timeout]")
            return None, None
        except self.sr.UnknownValueError:
            print("[Could not understand]")
            return None, self.last_audio
        except self.sr.RequestError as e:
            print(f"[STT Error: {e}]")
            return None, None


class Menzi:
    """Main assistant using Gemini API."""

    def __init__(self):
        print("Initializing Menzi...")

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env file")

        # Configure Gemini - single instance
        genai.configure(api_key=GEMINI_API_KEY)

        # Get tools in Gemini format
        self.tools = get_gemini_tools()

        # Create model with tools
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            tools=[self.tools],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=256,  # Keep responses short
            )
        )

        self.chat = None
        self.last_api_call = 0
        self.min_call_interval = 0.5  # Minimum seconds between API calls

        # Components
        self.tts = TTS()
        self.stt = STT()
        self.camera = Camera()
        self.vlm = VLM()

        self.identity = IdentityResolver(
            voice_dir=VOICE_EMBEDDINGS_DIR,
            face_dir=FACE_ENCODINGS_DIR
        )
        self.memory = UserMemory()
        self.registry = UserRegistry()
        self.routines = RoutineTracker()

        # State
        self.user = None
        self.is_admin = False
        self.executor = None
        self.vision_context = None
        self.vision_time = None
        self.message_count = 0

        print("Ready!")

    def _rate_limit(self):
        """Ensure minimum time between API calls."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_api_call = time.time()

    def _get_system_prompt(self) -> str:
        parts = [
            "You are Menzi, a helpful voice assistant. Keep responses to 1-2 sentences.",
            f"User: {self.user or 'Unknown'}"
        ]

        if self.user:
            memories = self.memory.get_all(self.user)
            if memories:
                parts.append(f"Known: {'; '.join(memories[:3])}")

        if self.vision_context and self.vision_time:
            age = (datetime.now() - self.vision_time).total_seconds() / 60
            if age < VISION_CONTEXT_EXPIRY_MINUTES:
                parts.append(f"Saw: {self.vision_context[:100]}")

        if self.is_admin:
            parts.append("Admin user.")

        return " ".join(parts)

    def _start_chat(self):
        """Start fresh chat session."""
        self._rate_limit()
        self.chat = self.model.start_chat(history=[])
        self.message_count = 0

    def _identify(self, audio: np.ndarray) -> Optional[str]:
        user, conf, method = self.identity.resolve(voice_audio=audio)
        if user and conf >= 0.6:
            print(f"[Identified: {user} ({conf:.0%})]")
            return user
        return None

    def _enroll(self, audio: np.ndarray) -> Optional[str]:
        self.tts.say_sync("I don't recognize you. Say just your first name.")

        name_text, _ = self.stt.listen("Your name")
        if not name_text:
            return None

        words = name_text.strip().split()
        name = words[-1].lower().replace(".", "").replace(",", "")

        if len(name) < 2 or not name.isalpha():
            self.tts.say_sync("Sorry, I didn't catch that.")
            return None

        print(f"[Enrolling: {name}]")
        self.tts.say_sync(f"Hi {name}! Saving your voice.")

        self.identity.enroll_user(name, voice_audio=audio)

        frame = self.camera.capture()
        if frame is not None:
            try:
                self.identity.enroll_user(name, face_image=frame)
            except:
                pass

        is_admin = name.lower() == ADMIN_USER.lower()
        self.registry.create_user(name, admin=is_admin)

        self.tts.say_sync(f"Nice to meet you, {name}!")
        return name

    def _set_user(self, name: str):
        self.user = name
        self.is_admin = self.registry.is_admin(name)
        self.executor = ToolExecutor(
            current_user=name,
            is_admin=self.is_admin,
            camera=self.camera,
            vlm=self.vlm
        )
        self._start_chat()

    def _call_gemini(self, user_input: str) -> str:
        """Send message to Gemini - sequential, rate-limited."""

        # Check if we need to reset chat
        self.message_count += 1
        if self.message_count > MAX_MESSAGES or self.chat is None:
            self._start_chat()

        # Build context prefix
        context = self._get_system_prompt()
        full_input = f"[Context: {context}]\n\nUser: {user_input}"

        try:
            # Rate limit before call
            self._rate_limit()

            response = self.chat.send_message(full_input)

            # Handle function calls - one at a time, sequentially
            while response.candidates and response.candidates[0].content.parts:
                has_function_call = False

                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        has_function_call = True
                        fc = part.function_call
                        name = fc.name
                        args = dict(fc.args) if fc.args else {}

                        print(f"[Tool: {name}]")

                        # Track routines
                        if self.user and name in ["get_weather", "get_news"]:
                            self.routines.record_query(self.user, name.replace("get_", ""))

                        result = self.executor.execute(name, args)

                        if name == "see_camera":
                            self.vision_context = result
                            self.vision_time = datetime.now()

                        # Rate limit before next call
                        self._rate_limit()

                        # Send function result back
                        response = self.chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=name,
                                        response={"result": result}
                                    )
                                )]
                            )
                        )
                        break  # Process one function at a time

                if not has_function_call:
                    break

            # Extract text response
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text.strip()
                        # Clean thinking tags
                        if "</think>" in text:
                            text = text.split("</think>")[-1].strip()
                        if self.user:
                            self.routines.record_interaction(self.user)
                        return text

            return "I'm not sure how to respond."

        except Exception as e:
            print(f"[Gemini Error: {e}]")
            return "Sorry, I had trouble with that request."

    def run(self):
        print("\n" + "=" * 40)
        print("MENZI VOICE ASSISTANT")
        print("=" * 40)
        print("Say 'quit' to stop\n")

        self.tts.say_sync("Hello! I'm Menzi.")

        try:
            # Identify user
            self.tts.say_sync("Who am I speaking with?")
            text, audio = self.stt.listen("Say your name")

            if audio is not None:
                user = self._identify(audio)
                if user:
                    self._set_user(user)
                    self.tts.say_sync(f"Welcome back, {user}!")
                else:
                    user = self._enroll(audio)
                    if user:
                        self._set_user(user)

            if not self.user:
                self._set_user("guest")
                self.tts.say_sync("I'll call you guest.")

            # Main loop
            while True:
                text, audio = self.stt.listen()

                if not text:
                    continue

                print(f"\nYou: {text}")

                if text.lower().strip() in ["quit", "exit", "goodbye", "bye", "stop"]:
                    self.tts.say_sync("Goodbye!")
                    break

                response = self._call_gemini(text)
                print(f"Menzi: {response}")

                # Speak
                for sentence in re.split(r'(?<=[.!?])\s+', response):
                    if sentence.strip():
                        self.tts.say(sentence.strip())
                self.tts.queue.join()

        except KeyboardInterrupt:
            print("\n[Interrupted]")
            self.tts.say_sync("Goodbye!")
        finally:
            self.tts.stop()
            self.camera.release()


def main():
    try:
        Menzi().run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
