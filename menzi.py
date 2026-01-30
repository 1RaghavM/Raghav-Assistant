"""Menzi Voice Assistant - Main Application."""

import re
import sys
import queue
import threading
import subprocess
import platform
import json
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

import ollama

from config import (
    LLM_MODEL,
    VOICE_EMBEDDINGS_DIR,
    FACE_ENCODINGS_DIR,
    MAX_MESSAGES,
    VISION_CONTEXT_EXPIRY_MINUTES,
    ADMIN_USER,
)
from core.tools import get_ollama_tools
from core.executor import ToolExecutor
from memory.user_memory import UserMemory
from memory.user_registry import UserRegistry
from memory.routines import RoutineTracker
from identity.resolver import IdentityResolver
from vision.camera import Camera
from vision.vlm import VLM


class TTS:
    """Simple text-to-speech."""

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
                            ["say", "-r", "180", text],
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
    """Simple speech-to-text."""

    def __init__(self):
        import speech_recognition as sr
        self.sr = sr
        self.rec = sr.Recognizer()
        self.rec.energy_threshold = 400
        self.rec.dynamic_energy_threshold = False
        self.rec.pause_threshold = 2.0  # Wait 2 seconds of silence
        self.rec.phrase_threshold = 0.3
        self.last_audio = None

    def listen(self, prompt: str = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Listen and return (text, audio_array)."""
        if prompt:
            print(f"\n[{prompt}]")
        else:
            print("\n[Listening...]")

        try:
            with self.sr.Microphone(sample_rate=16000) as source:
                self.rec.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.rec.listen(source, timeout=8, phrase_time_limit=15)

            print("[Processing...]")
            audio_data = np.frombuffer(
                audio.get_raw_data(), dtype=np.int16
            ).astype(np.float32) / 32768.0
            self.last_audio = audio_data

            text = self.rec.recognize_google(audio)
            return text, audio_data

        except self.sr.WaitTimeoutError:
            print("[Timeout - no speech detected]")
            return None, None
        except self.sr.UnknownValueError:
            print("[Could not understand]")
            return None, self.last_audio
        except self.sr.RequestError as e:
            print(f"[STT Error: {e}]")
            return None, None


class Menzi:
    """Main assistant."""

    def __init__(self):
        print("Initializing Menzi...")
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

        self.user = None
        self.is_admin = False
        self.messages = []
        self.executor = None
        self.tools = get_ollama_tools()
        self.vision_context = None
        self.vision_time = None

        print("Ready!")

    def _get_system_prompt(self) -> str:
        parts = [
            f"You are Menzi, a helpful voice assistant. Be concise (1-3 sentences).",
            f"Current user: {self.user or 'Unknown'}"
        ]

        if self.user:
            memories = self.memory.get_all(self.user)
            if memories:
                parts.append(f"User facts: {'; '.join(memories[:5])}")

            loc = self.registry.get_location(self.user)
            if loc:
                parts.append(f"User location: {loc}")

        if self.vision_context and self.vision_time:
            age = (datetime.now() - self.vision_time).total_seconds() / 60
            if age < VISION_CONTEXT_EXPIRY_MINUTES:
                parts.append(f"Recent visual: {self.vision_context}")

        if self.is_admin:
            parts.append("User is admin - can use admin_command tool.")

        return " ".join(parts)

    def _reset_chat(self):
        self.messages = [{"role": "system", "content": self._get_system_prompt()}]

    def _identify(self, audio: np.ndarray) -> Optional[str]:
        """Try to identify user from voice."""
        user, conf, method = self.identity.resolve(voice_audio=audio)
        if user and conf >= 0.6:
            print(f"[Identified: {user} ({conf:.0%} via {method})]")
            return user
        return None

    def _enroll(self, audio: np.ndarray) -> Optional[str]:
        """Enroll new user."""
        self.tts.say_sync("I don't recognize you. Please say just your first name.")

        name_text, _ = self.stt.listen("Say your name")
        if not name_text:
            return None

        # Clean the name - take last word to handle "My name is X" or "I'm X"
        words = name_text.strip().split()
        name = words[-1].lower().replace(".", "").replace(",", "")

        # Validate it looks like a name
        if len(name) < 2 or not name.isalpha():
            self.tts.say_sync("Sorry, I didn't catch that.")
            return None

        print(f"[Enrolling: {name}]")
        self.tts.say_sync(f"Hi {name}! Let me remember your voice.")

        # Enroll voice
        self.identity.enroll_user(name, voice_audio=audio)

        # Try face
        frame = self.camera.capture()
        if frame is not None:
            try:
                self.identity.enroll_user(name, face_image=frame)
            except:
                pass

        # Register
        is_admin = name.lower() == ADMIN_USER.lower()
        self.registry.create_user(name, admin=is_admin)

        self.tts.say_sync(f"Got it! Nice to meet you, {name}.")
        return name

    def _set_user(self, name: str):
        """Set current user."""
        self.user = name
        self.is_admin = self.registry.is_admin(name)
        self.executor = ToolExecutor(
            current_user=name,
            is_admin=self.is_admin,
            camera=self.camera,
            vlm=self.vlm
        )
        self._reset_chat()

    def _call_llm(self, user_input: str) -> str:
        """Send message to LLM and handle tool calls."""
        self.messages.append({"role": "user", "content": user_input})

        # Trim history
        if len(self.messages) > MAX_MESSAGES:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(MAX_MESSAGES-1):]

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=self.messages,
                tools=self.tools
            )
        except Exception as e:
            print(f"[LLM Error: {e}]")
            return "Sorry, I had trouble processing that."

        msg = response.get("message", {})

        # Handle tool calls
        while msg.get("tool_calls"):
            self.messages.append(msg)

            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                print(f"[Tool: {name}({args})]")

                # Track routines
                if self.user and name in ["get_weather", "get_news"]:
                    self.routines.record_query(self.user, name.replace("get_", ""))

                result = self.executor.execute(name, args)

                # Update vision context
                if name == "see_camera":
                    self.vision_context = result
                    self.vision_time = datetime.now()

                self.messages.append({"role": "tool", "content": result})

            try:
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=self.messages,
                    tools=self.tools
                )
            except Exception as e:
                print(f"[LLM Error: {e}]")
                return "Sorry, there was an error."

            msg = response.get("message", {})

        content = msg.get("content", "")

        # Clean thinking tags
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        self.messages.append({"role": "assistant", "content": content})

        # Record interaction
        if self.user:
            self.routines.record_interaction(self.user)

        return content or "I'm not sure how to respond."

    def run(self):
        """Main loop."""
        print("\n" + "=" * 40)
        print("MENZI VOICE ASSISTANT")
        print("=" * 40)
        print("Say 'quit' or 'exit' to stop\n")

        self.tts.say_sync("Hello! I'm Menzi.")

        try:
            # Initial identification
            self.tts.say_sync("Who am I speaking with?")
            text, audio = self.stt.listen("Say your name or greeting")

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
                self.user = "guest"
                self._set_user("guest")
                self.tts.say_sync("I'll call you guest for now.")

            # Main conversation loop
            while True:
                text, audio = self.stt.listen()

                if not text:
                    continue

                print(f"\nYou: {text}")

                # Exit check
                if text.lower().strip() in ["quit", "exit", "goodbye", "bye", "stop"]:
                    self.tts.say_sync("Goodbye!")
                    break

                # Get response
                response = self._call_llm(text)
                print(f"Menzi: {response}")

                # Speak response
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
