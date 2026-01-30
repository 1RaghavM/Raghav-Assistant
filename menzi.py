"""Menzi Voice Assistant - Local Voice Agent with Gemini."""

import os
import re
import sys
import subprocess
import platform
import time
import threading
import numpy as np
from datetime import datetime
from typing import Optional

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


def speak(text: str):
    """Speak text using system TTS."""
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[\*#`\[\]]', '', text)
    text = text.strip()
    if not text:
        return

    if platform.system() == "Darwin":
        subprocess.run(["say", "-r", "200", text], capture_output=True)
    elif platform.system() == "Linux":
        subprocess.run(["espeak", "-s", "150", text], capture_output=True)


def listen() -> tuple[Optional[str], Optional[np.ndarray]]:
    """Listen for speech and return (text, audio_array)."""
    import speech_recognition as sr

    rec = sr.Recognizer()
    rec.energy_threshold = 300
    rec.pause_threshold = 1.5
    rec.dynamic_energy_threshold = False

    print("\n🎤 Listening... (speak now)")

    try:
        with sr.Microphone(sample_rate=16000) as source:
            audio = rec.listen(source, timeout=10, phrase_time_limit=20)

        print("⏳ Processing...")

        # Get audio data for voice ID
        audio_data = np.frombuffer(
            audio.get_raw_data(), dtype=np.int16
        ).astype(np.float32) / 32768.0

        # Transcribe
        text = rec.recognize_google(audio)
        return text, audio_data

    except sr.WaitTimeoutError:
        print("⏰ No speech detected")
        return None, None
    except sr.UnknownValueError:
        print("❓ Could not understand")
        return None, None
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


class Menzi:
    """Voice assistant with Gemini."""

    def __init__(self):
        print("🚀 Starting Menzi...")

        if not GEMINI_API_KEY:
            raise ValueError("Set GEMINI_API_KEY in .env")

        genai.configure(api_key=GEMINI_API_KEY)

        self.tools = get_gemini_tools()
        self.model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            tools=[self.tools],
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        self.chat = None
        self.last_call = 0

        # Components
        self.identity = IdentityResolver(
            voice_dir=VOICE_EMBEDDINGS_DIR,
            face_dir=FACE_ENCODINGS_DIR
        )
        self.memory = UserMemory()
        self.registry = UserRegistry()
        self.routines = RoutineTracker()
        self.camera = Camera()
        self.vlm = VLM()

        # State
        self.user = None
        self.is_admin = False
        self.executor = None
        self.msg_count = 0

        print("✅ Ready!")

    def _rate_limit(self):
        elapsed = time.time() - self.last_call
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self.last_call = time.time()

    def _system_prompt(self) -> str:
        parts = ["You are Menzi, a voice assistant. Be brief (1-2 sentences)."]

        if self.user:
            parts.append(f"User: {self.user}")
            mems = self.memory.get_all(self.user)
            if mems:
                parts.append(f"Facts: {'; '.join(mems[:3])}")

        if self.is_admin:
            parts.append("Admin user.")

        return " ".join(parts)

    def _new_chat(self):
        self._rate_limit()
        self.chat = self.model.start_chat(history=[])
        self.msg_count = 0

    def _set_user(self, name: str):
        self.user = name
        self.is_admin = self.registry.is_admin(name)
        self.executor = ToolExecutor(
            current_user=name,
            is_admin=self.is_admin,
            camera=self.camera,
            vlm=self.vlm
        )
        self._new_chat()

    def ask(self, text: str) -> str:
        """Send message to Gemini."""
        self.msg_count += 1
        if self.msg_count > MAX_MESSAGES or not self.chat:
            self._new_chat()

        prompt = f"[{self._system_prompt()}]\n\nUser: {text}"

        try:
            self._rate_limit()
            resp = self.chat.send_message(prompt)

            # Handle tool calls
            while resp.candidates and resp.candidates[0].content.parts:
                fc = None
                for part in resp.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        break

                if not fc:
                    break

                name = fc.name
                args = dict(fc.args) if fc.args else {}
                print(f"🔧 Tool: {name}")

                result = self.executor.execute(name, args)

                self._rate_limit()
                resp = self.chat.send_message(
                    genai.protos.Content(parts=[
                        genai.protos.Part(function_response=genai.protos.FunctionResponse(
                            name=name, response={"result": result}
                        ))
                    ])
                )

            # Extract response
            for part in resp.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    txt = part.text.strip()
                    if "</think>" in txt:
                        txt = txt.split("</think>")[-1].strip()
                    return txt

            return "I'm not sure."

        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return "Sorry, something went wrong."

    def identify(self, audio: np.ndarray) -> Optional[str]:
        """Identify user from voice."""
        user, conf, _ = self.identity.resolve(voice_audio=audio)
        if user and conf >= 0.6:
            print(f"👤 Identified: {user} ({conf:.0%})")
            return user
        return None

    def enroll(self, audio: np.ndarray) -> Optional[str]:
        """Enroll new user."""
        speak("I don't recognize you. Please say your first name.")

        name_text, _ = listen()
        if not name_text:
            return None

        # Get name (last word handles "My name is X")
        name = name_text.strip().split()[-1].lower()
        name = re.sub(r'[^a-z]', '', name)

        if len(name) < 2:
            speak("I didn't catch that.")
            return None

        print(f"📝 Enrolling: {name}")
        speak(f"Hi {name}! Saving your voice.")

        self.identity.enroll_user(name, voice_audio=audio)

        # Try face
        frame = self.camera.capture()
        if frame is not None:
            try:
                self.identity.enroll_user(name, face_image=frame)
            except:
                pass

        is_admin = name == ADMIN_USER.lower()
        self.registry.create_user(name, admin=is_admin)

        speak(f"Nice to meet you, {name}!")
        return name

    def run(self):
        """Main loop."""
        print("\n" + "=" * 40)
        print("   MENZI VOICE ASSISTANT")
        print("=" * 40)
        print("Say 'quit' to exit\n")

        speak("Hello! I'm Menzi.")

        try:
            # Identify
            speak("Who am I speaking with?")
            text, audio = listen()

            if audio is not None:
                user = self.identify(audio)
                if user:
                    self._set_user(user)
                    speak(f"Welcome back, {user}!")
                else:
                    user = self.enroll(audio)
                    if user:
                        self._set_user(user)

            if not self.user:
                self._set_user("guest")
                speak("I'll call you guest for now.")

            # Chat loop
            while True:
                text, audio = listen()

                if not text:
                    continue

                print(f"\n💬 You: {text}")

                if text.lower().strip() in ["quit", "exit", "stop", "bye", "goodbye"]:
                    speak("Goodbye!")
                    break

                response = self.ask(text)
                print(f"🤖 Menzi: {response}")
                speak(response)

        except KeyboardInterrupt:
            print("\n👋 Interrupted")
            speak("Goodbye!")
        finally:
            self.camera.release()


if __name__ == "__main__":
    try:
        Menzi().run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
