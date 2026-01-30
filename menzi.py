"""Menzi Voice Assistant - Cerebras LLM Version."""

import os
import re
import sys
import subprocess
import platform
import time
import json
import numpy as np
from datetime import datetime
from typing import Optional

from cerebras.cloud.sdk import Cerebras

from config import (
    CEREBRAS_API_KEY,
    LLM_MODEL,
    VOICE_EMBEDDINGS_DIR,
    FACE_ENCODINGS_DIR,
    MAX_MESSAGES,
    VISION_CONTEXT_EXPIRY_MINUTES,
    ADMIN_USER,
)
from core.tools import get_openai_tools
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

        audio_data = np.frombuffer(
            audio.get_raw_data(), dtype=np.int16
        ).astype(np.float32) / 32768.0

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
    """Voice assistant with Cerebras LLM."""

    def __init__(self):
        print("🚀 Starting Menzi...")

        if not CEREBRAS_API_KEY:
            raise ValueError("Set CEREBRAS_API_KEY in .env")

        # Initialize Cerebras client
        self.client = Cerebras(api_key=CEREBRAS_API_KEY)
        self.model = LLM_MODEL
        self.tools = get_openai_tools()

        print(f"[LLM] Using Cerebras {self.model}")

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
        self.messages = []

        print("✅ Ready!")

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

    def _reset_messages(self):
        self.messages = [
            {"role": "system", "content": self._system_prompt()}
        ]

    def _set_user(self, name: str):
        self.user = name
        self.is_admin = self.registry.is_admin(name)
        self.executor = ToolExecutor(
            current_user=name,
            is_admin=self.is_admin,
            camera=self.camera,
            vlm=self.vlm
        )
        self._reset_messages()

    def ask(self, text: str) -> str:
        """Send message to Cerebras."""
        self.messages.append({"role": "user", "content": text})

        # Trim history
        if len(self.messages) > MAX_MESSAGES:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(MAX_MESSAGES - 1):]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_completion_tokens=300,
            )

            msg = response.choices[0].message

            # Handle tool calls
            while msg.tool_calls:
                # Add assistant message with tool calls
                self.messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                # Execute each tool
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except:
                        args = {}

                    print(f"🔧 Tool: {name}")
                    result = self.executor.execute(name, args)

                    # Add tool result
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

                # Get next response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_completion_tokens=300,
                )
                msg = response.choices[0].message

            # Extract final response
            content = msg.content or "I'm not sure."

            # Clean thinking tags if present
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()

            self.messages.append({"role": "assistant", "content": content})

            if self.user:
                self.routines.record_interaction(self.user)

            return content

        except Exception as e:
            print(f"❌ Cerebras error: {e}")
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

        name = name_text.strip().split()[-1].lower()
        name = re.sub(r'[^a-z]', '', name)

        if len(name) < 2:
            speak("I didn't catch that.")
            return None

        print(f"📝 Enrolling: {name}")
        speak(f"Hi {name}! Saving your voice.")

        self.identity.enroll_user(name, voice_audio=audio)

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
        print(f"   Powered by Cerebras {self.model}")
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
