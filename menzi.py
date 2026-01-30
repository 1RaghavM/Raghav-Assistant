"""Menzi Voice Assistant - Streaming Cerebras LLM + SambaNova Whisper + Edge TTS."""

import os
import re
import sys
import json
import numpy as np
from typing import Optional, Generator

from cerebras.cloud.sdk import Cerebras

from config import (
    CEREBRAS_API_KEY,
    LLM_MODEL,
    VOICE_EMBEDDINGS_DIR,
    FACE_ENCODINGS_DIR,
    MAX_MESSAGES,
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
from audio.stt import VoiceInput

# Try streaming TTS first, fall back to regular
try:
    from audio.streaming_tts import ChunkedStreamingTTS, REALTIME_TTS_AVAILABLE
    if REALTIME_TTS_AVAILABLE:
        STREAMING_TTS = True
    else:
        STREAMING_TTS = False
except ImportError:
    STREAMING_TTS = False

if not STREAMING_TTS:
    from audio.tts import speak


class Menzi:
    """Voice assistant with streaming Cerebras LLM + real-time TTS."""

    def __init__(self):
        print("Starting Menzi...")

        if not CEREBRAS_API_KEY:
            raise ValueError("Set CEREBRAS_API_KEY in .env")

        # Initialize Cerebras client
        self.client = Cerebras(api_key=CEREBRAS_API_KEY)
        self.model = LLM_MODEL
        self.tools = get_openai_tools()

        print(f"[LLM] Using Cerebras {self.model}")

        # Voice input with wake word + cloud STT
        self.voice = VoiceInput()

        # Streaming TTS for instant responses
        if STREAMING_TTS:
            self.tts = ChunkedStreamingTTS()
            print("[TTS] Streaming mode enabled")
        else:
            self.tts = None
            print("[TTS] Using non-streaming fallback")

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

        print("Ready!")

    def _system_prompt(self) -> str:
        parts = ["You are Menzi, a helpful voice assistant. Be conversational and brief (1-2 sentences)."]

        if self.user and self.user != "unknown":
            parts.append(f"You're speaking with {self.user}.")
            mems = self.memory.get_all(self.user)
            if mems:
                parts.append(f"You know: {'; '.join(mems[:5])}")
            if self.is_admin:
                parts.append("They have admin access.")
        else:
            parts.append("You don't know who this is yet. If context requires knowing their name, ask naturally.")

        return " ".join(parts)

    def _reset_messages(self):
        self.messages = [
            {"role": "system", "content": self._system_prompt()}
        ]

    def _on_user_change(self, name: str):
        """Callback when user is changed via tool."""
        self.user = name
        self.is_admin = self.registry.is_admin(name)
        if self.executor:
            self.executor.current_user = name
            self.executor.is_admin = self.is_admin
        # Update system prompt with new user info
        if self.messages:
            self.messages[0] = {"role": "system", "content": self._system_prompt()}

    def _set_user(self, name: str):
        self.user = name
        self.is_admin = self.registry.is_admin(name) if name != "unknown" else False
        self.executor = ToolExecutor(
            current_user=name,
            is_admin=self.is_admin,
            camera=self.camera,
            vlm=self.vlm,
            on_user_change=self._on_user_change
        )
        self._reset_messages()

    def _speak(self, text: str):
        """Speak text using available TTS."""
        if self.tts:
            self.tts.speak(text)
        else:
            speak(text)

    def _stream_response(self) -> Generator[str, None, str]:
        """Stream response from Cerebras, yielding chunks.

        Returns full response at the end.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_completion_tokens=300,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        return full_response

    def ask(self, text: str, speak_response: bool = True) -> str:
        """Send message to Cerebras with streaming.

        Args:
            text: User message
            speak_response: Whether to speak the response (default True)

        Returns:
            Response text
        """
        self.messages.append({"role": "user", "content": text})

        # Trim history
        if len(self.messages) > MAX_MESSAGES:
            system = self.messages[0]
            self.messages = [system] + self.messages[-(MAX_MESSAGES - 1):]

        try:
            # First check for tool calls (non-streaming)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_completion_tokens=300,
            )

            msg = response.choices[0].message

            # Handle tool calls (can't stream with tools)
            if msg.tool_calls:
                while msg.tool_calls:
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

                    for tc in msg.tool_calls:
                        name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except:
                            args = {}

                        print(f"[Tool] {name}")
                        result = self.executor.execute(name, args)

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        })

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=self.tools,
                        tool_choice="auto",
                        max_completion_tokens=300,
                    )
                    msg = response.choices[0].message

                content = msg.content or "Done."

                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                self.messages.append({"role": "assistant", "content": content})

                if self.user:
                    self.routines.record_interaction(self.user)

                # Speak tool response (non-streaming)
                if speak_response and content:
                    print(content)
                    self._speak(content)

                return content

            # No tool calls - use streaming for faster response
            full_response = ""

            if STREAMING_TTS and self.tts and speak_response:
                # Stream LLM -> TTS pipeline
                def text_generator():
                    nonlocal full_response
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        max_completion_tokens=300,
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            print(content, end="", flush=True)
                            yield content
                    print()  # Newline after streaming

                self.tts.feed_and_play(text_generator())
            else:
                # Non-streaming fallback
                full_response = msg.content or "I'm not sure."
                if speak_response:
                    print(full_response)
                    self._speak(full_response)

            # Clean thinking tags
            if "</think>" in full_response:
                full_response = full_response.split("</think>")[-1].strip()

            self.messages.append({"role": "assistant", "content": full_response})

            if self.user:
                self.routines.record_interaction(self.user)

            return full_response

        except Exception as e:
            print(f"❌ Cerebras error: {e}")
            return "Sorry, something went wrong."

    def identify(self, audio: np.ndarray) -> Optional[str]:
        """Identify user from voice."""
        user, conf, _ = self.identity.resolve(voice_audio=audio)
        if user and conf >= 0.6:
            print(f"Identified: {user} ({conf:.0%})")
            return user
        return None

    def enroll(self, audio: np.ndarray) -> Optional[str]:
        """Enroll new user."""
        self._speak("I don't recognize you. Please say your first name.")

        # Listen without wake word for enrollment
        name_text, _ = self.voice.listen(require_wake_word=False)
        if not name_text:
            return None

        name = name_text.strip().split()[-1].lower()
        name = re.sub(r'[^a-z]', '', name)

        if len(name) < 2:
            self._speak("I didn't catch that.")
            return None

        print(f"Enrolling: {name}")
        self._speak(f"Hi {name}! Saving your voice.")

        self.identity.enroll_user(name, voice_audio=audio)

        frame = self.camera.capture()
        if frame is not None:
            try:
                self.identity.enroll_user(name, face_image=frame)
            except:
                pass

        is_admin = name == ADMIN_USER.lower()
        self.registry.create_user(name, admin=is_admin)

        self._speak(f"Nice to meet you, {name}!")
        return name

    def run(self):
        """Main loop with streaming responses."""
        print("\n" + "=" * 40)
        print("   MENZI - Say 'Menzi' to start")
        print("=" * 40 + "\n")

        # Start with unknown user - will identify from voice
        self._set_user("unknown")

        try:
            while True:
                text, audio = self.voice.listen()

                if not text:
                    continue

                # Try to identify user from voice (silently in background)
                if audio is not None and self.user == "unknown":
                    identified = self.identify(audio)
                    if identified:
                        self._set_user(identified)

                print(f"\nYou: {text}")

                if text.lower().strip() in ["quit", "exit", "stop", "bye", "goodbye"]:
                    self._speak("Goodbye!")
                    break

                # ask() handles both streaming and speaking internally
                print("Menzi: ", end="", flush=True)
                self.ask(text)

        except KeyboardInterrupt:
            print("\nInterrupted")
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
