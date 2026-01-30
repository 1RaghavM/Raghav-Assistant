"""Menzi Voice Assistant - Main Application.

Integrates all modules for a multi-user voice assistant with:
- Google Gemini API for conversation (cloud LLM)
- Local VLM (moondream via Ollama) for vision
- Voice + Face identity resolution
- Per-user memories and routines
- Tool execution with user context
"""

import os
import re
import sys
import queue
import threading
import subprocess
import platform
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import google.generativeai as genai

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    DATA_DIR,
    VOICE_EMBEDDINGS_DIR,
    FACE_ENCODINGS_DIR,
    MAX_MESSAGES,
    VISION_CONTEXT_EXPIRY_MESSAGES,
    VISION_CONTEXT_EXPIRY_MINUTES,
    ADMIN_USER,
)
from core.tools import get_gemini_tools, get_tool_names
from core.executor import ToolExecutor
from memory.user_memory import UserMemory
from memory.user_registry import UserRegistry
from memory.routines import RoutineTracker
from identity.resolver import IdentityResolver
from vision.camera import Camera
from vision.vlm import VLM


class TTSEngine:
    """Text-to-speech engine using system commands.

    Uses 'say' on macOS and 'espeak' on Linux.
    Runs in a background thread with queued sentences.
    """

    def __init__(self):
        self.system = platform.system()
        self.queue = queue.Queue()
        self.running = True
        self.current_proc = None
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _clean_text(self, text: str) -> str:
        """Clean text for speech synthesis."""
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Remove markdown formatting
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'`+', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [link](url) -> link
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _worker(self):
        """Background worker that processes speech queue."""
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    self.queue.task_done()
                    break

                text = self._clean_text(text)
                if text and len(text) > 2:
                    if self.system == "Darwin":
                        self.current_proc = subprocess.Popen(
                            ["say", "-r", "200", text],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    elif self.system == "Linux":
                        self.current_proc = subprocess.Popen(
                            ["espeak", "-s", "160", text],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:
                        # Windows or other - just skip TTS
                        pass

                    if self.current_proc:
                        self.current_proc.wait()
                        self.current_proc = None

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS ERROR] {e}")
                self.queue.task_done()

    def speak(self, text: str):
        """Queue text for speech (non-blocking)."""
        self.queue.put(text)

    def speak_and_wait(self, text: str):
        """Speak text and wait for completion."""
        self.speak(text)
        self.wait()

    def wait(self):
        """Wait for all queued speech to complete."""
        self.queue.join()

    def stop(self):
        """Stop the TTS engine."""
        self.running = False
        # Send sentinel to signal worker to stop
        self.queue.put(None)
        # Wait for worker thread to finish
        self.worker.join(timeout=2.0)
        if self.current_proc:
            self.current_proc.terminate()


class STTEngine:
    """Speech-to-text engine using SpeechRecognition.

    Also captures raw audio for voice identification.
    """

    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self._last_audio_data = None

    def listen(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Listen for speech and return (text, audio_array).

        Returns:
            Tuple of (transcribed text, raw audio as numpy array).
            Either or both may be None if listening/recognition failed.
        """
        import speech_recognition as sr

        with sr.Microphone(sample_rate=16000) as source:
            print("\n[LISTENING...]")
            try:
                audio = self.recognizer.listen(
                    source, timeout=10, phrase_time_limit=30
                )
                print("[PROCESSING...]")

                # Convert audio to numpy array for voice ID
                audio_data = np.frombuffer(
                    audio.get_raw_data(), dtype=np.int16
                ).astype(np.float32) / 32768.0
                self._last_audio_data = audio_data

                # Transcribe using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                return text, audio_data

            except sr.WaitTimeoutError:
                return None, None
            except sr.UnknownValueError:
                return None, audio_data  # Return current audio, not previous
            except sr.RequestError as e:
                print(f"[STT ERROR] {e}")
                return None, None

    def get_last_audio(self) -> Optional[np.ndarray]:
        """Get the last captured audio data."""
        return self._last_audio_data


class VisionContextManager:
    """Manages vision context with expiry based on time and messages."""

    def __init__(self, vlm: VLM, camera: Camera):
        self.vlm = vlm
        self.camera = camera
        self.last_description = None
        self.last_timestamp = None
        self.messages_since_vision = 0

    def capture_and_analyze(self, query: str) -> str:
        """Capture from camera and analyze with VLM."""
        frame = self.camera.capture()
        if frame is None:
            return "Could not capture image from camera."

        result = self.vlm.analyze(frame, query)
        self.last_description = result
        self.last_timestamp = datetime.now()
        self.messages_since_vision = 0
        return result

    def get_context(self) -> Optional[str]:
        """Get current vision context if still valid."""
        if self.last_description is None:
            return None

        # Check time expiry
        if self.last_timestamp:
            age_minutes = (datetime.now() - self.last_timestamp).total_seconds() / 60
            if age_minutes > VISION_CONTEXT_EXPIRY_MINUTES:
                self.clear()
                return None

        # Check message expiry
        if self.messages_since_vision > VISION_CONTEXT_EXPIRY_MESSAGES:
            self.clear()
            return None

        return self.last_description

    def increment_messages(self):
        """Called after each message exchange."""
        self.messages_since_vision += 1

    def clear(self):
        """Clear vision context."""
        self.last_description = None
        self.last_timestamp = None
        self.messages_since_vision = 0


class Menzi:
    """Main Menzi Voice Assistant application.

    Integrates:
    - Google Gemini for conversation
    - Voice + face identity resolution
    - Per-user memories and routines
    - Tool execution
    - Vision context management
    """

    def __init__(self):
        # Validate API key
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Please set the GEMINI_API_KEY environment variable."
            )

        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.tools = get_gemini_tools()
        self.model = genai.GenerativeModel(LLM_MODEL, tools=[self.tools])

        # Initialize components
        self.tts = TTSEngine()
        self.stt = STTEngine()
        self.camera = Camera()
        self.vlm = VLM()
        self.vision_context = VisionContextManager(self.vlm, self.camera)
        self.identity = IdentityResolver(
            voice_dir=VOICE_EMBEDDINGS_DIR,
            face_dir=FACE_ENCODINGS_DIR
        )
        self.user_memory = UserMemory()
        self.registry = UserRegistry()
        self.routines = RoutineTracker()

        # Session state
        self.current_user = None
        self.is_admin = False
        self.conversation = []
        self.chat = None
        self.last_voice_embedding = None
        self.executor = None

    def _build_system_prompt(self) -> str:
        """Build system prompt with user context."""
        parts = [
            "You are Menzi, a friendly voice assistant. Keep responses SHORT and conversational (2-3 sentences max unless more detail is needed).",
            "You are speaking aloud, so be natural and concise.",
        ]

        if self.current_user:
            parts.append(f"\nCurrent user: {self.current_user}")

            # Add user memories
            memories = self.user_memory.get_all(self.current_user)
            if memories:
                parts.append("\nKnown facts about this user:")
                for mem in memories:
                    parts.append(f"- {mem}")

            # Add location if available
            location = self.registry.get_location(self.current_user)
            if location:
                parts.append(f"\nUser's default location: {location}")

            # Check for routine suggestions
            suggestion = self.routines.get_suggestion(self.current_user)
            if suggestion:
                parts.append(f"\nProactive suggestion available: {suggestion}")

        # Add vision context if available
        vision_ctx = self.vision_context.get_context()
        if vision_ctx:
            parts.append(f"\nRecent visual context (what you saw): {vision_ctx}")

        # Tool guidance
        parts.append("\n\nAvailable tools:")
        parts.append("- remember: Save facts about the current user")
        parts.append("- save_note: Save general notes to knowledge base")
        parts.append("- search_notes: Search saved notes")
        parts.append("- web_search: Search the internet")
        parts.append("- get_weather: Get weather for a location")
        parts.append("- get_news: Get news headlines")
        parts.append("- see_camera: Look through camera to see environment")
        if self.is_admin:
            parts.append("- admin_command: System administration (admin only)")

        parts.append("\nAddress the user by name when appropriate. Be helpful but concise.")

        return "\n".join(parts)

    def _start_new_chat(self):
        """Start a new Gemini chat session."""
        system_prompt = self._build_system_prompt()
        self.chat = self.model.start_chat(
            history=[
                {"role": "user", "parts": [f"System context: {system_prompt}"]},
                {"role": "model", "parts": ["Understood. I'm ready to help."]}
            ]
        )
        self.conversation = []

    def _identify_user(self, audio: np.ndarray) -> Tuple[Optional[str], float, str]:
        """Identify user from voice (with optional face backup)."""
        # Try voice first
        user, confidence, method = self.identity.resolve(voice_audio=audio)

        # If voice is uncertain, try face
        if confidence < 0.75 and method == "voice":
            face_frame = self.camera.capture()
            if face_frame is not None:
                user, confidence, method = self.identity.resolve(
                    voice_audio=audio, face_image=face_frame
                )

        return user, confidence, method

    def _detect_user_handoff(self, audio: np.ndarray) -> bool:
        """Detect if a different user is now speaking."""
        if self.current_user is None:
            return False

        user, confidence, _ = self.identity.resolve(voice_audio=audio)

        # If we confidently identify a different user, it's a handoff
        if user and user != self.current_user and confidence >= 0.75:
            return True

        return False

    def _enroll_new_user(self, audio: np.ndarray) -> Optional[str]:
        """Enroll a new user with voice and optional face."""
        self.tts.speak_and_wait("I don't recognize your voice. What's your name?")

        name_text, _ = self.stt.listen()
        if not name_text:
            self.tts.speak_and_wait("Sorry, I didn't catch that. Let's try again later.")
            return None

        # Extract name (simple approach - take first word)
        name = name_text.strip().split()[0].lower()

        self.tts.speak_and_wait(f"Nice to meet you, {name}! Let me save your voice.")

        # Enroll voice
        self.identity.enroll_user(name, voice_audio=audio)

        # Try to capture face
        self.tts.speak_and_wait("I'll also try to remember your face. Please look at the camera.")
        face_frame = self.camera.capture()
        if face_frame is not None:
            try:
                self.identity.enroll_user(name, face_image=face_frame)
                self.tts.speak_and_wait("Got it! I've saved your voice and face.")
            except Exception:
                self.tts.speak_and_wait("I couldn't see your face clearly, but I've saved your voice.")
        else:
            self.tts.speak_and_wait("Camera not available, but I've saved your voice.")

        # Create user in registry
        is_admin = name.lower() == ADMIN_USER.lower()
        self.registry.create_user(name, admin=is_admin)

        return name

    def _switch_user(self, new_user: str, confidence: float):
        """Switch to a different user."""
        print(f"\n[USER SWITCH] {self.current_user} -> {new_user} (confidence: {confidence:.2f})")
        self.current_user = new_user
        self.is_admin = self.registry.is_admin(new_user)
        self._start_new_chat()
        self._init_executor()

        greeting = f"Hello {new_user}! How can I help you?"
        self.tts.speak_and_wait(greeting)

    def _init_executor(self):
        """Initialize tool executor with current user context."""
        self.executor = ToolExecutor(
            current_user=self.current_user or "unknown",
            is_admin=self.is_admin,
            camera=self.camera,
            vlm=self.vlm
        )

    def _execute_function_call(self, function_call) -> str:
        """Execute a Gemini function call and return result."""
        name = function_call.name
        args = dict(function_call.args)

        print(f"[TOOL] {name}: {args}")

        # Record routine patterns
        if self.current_user:
            if name == "get_weather":
                self.routines.record_query(self.current_user, "weather")
            elif name == "get_news":
                self.routines.record_query(self.current_user, "news")

        result = self.executor.execute(name, args)
        return result

    def _process_response(self, response) -> str:
        """Process Gemini response, handling any function calls."""
        # Check if there are function calls
        while True:
            # Check for function calls in the response
            function_calls = []
            if response.candidates and len(response.candidates) > 0 and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)

            if not function_calls:
                break

            # Execute each function call and send results back
            for fc in function_calls:
                result = self._execute_function_call(fc)

                # Send the function response back to Gemini
                response = self.chat.send_message(
                    genai.protos.Content(
                        parts=[
                            genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=fc.name,
                                    response={"result": result}
                                )
                            )
                        ]
                    )
                )

        # Extract text from final response
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content.parts:
            text_parts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            return " ".join(text_parts)

        return "I'm sorry, I couldn't generate a response."

    def _speak_response(self, text: str):
        """Speak response, breaking into sentences for smoother TTS."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 2:
                self.tts.speak(sentence)

        self.tts.wait()

    def _chat(self, user_input: str) -> str:
        """Send message to Gemini and get response."""
        # Record interaction time
        if self.current_user:
            self.routines.record_interaction(self.current_user)

        # Keep conversation within limits
        if len(self.conversation) >= MAX_MESSAGES:
            # Rebuild chat with fresh system prompt
            self._start_new_chat()

        # Send message
        response = self.chat.send_message(user_input)

        # Process response (handle function calls)
        response_text = self._process_response(response)

        # Store in conversation history
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": response_text})

        # Increment vision context message count
        self.vision_context.increment_messages()

        return response_text

    def run(self):
        """Main run loop."""
        print("=" * 50)
        print("Menzi Voice Assistant")
        print("=" * 50)
        print("Say 'quit' or 'exit' to stop\n")

        self.tts.speak_and_wait("Hello! I'm Menzi, your voice assistant.")

        try:
            while True:
                # Listen for speech
                text, audio = self.stt.listen()

                if text is None:
                    continue

                print(f"\nYou: {text}")

                # Check for exit commands
                if text.lower() in ["quit", "exit", "goodbye", "bye"]:
                    self.tts.speak_and_wait("Goodbye!")
                    break

                # Handle user identification/handoff
                if audio is not None:
                    if self.current_user is None:
                        # First interaction - identify or enroll
                        user, confidence, method = self._identify_user(audio)

                        if user:
                            print(f"[IDENTIFIED] {user} via {method} (confidence: {confidence:.2f})")
                            self.current_user = user
                            self.is_admin = self.registry.is_admin(user)
                            self._init_executor()
                            self._start_new_chat()

                            # Check for proactive suggestion
                            suggestion = self.routines.get_suggestion(user)
                            greeting = f"Hello {user}!"
                            if suggestion:
                                greeting += f" {suggestion}"
                            self.tts.speak(greeting)
                            self.tts.wait()
                        else:
                            # Unknown user - enroll
                            new_user = self._enroll_new_user(audio)
                            if new_user:
                                self.current_user = new_user
                                self.is_admin = self.registry.is_admin(new_user)
                                self._init_executor()
                                self._start_new_chat()
                            else:
                                continue
                    else:
                        # Check for user handoff
                        if self._detect_user_handoff(audio):
                            user, confidence, _ = self._identify_user(audio)
                            if user:
                                self._switch_user(user, confidence)

                # Initialize executor if needed
                if self.executor is None:
                    self._init_executor()

                # Initialize chat if needed
                if self.chat is None:
                    self._start_new_chat()

                # Get response from Gemini
                try:
                    response = self._chat(text)
                    print(f"\nMenzi: {response}")
                    self._speak_response(response)
                except Exception as e:
                    print(f"\n[ERROR] {e}")
                    self.tts.speak_and_wait("Sorry, I encountered an error. Please try again.")

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED]")
            self.tts.speak_and_wait("Goodbye!")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.tts.stop()
        self.camera.release()


def main():
    """Entry point."""
    try:
        menzi = Menzi()
        menzi.run()
    except ValueError as e:
        print(f"[CONFIGURATION ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
