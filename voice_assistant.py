import ollama
import chromadb
import json
import uuid
import re
import os
import threading
import queue
from datetime import datetime

USER_ID = "raghav"
MODEL = "qwen3:1.7b"
MAX_MESSAGES = 16
SUMMARIZE_THRESHOLD = 20
MEMORY_FILE = "./user_memories.json"

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    knowledge_collection = chroma_client.get_collection(name="knowledge_base")
except:
    knowledge_collection = chroma_client.create_collection(name="knowledge_base")

USER_MEMORIES = []

def load_memories_from_disk():
    global USER_MEMORIES
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            USER_MEMORIES = json.load(f)
    else:
        USER_MEMORIES = []
    return USER_MEMORIES

def save_memories_to_disk():
    with open(MEMORY_FILE, "w") as f:
        json.dump(USER_MEMORIES, f, indent=2)

def add_memory(content):
    global USER_MEMORIES
    content = content.strip()
    for existing in USER_MEMORIES:
        if content.lower() in existing.lower() or existing.lower() in content.lower():
            return False
    USER_MEMORIES.append(content)
    save_memories_to_disk()
    return True

def get_all_memories():
    return USER_MEMORIES

def clear_memories():
    global USER_MEMORIES
    USER_MEMORIES = []
    save_memories_to_disk()

tools = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a fact about the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember"}
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def add_to_knowledge_base(content, topic):
    doc_id = str(uuid.uuid4())
    knowledge_collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[{"topic": topic, "timestamp": datetime.now().isoformat()}]
    )

def web_search(query):
    try:
        import urllib.request
        import urllib.parse
        import html as html_module
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode("utf-8")
        results = []
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        for i in range(min(3, len(titles), len(snippets))):
            title = html_module.unescape(titles[i].strip())
            snippet = re.sub(r'<[^>]+>', '', snippets[i])
            snippet = html_module.unescape(snippet.strip())
            if title and snippet:
                results.append(f"{title}: {snippet}")
        if results:
            formatted = " | ".join(results)
            add_to_knowledge_base(f"Search '{query}': {formatted}", "web_search")
            return formatted
        return "No results found"
    except Exception as e:
        return f"Search failed: {str(e)}"

def execute_tool(tool_name, arguments):
    if tool_name == "remember":
        if add_memory(arguments["fact"]):
            return f"Remembered: {arguments['fact']}"
        return "Already known"
    elif tool_name == "web_search":
        return web_search(arguments["query"])
    return "Unknown tool"

def auto_extract_memories(text):
    text_lower = text.lower()
    patterns = [
        (r"(?:my name is|i'm|i am|call me) (\w+)", "User's name is {}"),
        (r"i(?:'m| am) (\d+)(?: years old)?", "User is {} years old"),
        (r"i live in ([^,.!?]+)", "User lives in {}"),
        (r"i work (?:at|for) ([^,.!?]+)", "User works at {}"),
        (r"i(?:'m| am) a ([^,.!?]+?)(?:\.|,|!|\?|$)", "User is a {}"),
        (r"i like ([^,.!?]+)", "User likes {}"),
        (r"i love ([^,.!?]+)", "User loves {}"),
        (r"i(?:'m| am) from ([^,.!?]+)", "User is from {}"),
    ]
    extracted = []
    for pattern, template in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                mem = template.format(*[m.strip() for m in match])
            else:
                mem = template.format(match.strip())
            if 10 < len(mem) < 100:
                extracted.append(mem)
    return extracted

def process_user_input(text):
    memories = auto_extract_memories(text)
    for mem in memories:
        if add_memory(mem):
            print(f"[REMEMBERED] {mem}")

def get_knowledge_context(query):
    try:
        count = knowledge_collection.count()
        if count == 0:
            return ""
        results = knowledge_collection.query(query_texts=[query], n_results=min(3, count))
        if results and results.get("documents") and results["documents"][0]:
            return "Context: " + " | ".join(results["documents"][0])
        return ""
    except:
        return ""

last_tool_context = {"results": None}

class TTSEngine:
    def __init__(self):
        import platform
        self.system = platform.system()
        self.queue = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.current_proc = None

    def clean_text(self, text):
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _worker(self):
        import subprocess
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    self.queue.task_done()
                    break

                text = self.clean_text(text)
                if text and len(text) > 2:
                    if self.system == "Darwin":
                        self.current_proc = subprocess.Popen(["say", "-r", "200", text])
                    elif self.system == "Linux":
                        self.current_proc = subprocess.Popen(["espeak", "-s", "160", text])

                    if self.current_proc:
                        self.current_proc.wait()
                        self.current_proc = None

                self.queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text):
        self.queue.put(text)

    def speak_and_wait(self, text):
        self.speak(text)
        self.wait()

    def wait(self):
        self.queue.join()

    def stop(self):
        self.running = False
        self.queue.put(None)
        if self.current_proc:
            self.current_proc.terminate()

class STTEngine:
    def __init__(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def listen(self):
        import speech_recognition as sr
        with sr.Microphone() as source:
            print("\n[LISTENING...]")
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                print("[PROCESSING...]")
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"[STT ERROR] {e}")
                return None

def chat_streaming(conversation, tts):
    global last_tool_context
    user_msg = conversation[-1]["content"]

    memories = get_all_memories()
    memory_str = ""
    if memories:
        memory_str = "KNOWN FACTS:\n" + "\n".join(f"- {m}" for m in memories)

    knowledge_str = get_knowledge_context(user_msg)
    recent_context = ""
    if last_tool_context["results"]:
        recent_context = f"\nRECENT: {last_tool_context['results'][:500]}\n"

    system = f"""You are Menzi, a voice assistant. Keep responses SHORT and conversational (2-3 sentences max).
{memory_str}
{recent_context}
{knowledge_str}

Tools: remember (save user facts), web_search (internet).
Address user by name if known. Be concise - this is spoken aloud."""

    full_messages = [{"role": "system", "content": system}] + conversation

    response = ollama.chat(model=MODEL, messages=full_messages, tools=tools)

    tool_results = []
    while response["message"].get("tool_calls"):
        for tc in response["message"]["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            print(f"[TOOL] {name}")
            result = execute_tool(name, args)
            tool_results.append(result)
            full_messages.append(response["message"])
            full_messages.append({"role": "tool", "content": result})
        response = ollama.chat(model=MODEL, messages=full_messages, tools=tools)

    if tool_results:
        last_tool_context["results"] = "\n".join(tool_results)

    print("\nMenzi: ", end="", flush=True)
    full_response = ""
    speech_buffer = ""
    in_think = False

    stream = ollama.chat(model=MODEL, messages=full_messages, stream=True, options={"num_predict": 400})

    for chunk in stream:
        token = chunk["message"]["content"]
        full_response += token

        if "<think>" in token:
            in_think = True
            token = token.replace("<think>", "")
        if "</think>" in token:
            in_think = False
            token = token.replace("</think>", "")
            continue
        if in_think:
            continue

        print(token, end="", flush=True)
        speech_buffer += token

        while True:
            match = re.search(r'^(.*?[.!?])\s*', speech_buffer)
            if match:
                sentence = match.group(1).strip()
                if len(sentence) > 3:
                    tts.speak(sentence)
                speech_buffer = speech_buffer[match.end():]
            else:
                break

    if speech_buffer.strip() and len(speech_buffer.strip()) > 3:
        tts.speak(speech_buffer.strip())

    print()
    tts.wait()

    clean = full_response
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    return clean

def main():
    print("=" * 50)
    print("Menzi - Voice Assistant")
    print("=" * 50)
    print("Say 'quit' or 'exit' to stop\n")

    load_memories_from_disk()
    if USER_MEMORIES:
        print(f"[LOADED] {len(USER_MEMORIES)} memories")

    try:
        tts = TTSEngine()
        stt = STTEngine()
    except Exception as e:
        print(f"[ERROR] Could not initialize voice engines: {e}")
        print("Install: pip install pyttsx3 SpeechRecognition pyaudio")
        return

    tts.speak_and_wait("Hello! I'm Menzi, your voice assistant. How can I help?")

    conversation = []

    while True:
        text = stt.listen()

        if text is None:
            continue

        print(f"\nYou: {text}")

        if text.lower() in ["quit", "exit", "goodbye", "bye"]:
            tts.speak_and_wait("Goodbye!")
            break

        if text.lower() == "clear memory":
            clear_memories()
            tts.speak_and_wait("Memory cleared.")
            continue

        process_user_input(text)
        conversation.append({"role": "user", "content": text})

        if len(conversation) > MAX_MESSAGES:
            conversation = conversation[-MAX_MESSAGES:]

        try:
            response = chat_streaming(conversation, tts)
            conversation.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\n[ERROR] {e}")
            tts.speak_and_wait("Sorry, I encountered an error.")

    tts.stop()

if __name__ == "__main__":
    main()
