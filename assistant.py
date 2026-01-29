import ollama
import chromadb
import json
import uuid
import re
import os
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
            "name": "save_note",
            "description": "Save information to knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The information"},
                    "topic": {"type": "string", "description": "Topic category"}
                },
                "required": ["content", "topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the knowledge base",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
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
    return f"Saved to knowledge base"

def search_knowledge_base(query):
    try:
        count = knowledge_collection.count()
        if count == 0:
            return "Knowledge base is empty."
        results = knowledge_collection.query(query_texts=[query], n_results=min(5, count))
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No results found."
        return " | ".join(results["documents"][0])
    except Exception as e:
        return f"Search failed: {str(e)}"

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

        for i in range(min(5, len(titles), len(snippets))):
            title = html_module.unescape(titles[i].strip())
            snippet = re.sub(r'<[^>]+>', '', snippets[i])
            snippet = html_module.unescape(snippet.strip())
            if title and snippet:
                results.append(f"**{title}**: {snippet}")

        if results:
            formatted = "\n".join(results)
            add_to_knowledge_base(f"Web search for '{query}':\n{formatted}", "web_search")
            return formatted
        return "No results found"
    except Exception as e:
        return f"Search failed: {str(e)}"

def execute_tool(tool_name, arguments):
    if tool_name == "remember":
        if add_memory(arguments["fact"]):
            return f"Remembered: {arguments['fact']}"
        return "Already known"
    elif tool_name == "save_note":
        return add_to_knowledge_base(arguments["content"], arguments["topic"])
    elif tool_name == "search_notes":
        return search_knowledge_base(arguments["query"])
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
        (r"my job is ([^,.!?]+)", "User's job is {}"),
        (r"i like ([^,.!?]+)", "User likes {}"),
        (r"i love ([^,.!?]+)", "User loves {}"),
        (r"i hate ([^,.!?]+)", "User hates {}"),
        (r"my favorite (\w+) is ([^,.!?]+)", "User's favorite {} is {}"),
        (r"i(?:'m| am) from ([^,.!?]+)", "User is from {}"),
        (r"i speak ([^,.!?]+)", "User speaks {}"),
        (r"my birthday is ([^,.!?]+)", "User's birthday is {}"),
        (r"i(?:'m| am) studying ([^,.!?]+)", "User is studying {}"),
        (r"i study ([^,.!?]+)", "User studies {}"),
        (r"i go to ([^,.!?]+)", "User goes to {}"),
        (r"i have (?:a |an )?([^,.!?]+)", "User has {}"),
    ]

    extracted = []
    for pattern, template in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                mem = template.format(*[m.strip() for m in match])
            else:
                mem = template.format(match.strip())
            if len(mem) > 10 and len(mem) < 100:
                extracted.append(mem)
    return extracted

def auto_extract_knowledge(text):
    text_lower = text.lower()
    patterns = [
        (r"i(?:'m| am) working on ([^,.!?]+)", "projects"),
        (r"i(?:'m| am) building ([^,.!?]+)", "projects"),
        (r"i(?:'m| am) learning ([^,.!?]+)", "learning"),
        (r"i(?:'m| am) reading ([^,.!?]+)", "reading"),
    ]

    extracted = []
    for pattern, topic in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            content = f"User is working on: {match.strip()}"
            if len(content) > 10:
                extracted.append((content, topic))
    return extracted

def process_user_input(text):
    memories = auto_extract_memories(text)
    for mem in memories:
        if add_memory(mem):
            print(f"[REMEMBERED] {mem}")

    knowledge = auto_extract_knowledge(text)
    for content, topic in knowledge:
        add_to_knowledge_base(content, topic)
        print(f"[SAVED] {content}")

def get_knowledge_context(query):
    try:
        count = knowledge_collection.count()
        if count == 0:
            return ""
        results = knowledge_collection.query(query_texts=[query], n_results=min(3, count))
        if results and results.get("documents") and results["documents"][0]:
            return "Recent notes: " + " | ".join(results["documents"][0])
        return ""
    except:
        return ""

def summarize_conversation(messages):
    if len(messages) < 4:
        return None
    text = "\n".join([f"{m['role']}: {m.get('content', '')}" for m in messages])
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": f"/no_think Summarize in 2 sentences:\n{text}"}],
            options={"num_predict": 100}
        )
        summary = response["message"]["content"]
        if "</think>" in summary:
            summary = summary.split("</think>")[-1]
        return summary.strip()
    except:
        return None

def manage_context(conversation):
    if len(conversation) < SUMMARIZE_THRESHOLD:
        return conversation
    print("\n[CONTEXT] Compressing conversation...")
    old = conversation[:-MAX_MESSAGES]
    keep = conversation[-MAX_MESSAGES:]
    summary = summarize_conversation(old)
    if summary:
        add_to_knowledge_base(f"Conversation: {summary}", "history")
    return keep

def stream_chat(messages):
    print("\nMenzi: ", end="", flush=True)

    full_response = ""
    in_think = False
    buffer = ""

    stream = ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        options={"num_predict": 800}
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        buffer += token
        full_response += token

        while True:
            if not in_think:
                if "<think>" in buffer:
                    pre = buffer.split("<think>")[0]
                    print(pre, end="", flush=True)
                    print("[thinking...", end="", flush=True)
                    buffer = buffer.split("<think>", 1)[1]
                    in_think = True
                else:
                    if len(buffer) > 8:
                        print(buffer[:-8], end="", flush=True)
                        buffer = buffer[-8:]
                    break
            else:
                if "</think>" in buffer:
                    print("]", end="", flush=True)
                    buffer = buffer.split("</think>", 1)[1]
                    in_think = False
                else:
                    break

    if buffer and not in_think:
        print(buffer, end="", flush=True)
    print()

    if "</think>" in full_response:
        return full_response.split("</think>")[-1].strip()
    return full_response.strip()

last_tool_context = {"results": None}

def chat(conversation):
    global last_tool_context
    user_msg = conversation[-1]["content"]

    memories = get_all_memories()
    memory_str = ""
    if memories:
        memory_str = "KNOWN FACTS ABOUT USER:\n" + "\n".join(f"- {m}" for m in memories)

    knowledge_str = get_knowledge_context(user_msg)

    recent_context = ""
    if last_tool_context["results"]:
        recent_context = f"\nRECENT SEARCH/TOOL RESULTS (for follow-up questions):\n{last_tool_context['results']}\n"

    system = f"""You are Menzi, a conversational AI assistant. You maintain context across the conversation.

{memory_str}
{recent_context}
{knowledge_str}

Tools: remember (save facts about user), save_note (save info), search_notes (search saved), web_search (internet).

Guidelines:
- Address user by name if known
- When presenting search results, number them for easy reference
- If user says "tell me more about #2" or "the first one", use the recent context above
- Be conversational, not robotic
- Keep responses concise but informative"""

    full_messages = [{"role": "system", "content": system}] + conversation

    response = ollama.chat(
        model=MODEL,
        messages=full_messages,
        tools=tools
    )

    tool_results = []
    while response["message"].get("tool_calls"):
        for tc in response["message"]["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            print(f"\n[TOOL] {name}: {args}")
            result = execute_tool(name, args)
            print(f"[RESULT] {result[:200]}..." if len(result) > 200 else f"[RESULT] {result}")
            tool_results.append(result)
            full_messages.append(response["message"])
            full_messages.append({"role": "tool", "content": result})

        response = ollama.chat(model=MODEL, messages=full_messages, tools=tools)

    if tool_results:
        last_tool_context["results"] = "\n---\n".join(tool_results)

    if response["message"].get("content"):
        content = response["message"]["content"]
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        print(f"\nMenzi: {content}")
        return content

    return stream_chat(full_messages)

def main():
    print("=" * 50)
    print("Menzi - Autonomous Home Assistant")
    print("=" * 50)
    print("Commands: quit, clear, memories, forget\n")

    load_memories_from_disk()
    if USER_MEMORIES:
        print(f"[LOADED] {len(USER_MEMORIES)} memories")

    conversation = []

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            conversation = []
            print("[CLEARED] Conversation reset. Memories preserved.")
            continue
        if user_input.lower() == "memories":
            mems = get_all_memories()
            if mems:
                print("\n[MEMORIES]")
                for i, m in enumerate(mems, 1):
                    print(f"  {i}. {m}")
            else:
                print("[MEMORIES] None stored yet.")
            continue
        if user_input.lower() == "forget":
            clear_memories()
            print("[CLEARED] All memories deleted.")
            continue

        process_user_input(user_input)
        conversation.append({"role": "user", "content": user_input})
        conversation = manage_context(conversation)

        try:
            response = chat(conversation)
            conversation.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
