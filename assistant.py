import ollama
import chromadb
from mem0 import Memory
import json
import uuid
import sys
from datetime import datetime

USER_ID = "raghav"
MODEL = "qwen3:1.7b"
MAX_MESSAGES = 16
SUMMARIZE_THRESHOLD = 20

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    knowledge_collection = chroma_client.get_collection(name="knowledge_base")
except:
    knowledge_collection = chroma_client.create_collection(name="knowledge_base")

mem0_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": MODEL,
            "temperature": 0.7,
            "max_tokens": 4000,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "mem0_memories",
            "path": "./mem0_db"
        }
    }
}

memory = Memory.from_config(mem0_config)

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_user_memory",
            "description": "Add a personal memory or fact about the user (preferences, personal info, habits, relationships, etc)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory or fact to store about the user"
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_memories",
            "description": "Retrieve relevant memories about the user based on a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search for relevant user memories"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_knowledge_base",
            "description": "Store information in the knowledge base (projects, research, notes, news, things being worked on)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic or category for this information"
                    }
                },
                "required": ["content", "topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for relevant context and information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search the knowledge base"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def add_user_memory(content):
    memory.add(content, user_id=USER_ID)
    return f"Memory stored: {content}"

def get_user_memories(query):
    results = memory.search(query, user_id=USER_ID)
    if not results or len(results) == 0:
        return "No relevant memories found."
    memories = []
    for r in results:
        if isinstance(r, dict) and "memory" in r:
            memories.append(r["memory"])
        elif isinstance(r, dict) and "text" in r:
            memories.append(r["text"])
        elif isinstance(r, str):
            memories.append(r)
    if not memories:
        return "No relevant memories found."
    return "User memories: " + " | ".join(memories)

def add_to_knowledge_base(content, topic):
    doc_id = str(uuid.uuid4())
    knowledge_collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[{"topic": topic, "timestamp": datetime.now().isoformat()}]
    )
    return f"Added to knowledge base under topic '{topic}'"

def search_knowledge_base(query):
    try:
        count = knowledge_collection.count()
        if count == 0:
            return "Knowledge base is empty."
        results = knowledge_collection.query(query_texts=[query], n_results=min(5, count))
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant information found in knowledge base."
        docs = results["documents"][0]
        return "Knowledge base results: " + " | ".join(docs)
    except Exception as e:
        return f"Knowledge base search failed: {str(e)}"

def web_search(query):
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": f"/no_think Search the web and answer: {query}"}],
            options={"num_predict": 300}
        )
        content = response["message"]["content"]
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()
        return f"Web search results for '{query}': {content}"
    except Exception as e:
        return f"Web search failed: {str(e)}"

def execute_tool(tool_name, arguments):
    if tool_name == "add_user_memory":
        return add_user_memory(arguments["content"])
    elif tool_name == "get_user_memories":
        return get_user_memories(arguments["query"])
    elif tool_name == "add_to_knowledge_base":
        return add_to_knowledge_base(arguments["content"], arguments["topic"])
    elif tool_name == "search_knowledge_base":
        return search_knowledge_base(arguments["query"])
    elif tool_name == "web_search":
        return web_search(arguments["query"])
    return "Unknown tool"

def get_memory_context(query):
    try:
        results = memory.search(query, user_id=USER_ID)
        if not results or len(results) == 0:
            return ""
        memories = []
        for r in results[:3]:
            if isinstance(r, dict) and "memory" in r:
                memories.append(r["memory"])
            elif isinstance(r, dict) and "text" in r:
                memories.append(r["text"])
            elif isinstance(r, str):
                memories.append(r)
        if memories:
            return "User context: " + " | ".join(memories)
        return ""
    except:
        return ""

def get_knowledge_context(query):
    try:
        count = knowledge_collection.count()
        if count == 0:
            return ""
        results = knowledge_collection.query(query_texts=[query], n_results=min(3, count))
        if results and results.get("documents") and results["documents"][0]:
            return "Relevant knowledge: " + " | ".join(results["documents"][0])
        return ""
    except:
        return ""

def is_complex_query(text):
    complex_keywords = ["why", "how", "explain", "analyze", "compare", "difference", "should", "best", "recommend", "think about", "help me understand", "what if", "plan", "strategy"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in complex_keywords)

def estimate_tokens(messages):
    total = 0
    for msg in messages:
        total += len(msg.get("content", "")) // 4
    return total

def summarize_conversation(messages):
    if len(messages) < 4:
        return None
    conversation_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content}\n"

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": f"/no_think Summarize this conversation in 2-3 sentences, capturing key topics, decisions, and any important information about the user:\n\n{conversation_text}"
            }],
            options={"num_predict": 200}
        )
        summary = response["message"]["content"]
        if "<think>" in summary:
            summary = summary.split("</think>")[-1].strip()
        return summary
    except:
        return None

def manage_context(conversation):
    if len(conversation) < SUMMARIZE_THRESHOLD:
        return conversation

    print("\n[CONTEXT] Summarizing old conversation...")
    messages_to_summarize = conversation[:-MAX_MESSAGES]
    messages_to_keep = conversation[-MAX_MESSAGES:]

    summary = summarize_conversation(messages_to_summarize)
    if summary:
        add_to_knowledge_base(
            f"Conversation summary: {summary}",
            "conversation_history"
        )
        print(f"[CONTEXT] Saved summary to knowledge base")

    print(f"[CONTEXT] Keeping last {len(messages_to_keep)} messages")
    return messages_to_keep

def stream_response(full_messages, use_thinking):
    print("\nMenzi: ", end="", flush=True)

    full_response = ""
    in_think_block = False
    showed_thinking = False

    stream = ollama.chat(
        model=MODEL,
        messages=full_messages,
        stream=True,
        options={"num_predict": 1000}
    )

    buffer = ""
    for chunk in stream:
        token = chunk["message"]["content"]
        buffer += token
        full_response += token

        while True:
            if not in_think_block:
                if "<think>" in buffer:
                    pre_think = buffer.split("<think>")[0]
                    if pre_think:
                        print(pre_think, end="", flush=True)
                    if not showed_thinking:
                        print("[thinking...]", end="", flush=True)
                        showed_thinking = True
                    buffer = buffer.split("<think>", 1)[1]
                    in_think_block = True
                else:
                    safe_output = buffer[:-10] if len(buffer) > 10 else ""
                    if safe_output:
                        print(safe_output, end="", flush=True)
                        buffer = buffer[len(safe_output):]
                    break
            else:
                if "</think>" in buffer:
                    buffer = buffer.split("</think>", 1)[1]
                    in_think_block = False
                else:
                    break

    if buffer and not in_think_block:
        print(buffer, end="", flush=True)

    print()
    return full_response

def chat(messages):
    user_message = messages[-1]["content"]
    memory_context = get_memory_context(user_message)
    knowledge_context = get_knowledge_context(user_message)

    use_thinking = is_complex_query(user_message)
    thinking_instruction = "" if use_thinking else "For simple queries, respond directly without extensive reasoning. "

    system_message = f"""You are Menzi, an autonomous AI assistant for {USER_ID}. {thinking_instruction}You have access to tools and should use them proactively:

- add_user_memory: When user shares personal info, preferences, or facts about themselves
- get_user_memories: When you need to recall something about the user
- add_to_knowledge_base: When user discusses projects, research, news, or things they're working on
- search_knowledge_base: When you need context from previous conversations or stored knowledge
- web_search: When you need current information from the internet

{memory_context}
{knowledge_context}

Be proactive about storing relevant information. Respond naturally and helpfully. Keep responses concise."""

    full_messages = [{"role": "system", "content": system_message}] + messages

    response = ollama.chat(
        model=MODEL,
        messages=full_messages,
        tools=tools
    )

    while response["message"].get("tool_calls"):
        for tool_call in response["message"]["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            print(f"\n[TOOL] {tool_name}: {arguments}")
            result = execute_tool(tool_name, arguments)
            print(f"[RESULT] {result}")
            full_messages.append(response["message"])
            full_messages.append({"role": "tool", "content": result})

        response = ollama.chat(
            model=MODEL,
            messages=full_messages,
            tools=tools
        )

    if response["message"].get("tool_calls"):
        return response["message"]["content"]

    full_messages_for_stream = full_messages.copy()
    if response["message"]["content"]:
        return stream_response(full_messages_for_stream, use_thinking)

    return stream_response(full_messages, use_thinking)

def clean_response(text):
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1:
            return parts[-1].strip()
    if "<think>" in text:
        parts = text.split("<think>")
        return parts[0].strip()
    return text.strip()

def main():
    print("=" * 50)
    print("Menzi - Autonomous Home Assistant")
    print("=" * 50)
    print("Commands: 'quit' to exit, 'clear' to reset conversation\n")

    conversation = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            if conversation:
                summary = summarize_conversation(conversation)
                if summary:
                    add_to_knowledge_base(f"Conversation summary: {summary}", "conversation_history")
                    print("[CONTEXT] Saved conversation summary")
            conversation = []
            print("[CONTEXT] Conversation cleared. Long-term memory preserved.")
            continue
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        try:
            conversation = manage_context(conversation)
            response = chat(conversation)
            if response:
                response = clean_response(response)
                conversation.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
