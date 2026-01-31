import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List

# Try to import chromadb, provide fallback if not available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from memory.user_memory import UserMemory
from memory.user_registry import UserRegistry
from memory.routines import RoutineTracker
from services.weather import get_weather as weather_service
from services.news import get_news as news_service


class SimpleKnowledgeBase:
    """Simple in-memory knowledge base fallback when chromadb is unavailable."""

    def __init__(self):
        self._documents: List[Dict] = []

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self._documents.append({"id": doc_id, "document": doc, "metadata": meta})

    def count(self) -> int:
        return len(self._documents)

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict:
        # Simple substring matching for fallback
        query = query_texts[0].lower() if query_texts else ""
        matches = []
        for doc in self._documents:
            if query in doc["document"].lower():
                matches.append(doc["document"])
        return {"documents": [matches[:n_results]]}


class ToolExecutor:
    def __init__(
        self,
        current_user: str,
        is_admin: bool,
        data_dir: str = None,
        camera=None,
        vlm=None,
        face_tracker=None,
        on_user_change=None  # Callback when user is set/changed
    ):
        self.current_user = current_user
        self.is_admin = is_admin
        self.on_user_change = on_user_change

        if data_dir is None:
            from config import DATA_DIR, CHROMA_DB_PATH
            data_dir = DATA_DIR
            chroma_path = CHROMA_DB_PATH
        else:
            chroma_path = os.path.join(data_dir, "chroma_db")

        self.user_memory = UserMemory(memories_dir=os.path.join(data_dir, "memories"))
        self.registry = UserRegistry(users_file=os.path.join(data_dir, "users.json"))
        self.routines = RoutineTracker(routines_file=os.path.join(data_dir, "routines.json"))

        # Use chromadb if available, otherwise fallback to simple in-memory storage
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            try:
                self.knowledge_collection = self.chroma_client.get_collection("knowledge_base")
            except:
                self.knowledge_collection = self.chroma_client.create_collection("knowledge_base")
        else:
            self.chroma_client = None
            self.knowledge_collection = SimpleKnowledgeBase()

        self.camera = camera
        self.vlm = vlm
        self.face_tracker = face_tracker

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if tool_name == "set_user":
                return self._set_user(arguments["name"])
            elif tool_name == "remember":
                return self._remember(arguments["fact"])
            elif tool_name == "save_note":
                return self._save_note(arguments["content"], arguments["topic"])
            elif tool_name == "search_notes":
                return self._search_notes(arguments["query"])
            elif tool_name == "web_search":
                return self._web_search(arguments["query"])
            elif tool_name == "get_weather":
                return self._get_weather(arguments["location"])
            elif tool_name == "get_news":
                return self._get_news(arguments["topic"])
            elif tool_name == "see_camera":
                return self._see_camera(arguments["query"])
            elif tool_name == "admin_command":
                return self._admin_command(arguments["action"], arguments.get("target_user"))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {str(e)}"

    def _set_user(self, name: str) -> str:
        """Set the current user's name."""
        import re
        name = name.strip().lower()
        name = re.sub(r'[^a-z]', '', name)

        if len(name) < 2:
            return "Invalid name."

        # Create user if doesn't exist
        if not self.registry.get_user(name):
            from config import ADMIN_USER
            is_admin = name == ADMIN_USER.lower()
            self.registry.create_user(name, admin=is_admin)

        # Update current user
        old_user = self.current_user
        self.current_user = name
        self.is_admin = self.registry.is_admin(name)

        # Notify callback
        if self.on_user_change:
            self.on_user_change(name)

        return f"User set to {name}."

    def _remember(self, fact: str) -> str:
        if self.user_memory.add(self.current_user, fact):
            return f"Remembered: {fact}"
        return "Already known."

    def _save_note(self, content: str, topic: str) -> str:
        doc_id = str(uuid.uuid4())
        self.knowledge_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"topic": topic, "timestamp": datetime.now().isoformat(), "user": self.current_user}]
        )
        return f"Saved to knowledge base under '{topic}'."

    def _search_notes(self, query: str) -> str:
        try:
            count = self.knowledge_collection.count()
            if count == 0:
                return "Knowledge base is empty."
            results = self.knowledge_collection.query(query_texts=[query], n_results=min(5, count))
            if results and results.get("documents") and results["documents"][0]:
                return " | ".join(results["documents"][0])
            return "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    def _web_search(self, query: str) -> str:
        import requests
        import re
        import html
        try:
            url = "https://html.duckduckgo.com/html/"
            response = requests.get(url, params={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            html_content = response.text

            titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html_content)
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_content, re.DOTALL)

            results = []
            for i in range(min(5, len(titles), len(snippets))):
                title = html.unescape(titles[i].strip())
                snippet = re.sub(r'<[^>]+>', '', snippets[i])
                snippet = html.unescape(snippet.strip())
                if title and snippet:
                    results.append(f"**{title}**: {snippet[:200]}")

            if results:
                return "\n".join(results)
            return "No results found."
        except Exception as e:
            return f"Search error: {str(e)}"

    def _get_weather(self, location: str) -> str:
        return weather_service(location)

    def _get_news(self, topic: str) -> str:
        return news_service(topic)

    def _see_camera(self, query: str) -> str:
        if self.camera is None or self.vlm is None:
            return "Camera not available."

        frame = self.camera.capture()
        if frame is None:
            return "Could not capture image from camera."

        try:
            # Enhance query with face annotations if available
            if self.face_tracker:
                face_context = self.face_tracker.get_faces_for_vlm()
                if face_context:
                    enhanced_query = f"{face_context}\n\nDescribe what each person is doing, with emphasis on the speaker. {query}"
                else:
                    enhanced_query = query
            else:
                enhanced_query = query

            result = self.vlm.analyze(frame, enhanced_query)
            return result
        except Exception as e:
            return f"Vision error: {str(e)}"

    def _admin_command(self, action: str, target_user: Optional[str] = None) -> str:
        if not self.is_admin:
            return "Sorry, only the admin can do that."

        if action == "list_users":
            users = self.registry.list_users()
            return f"Enrolled users: {', '.join(users)}" if users else "No users enrolled."

        elif action == "delete_user":
            if not target_user:
                return "Please specify which user to delete."
            if target_user.lower() == self.current_user.lower():
                return "Cannot delete yourself."
            self.registry.delete_user(target_user)
            self.user_memory.clear(target_user)
            return f"Deleted user: {target_user}"

        elif action == "view_memories":
            if not target_user:
                return "Please specify which user's memories to view."
            memories = self.user_memory.get_all(target_user)
            return f"{target_user}'s memories:\n" + "\n".join(f"- {m}" for m in memories) if memories else f"No memories for {target_user}."

        elif action == "clear_memories":
            if not target_user:
                return "Please specify which user's memories to clear."
            self.user_memory.clear(target_user)
            return f"Cleared memories for {target_user}."

        elif action == "reset_routines":
            if target_user:
                self.routines.clear(target_user)
                return f"Reset routines for {target_user}."
            return "Routines reset."

        elif action == "system_status":
            user_count = len(self.registry.list_users())
            note_count = self.knowledge_collection.count()
            return f"System Status:\n- Users: {user_count}\n- Knowledge base entries: {note_count}\n- Camera: {'Available' if self.camera else 'Not configured'}"

        return f"Unknown admin action: {action}"
