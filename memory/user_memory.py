import os
import json
from typing import List

class UserMemory:
    def __init__(self, memories_dir: str = None):
        if memories_dir is None:
            from config import MEMORIES_DIR
            memories_dir = MEMORIES_DIR
        self.memories_dir = memories_dir
        os.makedirs(self.memories_dir, exist_ok=True)

    def _get_path(self, user: str) -> str:
        return os.path.join(self.memories_dir, f"{user.lower()}.json")

    def _load(self, user: str) -> List[str]:
        path = self._get_path(user)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save(self, user: str, memories: List[str]):
        path = self._get_path(user)
        with open(path, "w") as f:
            json.dump(memories, f, indent=2)

    def add(self, user: str, content: str) -> bool:
        content = content.strip()
        memories = self._load(user)
        for existing in memories:
            if content.lower() in existing.lower() or existing.lower() in content.lower():
                return False
        memories.append(content)
        self._save(user, memories)
        return True

    def get_all(self, user: str) -> List[str]:
        return self._load(user)

    def clear(self, user: str):
        self._save(user, [])
