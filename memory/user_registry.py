import os
import json
from datetime import datetime
from typing import Optional, List, Dict

class UserRegistry:
    def __init__(self, users_file: str = None):
        if users_file is None:
            from config import USERS_FILE
            users_file = USERS_FILE
        self.users_file = users_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.users_file):
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        with open(self.users_file, "r") as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.users_file, "w") as f:
            json.dump(data, f, indent=2)

    def create_user(self, name: str, admin: bool = False, location: str = None):
        data = self._load()
        name = name.lower()
        if name not in data:
            data[name] = {
                "admin": admin,
                "created_at": datetime.now().isoformat(),
                "default_location": location
            }
            self._save(data)

    def user_exists(self, name: str) -> bool:
        return name.lower() in self._load()

    def is_admin(self, name: str) -> bool:
        data = self._load()
        user = data.get(name.lower(), {})
        return user.get("admin", False)

    def list_users(self) -> List[str]:
        return list(self._load().keys())

    def delete_user(self, name: str):
        data = self._load()
        name = name.lower()
        if name in data:
            del data[name]
            self._save(data)

    def get_location(self, name: str) -> Optional[str]:
        data = self._load()
        user = data.get(name.lower(), {})
        return user.get("default_location")

    def set_location(self, name: str, location: str):
        data = self._load()
        name = name.lower()
        if name in data:
            data[name]["default_location"] = location
            self._save(data)
