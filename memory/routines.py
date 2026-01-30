import os
import json
from datetime import datetime
from typing import Optional, Dict, List

class RoutineTracker:
    def __init__(self, routines_file: str = None):
        if routines_file is None:
            from config import ROUTINES_FILE
            routines_file = ROUTINES_FILE
        self.routines_file = routines_file
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.routines_file):
            os.makedirs(os.path.dirname(self.routines_file), exist_ok=True)
            with open(self.routines_file, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict:
        with open(self.routines_file, "r") as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.routines_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_user_data(self, user: str) -> Dict:
        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {
                "interaction_times": [],
                "common_queries": [],
                "recent_topics": []
            }
            self._save(data)
        return data[user]

    def record_interaction(self, user: str, hour: int = None, day: str = None):
        if hour is None:
            hour = datetime.now().hour
        if day is None:
            day = datetime.now().strftime("%A").lower()

        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {"interaction_times": [], "common_queries": [], "recent_topics": []}

        times = data[user]["interaction_times"]
        day_type = "weekend" if day in ["saturday", "sunday"] else "weekday"

        # Find or create time bucket
        found = False
        for t in times:
            if t["hour"] == hour and t["day"] == day_type:
                t["count"] += 1
                found = True
                break
        if not found:
            times.append({"hour": hour, "day": day_type, "count": 1})

        self._save(data)

    def record_query(self, user: str, pattern: str, hour: int = None):
        if hour is None:
            hour = datetime.now().hour

        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {"interaction_times": [], "common_queries": [], "recent_topics": []}

        time_bucket = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        queries = data[user]["common_queries"]

        found = False
        for q in queries:
            if q["pattern"] == pattern and q["time"] == time_bucket:
                q["count"] += 1
                found = True
                break
        if not found:
            queries.append({"pattern": pattern, "time": time_bucket, "count": 1})

        self._save(data)

    def record_topic(self, user: str, topic: str):
        data = self._load()
        user = user.lower()
        if user not in data:
            data[user] = {"interaction_times": [], "common_queries": [], "recent_topics": []}

        topics = data[user]["recent_topics"]
        if topic in topics:
            topics.remove(topic)
        topics.insert(0, topic)
        data[user]["recent_topics"] = topics[:10]  # Keep last 10
        self._save(data)

    def get_patterns(self, user: str) -> Dict:
        return self._get_user_data(user)

    def get_suggestion(self, user: str, hour: int = None) -> Optional[str]:
        if hour is None:
            hour = datetime.now().hour

        patterns = self._get_user_data(user)
        time_bucket = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"

        # Find frequent queries for this time
        for q in patterns.get("common_queries", []):
            if q["time"] == time_bucket and q["count"] >= 3:
                if q["pattern"] == "weather":
                    return "Would you like a weather update?"
                elif q["pattern"] == "news":
                    return "Want me to fetch today's news?"
        return None

    def clear(self, user: str):
        data = self._load()
        user = user.lower()
        if user in data:
            del data[user]
            self._save(data)
