import requests
import re
import html

DUCKDUCKGO_NEWS_URL = "https://html.duckduckgo.com/html/"

def get_news(topic: str = "world") -> str:
    """Get news headlines for a topic using DuckDuckGo."""
    try:
        query = f"{topic} news today"
        response = requests.get(
            DUCKDUCKGO_NEWS_URL,
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        html_content = response.text

        # Extract titles and snippets
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html_content)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_content, re.DOTALL)

        results = []
        for i in range(min(5, len(titles), len(snippets))):
            title = html.unescape(titles[i].strip())
            snippet = re.sub(r'<[^>]+>', '', snippets[i])
            snippet = html.unescape(snippet.strip())
            if title and snippet:
                results.append(f"- {title}: {snippet[:150]}")

        if results:
            return f"Headlines for {topic}:\n" + "\n".join(results)

        return f"No news found for {topic}."

    except Exception as e:
        return f"News error: {str(e)}"
