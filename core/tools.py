"""Tool definitions for LLM function calling.

This module defines all available tools/functions that Menzi can use
during conversations. Supports both Ollama and Gemini formats.
"""

from typing import List, Dict, Any

# Tool definitions in Gemini-compatible format
# These will be converted to genai.protos.Tool format when used

TOOL_DEFINITIONS = [
    {
        "name": "remember",
        "description": "Store a fact about the current user for future reference. Use when user shares personal information like preferences, facts about themselves, or things to remember.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to remember about the user"
                }
            },
            "required": ["fact"]
        }
    },
    {
        "name": "save_note",
        "description": "Save information to the knowledge base for later retrieval. Use for general notes, not user-specific facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to save"
                },
                "topic": {
                    "type": "string",
                    "description": "Topic category for organization"
                }
            },
            "required": ["content", "topic"]
        }
    },
    {
        "name": "search_notes",
        "description": "Search the knowledge base for saved information.",
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
    },
    {
        "name": "web_search",
        "description": "Search the internet for current information. Use when user needs up-to-date info.",
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
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_news",
        "description": "Get news headlines for a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "News topic"
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "see_camera",
        "description": "Look through the camera to see what's happening. Use when user asks 'what am I doing', 'what do you see', 'look at this', or needs visual context about their environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for or describe in the scene"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "admin_command",
        "description": "Execute admin-only system commands. Only works for admin users. Use for system management like listing users, viewing memories, or system status.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_users", "delete_user", "view_memories", "clear_memories", "interaction_log", "reset_routines", "system_status"],
                    "description": "The admin action to perform"
                },
                "target_user": {
                    "type": "string",
                    "description": "Username for user-specific actions (optional)"
                }
            },
            "required": ["action"]
        }
    }
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions in Gemini-compatible format.

    Returns:
        List of tool definition dictionaries with name, description, and parameters.
    """
    return TOOL_DEFINITIONS


def get_ollama_tools() -> List[Dict[str, Any]]:
    """Convert tool definitions to Ollama's tool format.

    Returns tools in the format expected by Ollama's chat API with
    function calling support.

    Returns:
        List of tool definitions with 'type': 'function' wrapper.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        }
        for tool in TOOL_DEFINITIONS
    ]


def get_gemini_tools():
    """Convert tool definitions to Gemini's Tool format for the API.

    This function creates proper FunctionDeclaration objects that can be
    passed to the Gemini API for function calling.

    Returns:
        A genai.types.Tool object with all function declarations,
        or the raw definitions if google-generativeai is not installed.
    """
    try:
        import google.generativeai as genai
        from google.generativeai.types import Tool, FunctionDeclaration

        function_declarations = []
        for tool in TOOL_DEFINITIONS:
            func_decl = FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"]
            )
            function_declarations.append(func_decl)

        return Tool(function_declarations=function_declarations)
    except ImportError:
        # Return raw definitions if genai not installed
        return TOOL_DEFINITIONS


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get a specific tool definition by name.

    Args:
        name: The name of the tool to retrieve.

    Returns:
        The tool definition dictionary, or None if not found.
    """
    for tool in TOOL_DEFINITIONS:
        if tool["name"] == name:
            return tool
    return None


def get_tool_names() -> List[str]:
    """Get a list of all available tool names.

    Returns:
        List of tool name strings.
    """
    return [tool["name"] for tool in TOOL_DEFINITIONS]
