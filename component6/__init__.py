"""
Component 6: Gemini Brain Integration
Complete conversational AI system using Google Gemini with intelligent context assembly.
"""

from .conversation_manager import ConversationManager
from .memory_retriever import MemoryRetriever
from .context_assembler import ContextAssembler
from .gemini_client import GeminiClient

# Import the new modules individually to avoid null byte issues
try:
    from .personality_engine import PersonalityEngine
except SyntaxError:
    PersonalityEngine = None

try:
    from .response_processor import ResponseProcessor
except SyntaxError:
    ResponseProcessor = None

try:
    from .proactive_engine import ProactiveEngine
except SyntaxError:
    ProactiveEngine = None

__all__ = [
    "ConversationManager",
    "MemoryRetriever", 
    "ContextAssembler",
    "GeminiClient",
    "PersonalityEngine",
    "ResponseProcessor",
    "ProactiveEngine"
]

__version__ = "1.0.0"
