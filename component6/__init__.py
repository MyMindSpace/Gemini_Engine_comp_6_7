"""
Component 6: Gemini Brain Integration
Complete conversational AI system using Google Gemini with intelligent context assembly.
"""

from .conversation_manager import ConversationManager
from .memory_retriever import MemoryRetriever
from .context_assembler import ContextAssembler
from .gemini_client import GeminiClient
from .personality_engine import PersonalityEngine
from .response_processor import ResponseProcessor
from .proactive_engine import ProactiveEngine

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
__author__ = "Gemini Engine Team"
__description__ = "Production-ready Gemini integration with memory management and context assembly"
