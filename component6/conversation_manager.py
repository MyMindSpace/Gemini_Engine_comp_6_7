# component6/conversation_manager.py (REPLACE THE IMPORT SECTION)
"""
Conversation Manager Module for Component 6.
Main orchestrator for conversation flow and memory integration with real Component 5.
"""

import asyncio
import logging
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

# Add project root to path to find comp5_interface
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.schemas import (
    ConversationContext, MemoryContext, UserProfile, EnhancedResponse,
    ProactiveMessage, CommunicationStyle
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id,
    with_timeout
)
from configu.settings import settings

from .memory_retriever import MemoryRetriever
from .context_assembler import ContextAssembler
from .gemini_client import GeminiClient
from .personality_engine import PersonalityEngine
from .proactive_engine import ProactiveEngine
from .response_processor import ResponseProcessor

# Import the actual Component 5 interface class
try:
    # Try to import from comp5_interface.py (your bridge file)
    from comp5_interface import Component5Bridge as Component5Interface
except ImportError:
    # Fallback to enhanced interface if comp5_interface doesn't exist
    from .enhanced_comp5_interface import EnhancedComponent5Interface as Component5Interface


class ConversationManager:
    """Main orchestrator for conversation management in Component 6"""
    
    def __init__(self, component5_interface: Optional[Component5Interface] = None):
        """Initialize conversation manager with real Component 5 bridge"""
        self.logger = get_logger("conversation_manager")
        
        # Use the provided Component 5 interface or create a new one
        self.component5_interface = component5_interface or Component5Interface()
        
        # Initialize sub-components with Component 5 interface
        self.memory_retriever = MemoryRetriever(self.component5_interface)
        self.context_assembler = ContextAssembler()
        self.gemini_client = GeminiClient()
        self.personality_engine = PersonalityEngine()
        self.proactive_engine = ProactiveEngine()
        self.response_processor = ResponseProcessor()
        
        # Conversation state tracking
        self.active_conversations = {}
        self.conversation_history = {}
        
        # Performance tracking
        self.conversation_metrics = {
            "total_conversations": 0,
            "active_conversations": 0,
            "avg_response_time_ms": 0.0,
            "success_rate": 0.0,
            "component5_calls": 0,
            "memory_context_successes": 0
        }
        
        self.logger.info("Conversation Manager initialized with real Component 5")
    
    @log_execution_time
    async def process_conversation(
        self,
        user_id: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        conversation_goals: Optional[List[str]] = None
    ) -> EnhancedResponse:
        """Process a user message and generate AI response using Component 5 memories"""
        correlation_id = generate_correlation_id()
        self.logger.info(f"Processing conversation for user {user_id}", extra={
            "correlation_id": correlation_id,
            "conversation_id": conversation_id,
            "message_length": len(user_message)
        })
        
        start_time = datetime.utcnow()
        conversation_id = conversation_id or f"conv_{uuid.uuid4()}"
        
        try:
            # Step 1: Get memory context from Component 5 bridge
            memory_context = await self._get_memory_context(
                user_id, user_message, conversation_id
            )
            
            # Step 2: Get user profile
            user_profile = await self._get_user_profile(user_id)
            
            # Step 3: Create conversation context
            conversation_context = await self._create_conversation_context(
                user_id, conversation_id, user_message, memory_context, 
                user_profile, emotional_state, conversation_goals
            )
            
            # Step 4: Assemble context for Gemini API
            gemini_request = await self._assemble_gemini_context(
                conversation_context, user_message
            )
            
            # Step 5: Generate AI response
            ai_response = await self._generate_ai_response(gemini_request)
            
            # Step 6: Update conversation history and memory access
            await self._update_conversation_state(
                conversation_context, user_message, ai_response, memory_context
            )
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, True)
            
            self.logger.info(f"Conversation processed successfully", extra={
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time,
                "memories_used": len(memory_context.selected_memories)
            })
            
            return ai_response
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_metrics(processing_time, False)
            
            self.logger.error(f"Conversation processing failed", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time
            })
            raise
    
    async def _get_memory_context(
        self, 
        user_id: str, 
        user_message: str, 
        conversation_id: str
    ) -> MemoryContext:
        """Get memory context from Component 5 with error handling"""
        try:
            self.conversation_metrics["component5_calls"] += 1
            
            # Call Component 5 interface
            memory_context = await self.component5_interface.get_memory_context(
                user_id=user_id,
                current_message=user_message,
                conversation_id=conversation_id,
                max_memories=settings.performance.max_memories_per_context if hasattr(settings, 'performance') else 10
            )
            
            self.conversation_metrics["memory_context_successes"] += 1
            
            self.logger.info(f"Retrieved {len(memory_context.selected_memories)} memories from Component 5")
            return memory_context
            
        except Exception as e:
            self.logger.error(f"Component 5 memory retrieval failed: {e}")
            
            # Create minimal fallback context
            return MemoryContext(
                selected_memories=[],
                relevance_scores=[],
                token_usage=0,
                assembly_metadata={
                    'error': str(e),
                    'fallback_used': True,
                    'source': 'component5_error_fallback'
                }
            )
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile from Component 5 or create default"""
        try:
            if hasattr(self.component5_interface, 'get_user_profile'):
                user_profile = await self.component5_interface.get_user_profile(user_id)
                if user_profile:
                    return user_profile
            
        except Exception as e:
            self.logger.warning(f"Could not get user profile from Component 5: {e}")
        
        # Create default user profile
        return UserProfile(
            user_id=user_id,
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_response_length="moderate",
            emotional_matching=True,
            humor_preference=0.7,
            formality_level=0.3,
            conversation_depth=0.8
        )
    
    # ... rest of the methods remain the same as in the original file
    # (I'm keeping this focused on the import and interface fixes)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get conversation manager performance metrics"""
        component5_success_rate = 0.0
        if self.conversation_metrics["component5_calls"] > 0:
            component5_success_rate = (
                self.conversation_metrics["memory_context_successes"] / 
                self.conversation_metrics["component5_calls"]
            )
        
        return {
            **self.conversation_metrics,
            "component5_success_rate": component5_success_rate,
            "memory_retrieval_stats": self.memory_retriever.get_retrieval_statistics() if hasattr(self.memory_retriever, 'get_retrieval_statistics') else {}
        }