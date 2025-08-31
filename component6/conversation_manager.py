"""
Conversation Manager Module for Component 6.
Main orchestrator for conversation flow and memory integration.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from shared.schemas import (
    ConversationContext, MemoryContext, UserProfile, EnhancedResponse,
    ProactiveMessage, CommunicationStyle
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id,
    with_timeout
)
from shared.mock_interfaces import MockComponent5Interface
from config.settings import settings

from .memory_retriever import MemoryRetriever
from .context_assembler import ContextAssembler
from .gemini_client import GeminiClient


class ConversationManager:
    """Main orchestrator for conversation management in Component 6"""
    
    def __init__(self, component5_interface: Optional[MockComponent5Interface] = None):
        """Initialize conversation manager"""
        self.logger = get_logger("conversation_manager")
        
        # Initialize sub-components
        self.memory_retriever = MemoryRetriever(component5_interface)
        self.context_assembler = ContextAssembler()
        self.gemini_client = GeminiClient()
        
        # Conversation state tracking
        self.active_conversations = {}
        self.conversation_history = {}
        
        # Performance tracking
        self.conversation_metrics = {
            "total_conversations": 0,
            "active_conversations": 0,
            "avg_response_time_ms": 0.0,
            "success_rate": 0.0
        }
        
        self.logger.info("Conversation Manager initialized")
    
    @log_execution_time
    async def process_conversation(
        self,
        user_id: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        conversation_goals: Optional[List[str]] = None
    ) -> EnhancedResponse:
        """Process a user message and generate AI response"""
        correlation_id = generate_correlation_id()
        self.logger.info(f"Processing conversation for user {user_id}", extra={
            "correlation_id": correlation_id,
            "conversation_id": conversation_id,
            "message_length": len(user_message)
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Get or create conversation context
            conversation_context = await self._get_conversation_context(
                user_id, conversation_id, emotional_state, conversation_goals
            )
            
            # Retrieve relevant memories
            memory_context = await self._retrieve_memories(
                user_id, user_message, conversation_context.conversation_id
            )
            
            # Update conversation context with memories
            conversation_context.memories = memory_context.selected_memories
            conversation_context.last_updated = datetime.utcnow()
            
            # Assemble context for Gemini
            gemini_request = await self._assemble_context(
                conversation_context, user_message
            )
            
            # Generate AI response
            ai_response = await self._generate_ai_response(gemini_request)
            
            # Update conversation history
            await self._update_conversation_history(
                conversation_context, user_message, ai_response
            )
            
            # Update memory access tracking
            await self._update_memory_access(memory_context, user_id)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, True)
            
            self.logger.info(f"Conversation processed successfully", extra={
                "correlation_id": correlation_id,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "response_length": len(ai_response.enhanced_response)
            })
            
            return ai_response
            
        except Exception as e:
            # Update performance metrics
            self._update_performance_metrics(start_time, False)
            
            self.logger.error(f"Failed to process conversation: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "user_id": user_id
            })
            raise
    
    async def _get_conversation_context(
        self,
        user_id: str,
        conversation_id: Optional[str],
        emotional_state: Optional[Dict[str, Any]],
        conversation_goals: Optional[List[str]]
    ) -> ConversationContext:
        """Get or create conversation context"""
        if conversation_id and conversation_id in self.active_conversations:
            # Update existing conversation
            context = self.active_conversations[conversation_id]
            if emotional_state:
                context.emotional_state.update(emotional_state)
            if conversation_goals:
                context.conversation_goals.extend(conversation_goals)
            return context
        
        # Create new conversation
        new_conversation_id = conversation_id or str(uuid.uuid4())
        
        # Get user profile
        user_profile = await self.memory_retriever.get_user_profile(user_id)
        if not user_profile:
            # Create default profile
            user_profile = UserProfile(
                user_id=user_id,
                communication_style=CommunicationStyle.FRIENDLY,
                preferred_response_length="moderate"
            )
        
        # Create conversation context
        context = ConversationContext(
            user_id=user_id,
            conversation_id=new_conversation_id,
            memories=[],
            emotional_state=emotional_state or {},
            personality_profile=user_profile,
            recent_history=[],
            conversation_goals=conversation_goals or [],
            context_window_size=settings.performance.max_context_tokens
        )
        
        # Store in active conversations
        self.active_conversations[new_conversation_id] = context
        self.conversation_metrics["active_conversations"] = len(self.active_conversations)
        
        return context
    
    async def _retrieve_memories(
        self,
        user_id: str,
        user_message: str,
        conversation_id: str
    ) -> MemoryContext:
        """Retrieve relevant memories for the conversation"""
        try:
            memory_context = await with_timeout(
                self.memory_retriever.get_memory_context(
                    user_id=user_id,
                    current_message=user_message,
                    conversation_id=conversation_id,
                    max_memories=10
                ),
                timeout_seconds=settings.performance.memory_retrieval_timeout_ms / 1000,
                default_value=None
            )
            
            if not memory_context:
                # Create empty memory context if retrieval fails
                memory_context = MemoryContext(
                    selected_memories=[],
                    relevance_scores=[],
                    token_usage=0,
                    assembly_metadata={"error": "Memory retrieval failed"}
                )
            
            return memory_context
            
        except Exception as e:
            self.logger.warning(f"Memory retrieval failed: {e}, using empty context")
            return MemoryContext(
                selected_memories=[],
                relevance_scores=[],
                token_usage=0,
                assembly_metadata={"error": str(e)}
            )
    
    async def _assemble_context(
        self,
        conversation_context: ConversationContext,
        user_message: str
    ) -> Any:
        """Assemble context for Gemini API request"""
        try:
            gemini_request = await with_timeout(
                self.context_assembler.assemble_context(
                    conversation_context=conversation_context,
                    current_message=user_message,
                    target_tokens=settings.performance.max_context_tokens
                ),
                timeout_seconds=settings.performance.context_assembly_timeout_ms / 1000,
                default_value=None
            )
            
            if not gemini_request:
                raise RuntimeError("Context assembly failed")
            
            return gemini_request
            
        except Exception as e:
            self.logger.error(f"Context assembly failed: {e}")
            raise
    
    async def _generate_ai_response(self, gemini_request: Any) -> EnhancedResponse:
        """Generate AI response using Gemini"""
        try:
            response = await with_timeout(
                self.gemini_client.generate_response(gemini_request),
                timeout_seconds=settings.performance.total_response_timeout_ms / 1000,
                default_value=None
            )
            
            if not response:
                raise RuntimeError("AI response generation failed")
            
            return response
            
        except Exception as e:
            self.logger.error(f"AI response generation failed: {e}")
            raise
    
    async def _update_conversation_history(
        self,
        conversation_context: ConversationContext,
        user_message: str,
        ai_response: EnhancedResponse
    ) -> None:
        """Update conversation history"""
        history_entry = {
            "timestamp": datetime.utcnow(),
            "user_message": user_message,
            "ai_response": ai_response.processed_response,
            "response_id": ai_response.response_id
        }
        
        conversation_context.recent_history.append(history_entry)
        
        # Keep only last 20 exchanges
        if len(conversation_context.recent_history) > 20:
            conversation_context.recent_history = conversation_context.recent_history[-20:]
        
        # Store in conversation history
        self.conversation_history[conversation_context.conversation_id] = {
            "user_id": conversation_context.user_id,
            "created_at": conversation_context.created_at,
            "last_updated": conversation_context.last_updated,
            "total_exchanges": len(conversation_context.recent_history)
        }
    
    async def _update_memory_access(
        self,
        memory_context: MemoryContext,
        user_id: str
    ) -> None:
        """Update memory access tracking"""
        try:
            for memory in memory_context.selected_memories:
                if 'id' in memory:
                    await self.memory_retriever.update_memory_access(
                        memory_id=memory['id'],
                        user_id=user_id,
                        access_type="conversation_reference"
                    )
        except Exception as e:
            self.logger.warning(f"Failed to update memory access: {e}")
    
    async def get_conversation_summary(
        self,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary of a conversation"""
        if conversation_id not in self.active_conversations:
            return None
        
        context = self.active_conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "user_id": context.user_id,
            "created_at": context.created_at,
            "last_updated": context.last_updated,
            "total_exchanges": len(context.recent_history),
            "memories_used": len(context.memories),
            "conversation_goals": context.conversation_goals,
            "current_topic": context.current_topic
        }
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """End an active conversation"""
        if conversation_id not in self.active_conversations:
            return False
        
        # Remove from active conversations
        del self.active_conversations[conversation_id]
        self.conversation_metrics["active_conversations"] = len(self.active_conversations)
        
        self.logger.info(f"Conversation {conversation_id} ended")
        return True
    
    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        user_conversations = []
        
        for conv_id, context in self.active_conversations.items():
            if context.user_id == user_id:
                summary = await self.get_conversation_summary(conv_id)
                if summary:
                    user_conversations.append(summary)
        
        return user_conversations
    
    def _update_performance_metrics(self, start_time: datetime, success: bool) -> None:
        """Update performance tracking metrics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Update total conversations
        self.conversation_metrics["total_conversations"] += 1
        
        # Update success rate
        if success:
            self.conversation_metrics["success_rate"] = (
                (self.conversation_metrics["success_rate"] * 
                 (self.conversation_metrics["total_conversations"] - 1) + 1) /
                self.conversation_metrics["total_conversations"]
            )
        else:
            self.conversation_metrics["success_rate"] = (
                (self.conversation_metrics["success_rate"] * 
                 (self.conversation_metrics["total_conversations"] - 1)) /
                self.conversation_metrics["total_conversations"]
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        # Get sub-component metrics
        memory_metrics = self.memory_retriever.get_performance_metrics()
        context_metrics = self.context_assembler.get_performance_metrics()
        gemini_metrics = self.gemini_client.get_performance_metrics()
        
        return {
            "conversation_manager": self.conversation_metrics,
            "memory_retriever": memory_metrics,
            "context_assembler": context_metrics,
            "gemini_client": gemini_metrics,
            "active_conversations": len(self.active_conversations),
            "total_conversation_history": len(self.conversation_history)
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components"""
        health_status = {}
        
        try:
            # Check memory retriever
            health_status["memory_retriever"] = await self.memory_retriever.health_check()
        except Exception as e:
            self.logger.error(f"Memory retriever health check failed: {e}")
            health_status["memory_retriever"] = False
        
        try:
            # Check Gemini client
            health_status["gemini_client"] = await self.gemini_client.health_check()
        except Exception as e:
            self.logger.error(f"Gemini client health check failed: {e}")
            health_status["gemini_client"] = False
        
        # Context assembler is stateless, so always healthy
        health_status["context_assembler"] = True
        
        # Overall health
        health_status["overall"] = all(health_status.values())
        
        return health_status
    
    async def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """Clean up old inactive conversations"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        conversations_to_remove = []
        
        for conv_id, context in self.active_conversations.items():
            if context.last_updated < cutoff_time:
                conversations_to_remove.append(conv_id)
        
        # Remove old conversations
        for conv_id in conversations_to_remove:
            await self.end_conversation(conv_id)
        
        if conversations_to_remove:
            self.logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
        
        return len(conversations_to_remove)
    
    async def close(self):
        """Cleanup resources"""
        try:
            await self.gemini_client.close()
            self.logger.info("Conversation Manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Schedule cleanup
            asyncio.create_task(self.close())
        except:
            pass
