import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add comp5 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'comp5'))

from comp5.core.memory_manager import MemoryManager
from comp5.config.settings import LSTMConfig
from comp5.models.memory_item import MemoryItem
from shared.schemas import MemoryContext, MemoryItem as SharedMemoryItem, UserProfile

class Component5Interface:
    """Bridge interface connecting Component 6 to Component 5"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize Component 5 LSTM system
        self.config = LSTMConfig()
        self.memory_manager = MemoryManager(self.config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LSTM memory system"""
        if not self._initialized:
            await self.memory_manager.initialize()
            self._initialized = True
    
    async def get_memory_context(self, 
                                user_id: str,
                                current_message: str,
                                conversation_id: str,
                                max_memories: int = 10) -> MemoryContext:
        """Get memory context - Component 6 interface"""
        
        # Generate dummy feature vector (replace with Component 4 integration)
        query_features = [0.5] * 90  # Component 5 expects 90-dim features
        
        # Get relevant memories from Component 5
        memories, metadata = await self.memory_manager.get_relevant_context(
            user_id=user_id,
            query=current_message,
            query_features=query_features,
            max_tokens=2000
        )
        
        # Convert Component 5 MemoryItem to Component 6 format
        converted_memories = []
        relevance_scores = []
        
        for memory in memories:
            converted_memory = self._convert_memory_item(memory)
            converted_memories.append(converted_memory)
            relevance_scores.append(memory.importance_score)
        
        return MemoryContext(
            selected_memories=converted_memories,
            relevance_scores=relevance_scores,
            token_usage=metadata.get('total_tokens_used', 0),
            assembly_metadata=metadata
        )
    
    async def save_memory(self,
                         user_id: str,
                         content: str,
                         feature_vector: List[float],
                         embeddings: List[float],
                         metadata: Optional[Dict] = None) -> bool:
        """Save new memory through Component 5"""
        
        memory = await self.memory_manager.process_new_entry(
            user_id=user_id,
            feature_vector=feature_vector,
            content=content,
            embeddings=embeddings,
            metadata=metadata
        )
        
        return memory is not None
    
    async def update_memory_access(self,
                                  memory_id: str,
                                  user_id: str,
                                  access_type: str = "conversation_reference") -> bool:
        """Update memory access tracking"""
        try:
            # Component 5 handles this through memory access updates
            # This is automatically handled when memories are retrieved
            return True
        except Exception:
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile - delegated to Component 6's own storage"""
        # Component 5 doesn't handle user profiles, so return None
        # Component 6 will create default profiles
        return None
    
    def _convert_memory_item(self, comp5_memory: MemoryItem) -> Dict[str, Any]:
        """Convert Component 5 MemoryItem to Component 6 format"""
        return {
            'id': comp5_memory.id,
            'user_id': comp5_memory.user_id,
            'content_summary': comp5_memory.content_summary,
            'memory_type': comp5_memory.memory_type,
            'importance_score': comp5_memory.importance_score,
            'emotional_significance': comp5_memory.emotional_significance or 0.5,
            'created_at': comp5_memory.created_at,
            'last_accessed': comp5_memory.last_accessed,
            'access_frequency': comp5_memory.access_frequency,
            'retrieval_triggers': comp5_memory.retrieval_triggers,
            'relationships': comp5_memory.relationships
        }
