"""
Memory Retriever Module for Component 6.
Interfaces with Component 5 LSTM Memory Gates for context retrieval.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from shared.schemas import (
    MemoryContext, MemoryItem, UserProfile, ConversationContext
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id,
    calculate_memory_relevance, compress_memories, TokenCounter
)
from shared.mock_interfaces import MockComponent5Interface
from config.settings import settings
from .comp5_interface import Component5Interface


class MemoryRetriever:
    """Retrieves and processes memory context from Component 5"""
    
    def __init__(self, component5_interface: Optional[Component5Interface] = None):
        """Initialize memory retriever"""
        self.logger = get_logger("memory_retriever")
        self.component5 = component5_interface or Component5Interface()
        self.token_counter = TokenCounter()
        self.cache = {}
        
        # Performance tracking
        self.retrieval_times = []
        self.cache_hit_rate = 0.0
        self.total_requests = 0
        
        self.logger.info("Memory Retriever initialized")
    
    @log_execution_time
    async def get_memory_context(
        self,
        user_id: str,
        current_message: str,
        conversation_id: str,
        max_memories: int = 10,
        use_cache: bool = True
    ) -> MemoryContext:
        """Retrieve memory context from Component 5"""
        correlation_id = generate_correlation_id()
        self.logger.info(f"Retrieving memory context for user {user_id}", extra={
            "correlation_id": correlation_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "max_memories": max_memories
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"{user_id}_{conversation_id}_{hash(current_message)}"
            if use_cache and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self._update_cache_metrics(True)
                    self.logger.info(f"Cache hit for memory context", extra={
                        "correlation_id": correlation_id,
                        "cache_key": cache_key
                    })
                    return cached_result
            
            # Retrieve from Component 5
            memory_context = await self.component5.get_memory_context(
                user_id=user_id,
                current_message=current_message,
                conversation_id=conversation_id,
                max_memories=max_memories
            )
            
            # Validate memory context
            self._validate_memory_context(memory_context)
            
            # Optimize memory selection
            optimized_context = await self._optimize_memory_context(
                memory_context, current_message, max_memories
            )
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = optimized_context
                self._cleanup_cache()
            
            # Update metrics
            self._update_cache_metrics(False)
            self._update_performance_metrics(start_time)
            
            self.logger.info(f"Memory context retrieved successfully", extra={
                "correlation_id": correlation_id,
                "memories_count": len(optimized_context.selected_memories),
                "token_usage": optimized_context.token_usage,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            })
            
            return optimized_context
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory context: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "user_id": user_id
            })
            raise
    
    @log_execution_time
    async def _optimize_memory_context(
        self,
        memory_context: MemoryContext,
        current_message: str,
        max_memories: int
    ) -> MemoryContext:
        """Optimize memory context for better relevance and token efficiency"""
        self.logger.debug("Optimizing memory context")
        
        if not memory_context.selected_memories:
            return memory_context
        
        # Calculate additional relevance scores
        enhanced_memories = []
        enhanced_scores = []
        
        for memory, base_score in zip(
            memory_context.selected_memories,
            memory_context.relevance_scores
        ):
            # Calculate enhanced relevance
            enhanced_score = self._calculate_enhanced_relevance(
                memory, current_message, base_score
            )
            
            enhanced_memories.append(memory)
            enhanced_scores.append(enhanced_score)
        
        # Sort by enhanced relevance
        sorted_pairs = sorted(
            zip(enhanced_memories, enhanced_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top memories
        top_memories, top_scores = zip(*sorted_pairs[:max_memories])
        
        # Apply diversity filtering as specified in documentation
        final_memories, final_scores = self._apply_diversity_filtering(
            list(top_memories), list(top_scores), max_memories
        )
        
        # Compress memories if needed
        compressed_memories, final_token_usage = compress_memories(
            final_memories,
            settings.performance.max_context_tokens,
            self.token_counter
        )
        
        # Update assembly metadata
        assembly_metadata = memory_context.assembly_metadata.copy()
        assembly_metadata.update({
            "optimization_applied": True,
            "enhanced_scoring": True,
            "compression_applied": len(compressed_memories) < len(top_memories),
            "final_memory_count": len(compressed_memories),
            "optimization_timestamp": datetime.utcnow().isoformat()
        })
        
        return MemoryContext(
            selected_memories=compressed_memories,
            relevance_scores=top_scores[:len(compressed_memories)],
            token_usage=final_token_usage,
            assembly_metadata=assembly_metadata
        )
    
    def _calculate_enhanced_relevance(
        self,
        memory: Dict[str, Any],
        current_message: str,
        base_score: float
    ) -> float:
        """Calculate enhanced relevance score using multiple factors"""
        # Base relevance from Component 5
        enhanced_score = base_score
        
        # Content relevance boost
        content_relevance = calculate_memory_relevance(
            memory.get('content_summary', ''),
            current_message,
            memory
        )
        enhanced_score = enhanced_score * 0.7 + content_relevance * 0.3
        
        # Recency weighting as specified in documentation
        if 'created_at' in memory:
            age_days = (datetime.utcnow() - memory['created_at']).days
            # Enhanced recency weighting algorithm
            if age_days <= 1:
                recency_boost = 1.2  # Recent memories get significant boost
            elif age_days <= 7:
                recency_boost = 1.1  # This week gets moderate boost
            elif age_days <= 30:
                recency_boost = 1.0  # This month is neutral
            else:
                recency_boost = max(0.7, 1.0 - (age_days / 365.0) * 0.3)
            enhanced_score *= recency_boost
        
        # Importance boost
        importance_score = memory.get('importance_score', 0.5)
        enhanced_score *= (1.0 + importance_score * 0.2)
        
        # Emotional significance boost
        emotional_score = memory.get('emotional_significance', 0.5)
        enhanced_score *= (1.0 + emotional_score * 0.15)
        
        # Access frequency normalization
        access_freq = memory.get('access_frequency', 0)
        if access_freq > 0:
            # Slightly boost frequently accessed memories
            access_boost = min(0.1, access_freq * 0.02)
            enhanced_score *= (1.0 + access_boost)
        
        return min(1.0, max(0.0, enhanced_score))
    
    def _apply_diversity_filtering(
        self,
        memories: List[Dict[str, Any]],
        scores: List[float],
        max_memories: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Apply diversity filtering to ensure balanced representation of different memory types"""
        if len(memories) <= max_memories:
            return memories, scores
        
        # Group memories by type for diversity
        memory_groups = {}
        for i, memory in enumerate(memories):
            memory_type = memory.get('memory_type', 'conversation')
            if memory_type not in memory_groups:
                memory_groups[memory_type] = []
            memory_groups[memory_type].append((memory, scores[i], i))
        
        # Calculate target count per type
        num_types = len(memory_groups)
        base_per_type = max_memories // num_types
        remaining_slots = max_memories % num_types
        
        diverse_memories = []
        diverse_scores = []
        
        # Select memories from each type
        for memory_type, type_memories in memory_groups.items():
            target_count = base_per_type + (1 if remaining_slots > 0 else 0)
            if remaining_slots > 0:
                remaining_slots -= 1
            
            # Sort by score within type and take top N
            type_memories.sort(key=lambda x: x[1], reverse=True)
            selected = type_memories[:target_count]
            
            for memory, score, _ in selected:
                diverse_memories.append(memory)
                diverse_scores.append(score)
        
        # If we still have slots, fill with highest scoring remaining memories
        if len(diverse_memories) < max_memories:
            used_indices = set()
            for memory, score, idx in sum(memory_groups.values(), []):
                if memory in diverse_memories:
                    used_indices.add(idx)
            
            remaining_memories = [
                (memories[i], scores[i]) for i in range(len(memories))
                if i not in used_indices
            ]
            remaining_memories.sort(key=lambda x: x[1], reverse=True)
            
            slots_left = max_memories - len(diverse_memories)
            for memory, score in remaining_memories[:slots_left]:
                diverse_memories.append(memory)
                diverse_scores.append(score)
        
        return diverse_memories, diverse_scores
    
    def _validate_memory_context(self, memory_context: MemoryContext) -> None:
        """Validate memory context data structure"""
        if not isinstance(memory_context.selected_memories, list):
            raise ValueError("selected_memories must be a list")
        
        if not isinstance(memory_context.relevance_scores, list):
            raise ValueError("relevance_scores must be a list")
        
        if len(memory_context.selected_memories) != len(memory_context.relevance_scores):
            raise ValueError("Memory count and score count must match")
        
        if memory_context.token_usage < 0:
            raise ValueError("Token usage cannot be negative")
        
        # Validate individual memories
        for memory in memory_context.selected_memories:
            if not isinstance(memory, dict):
                raise ValueError("Each memory must be a dictionary")
            
            required_fields = ['id', 'user_id', 'content_summary']
            for field in required_fields:
                if field not in memory:
                    raise ValueError(f"Memory missing required field: {field}")
    
    def _is_cache_valid(self, cached_result: Any) -> bool:
        """Check if cached result is still valid"""
        if not isinstance(cached_result, MemoryContext):
            return False
        
        # Check if cache is too old
        if 'optimization_timestamp' in cached_result.assembly_metadata:
            timestamp_str = cached_result.assembly_metadata['optimization_timestamp']
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if datetime.utcnow() - timestamp > timedelta(minutes=30):
                    return False
            except ValueError:
                return False
        
        return True
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        if len(self.cache) > 1000:  # Limit cache size
            # Remove oldest entries
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].assembly_metadata.get('optimization_timestamp', ''),
                reverse=True
            )
            
            # Keep only top 500 entries
            self.cache = dict(sorted_items[:500])
    
    def _update_cache_metrics(self, cache_hit: bool) -> None:
        """Update cache performance metrics"""
        self.total_requests += 1
        
        if cache_hit:
            self.cache_hit_rate = (
                (self.cache_hit_rate * (self.total_requests - 1) + 1) / 
                self.total_requests
            )
        else:
            self.cache_hit_rate = (
                (self.cache_hit_rate * (self.total_requests - 1)) / 
                self.total_requests
            )
    
    def _update_performance_metrics(self, start_time: datetime) -> None:
        """Update performance tracking metrics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.retrieval_times.append(processing_time)
        
        # Keep only last 100 measurements
        if len(self.retrieval_times) > 100:
            self.retrieval_times = self.retrieval_times[-100:]
    
    async def update_memory_access(
        self,
        memory_id: str,
        user_id: str,
        access_type: str = "conversation_reference"
    ) -> bool:
        """Update memory access tracking in Component 5"""
        try:
            return await self.component5.update_memory_access(
                memory_id=memory_id,
                user_id=user_id,
                access_type=access_type
            )
        except Exception as e:
            self.logger.error(f"Failed to update memory access: {e}", extra={
                "memory_id": memory_id,
                "user_id": user_id,
                "access_type": access_type,
                "error": str(e)
            })
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from Component 5"""
        try:
            return await self.component5.get_user_profile(user_id=user_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve user profile: {e}", extra={
                "user_id": user_id,
                "error": str(e)
            })
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.retrieval_times:
            return {
                "avg_retrieval_time_ms": 0.0,
                "cache_hit_rate": self.cache_hit_rate,
                "total_requests": self.total_requests,
                "cache_size": len(self.cache)
            }
        
        return {
            "avg_retrieval_time_ms": sum(self.retrieval_times) / len(self.retrieval_times),
            "min_retrieval_time_ms": min(self.retrieval_times),
            "max_retrieval_time_ms": max(self.retrieval_times),
            "cache_hit_rate": self.cache_hit_rate,
            "total_requests": self.total_requests,
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached memory contexts"""
        self.cache.clear()
        self.logger.info("Memory cache cleared")
    
    async def health_check(self) -> bool:
        """Perform health check on Component 5 interface"""
        try:
            # Try to get a simple response from Component 5
            test_context = await self.component5.get_memory_context(
                user_id="test_user",
                current_message="test message",
                conversation_id="test_conversation",
                max_memories=1
            )
            
            return isinstance(test_context, MemoryContext)
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
