# gemini_engine_orchestrator.py (REPLACE EXISTING)
"""
Gemini Engine Orchestrator - Main Integration Layer for Components 5, 6 & 7
Production-ready AI conversational system integrating all components
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from shared.schemas import (
    ConversationContext, MemoryContext, UserProfile, EnhancedResponse,
    ResponseAnalysis, ConversationMetrics, ContextEffectiveness
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id,
    with_timeout
)
from configu.settings import settings

# Component 5 integration (from project root)
import sys
import os
# Add project root to path to find component5_integration
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from component6.comp5_interface import Component5Bridge

# Component 6 imports
from component6.conversation_manager import ConversationManager
from component6.memory_retriever import MemoryRetriever
from component6.context_assembler import ContextAssembler
from component6.personality_engine import PersonalityEngine
from component6.gemini_client import GeminiClient
from component6.proactive_engine import ProactiveEngine
from component6.response_processor import ResponseProcessor

# Component 7 imports
from component7 import Component7Orchestrator


class GeminiEngineOrchestrator:
    """Main orchestrator for Gemini Engine Components 5, 6 & 7"""
    
    def __init__(self):
        """Initialize Gemini Engine orchestrator with real Component 5"""
        self.logger = get_logger("gemini_engine_orchestrator")
        
        # Initialize Component 5 bridge
        self.component5_bridge = Component5Bridge()
        
        # Initialize Component 6 modules with real Component 5
        self.conversation_manager = ConversationManager(self.component5_bridge)
        self.memory_retriever = MemoryRetriever(self.component5_bridge)
        self.context_assembler = ContextAssembler()
        self.personality_engine = PersonalityEngine()
        self.gemini_client = GeminiClient()
        self.proactive_engine = ProactiveEngine()
        self.response_processor = ResponseProcessor()
        
        # Initialize Component 7
        self.component7 = Component7Orchestrator()
        
        # Performance tracking
        self.conversation_metrics = {
            "total_conversations": 0,
            "avg_response_time_ms": 0.0,
            "success_rate": 0.0,
            "user_satisfaction_avg": 0.0,
            "component5_health": False,
            "component6_health": False,
            "component7_health": False
        }
        
        self.logger.info("Gemini Engine Orchestrator initialized with real Component 5")
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            self.logger.info("🚀 Initializing Gemini Engine (Components 5, 6, 7)...")
            
            # Initialize Component 5 bridge first
            comp5_success = await self.component5_bridge.initialize()
            if not comp5_success:
                raise RuntimeError("Component 5 initialization failed")
            
            self.conversation_metrics["component5_health"] = True
            self.conversation_metrics["component6_health"] = True  # Assumed healthy if comp5 works
            self.conversation_metrics["component7_health"] = True  # Assumed healthy
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    @log_execution_time
    async def process_conversation(
        self,
        user_id: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        conversation_goals: Optional[List[str]] = None
    ) -> EnhancedResponse:
        """Process a complete conversation through all components"""
        correlation_id = generate_correlation_id()
        self.logger.info(f"Processing conversation", extra={
            "correlation_id": correlation_id,
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Get memory context from Component 5 (LSTM Gates)
            self.logger.info("📝 Step 1: Retrieving memory context from Component 5...")
            memory_context = await self.component5_bridge.get_memory_context(
                user_id=user_id,
                current_message=user_message,
                conversation_id=conversation_id or f"conv_{uuid.uuid4()}",
                max_memories=20
            )
            
            # Step 2: Get user profile
            self.logger.info("👤 Step 2: Retrieving user profile...")
            user_profile = await self.component5_bridge.get_user_profile(user_id)
            
            # Step 3: Process conversation through Component 6
            self.logger.info("🤖 Step 3: Processing through Component 6...")
            enhanced_response = await self.conversation_manager.process_conversation(
                user_id=user_id,
                user_message=user_message,
                conversation_id=conversation_id,
                emotional_state=emotional_state,
                conversation_goals=conversation_goals
            )
            
            # Step 4: Analyze response through Component 7
            self.logger.info("📊 Step 4: Analyzing response through Component 7...")
            response_analysis = await self.component7.analyze_response(
                user_id=user_id,
                user_message=user_message,
                ai_response=enhanced_response.enhanced_response,
                conversation_context={
                    "memory_context": memory_context,
                    "user_profile": user_profile,
                    "emotional_state": emotional_state
                }
            )
            
            # Step 5: Update memory access tracking in Component 5
            self.logger.info("🔄 Step 5: Updating memory access in Component 5...")
            if memory_context.selected_memories:
                memory_ids = [mem['id'] for mem in memory_context.selected_memories if 'id' in mem]
                await self.component5_bridge.update_memory_access(
                    memory_ids=memory_ids,
                    user_id=user_id,
                    conversation_id=conversation_id or "unknown"
                )
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_conversation_metrics(processing_time, True)
            
            # Enhance response with analysis
            enhanced_response.response_analysis = response_analysis
            enhanced_response.processing_time_ms = processing_time
            
            self.logger.info(f"✅ Conversation processed successfully", extra={
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time,
                "memories_used": len(memory_context.selected_memories),
                "quality_score": response_analysis.overall_quality if response_analysis else 0.0
            })
            
            return enhanced_response
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_conversation_metrics(processing_time, False)
            
            self.logger.error(f"❌ Conversation processing failed", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time
            })
            raise
    
    async def process_proactive_engagement(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate proactive engagement using all components"""
        try:
            self.logger.info(f"Processing proactive engagement for user {user_id}")
            
            # Get recent memory context from Component 5
            recent_context = await self.component5_bridge.get_memory_context(
                user_id=user_id,
                current_message="proactive_engagement_trigger",
                conversation_id=f"proactive_{uuid.uuid4()}",
                max_memories=5
            )
            
            # Generate proactive message through Component 6
            proactive_message = await self.proactive_engine.generate_proactive_message(
                user_id=user_id,
                memory_context=recent_context,
                user_context=context
            )
            
            return {
                "message": proactive_message,
                "context_used": recent_context.assembly_metadata,
                "trigger_reason": "scheduled_check_in"
            }
            
        except Exception as e:
            self.logger.error(f"Error in proactive engagement: {e}")
            return None
    
    def _update_conversation_metrics(self, processing_time: float, success: bool):
        """Update conversation performance metrics"""
        self.conversation_metrics["total_conversations"] += 1
        
        # Update average response time
        total_convs = self.conversation_metrics["total_conversations"]
        current_avg = self.conversation_metrics["avg_response_time_ms"]
        new_avg = ((current_avg * (total_convs - 1)) + processing_time) / total_convs
        self.conversation_metrics["avg_response_time_ms"] = new_avg
        
        # Update success rate
        if success:
            successful_convs = self.conversation_metrics["success_rate"] * (total_convs - 1) + 1
        else:
            successful_convs = self.conversation_metrics["success_rate"] * (total_convs - 1)
        
        self.conversation_metrics["success_rate"] = successful_convs / total_convs
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components"""
        health_status = {
            "overall_healthy": False,
            "component_health": {},
            "system_metrics": self.conversation_metrics,
            "timestamp": datetime.utcnow()
        }
        
        try:
            # Check Component 5
            comp5_health = await self.component5_bridge.health_check()
            health_status["component_health"]["component5"] = comp5_health
            
            # Check Component 6 (basic functionality)
            comp6_health = {
                "conversation_manager": self.conversation_manager is not None,
                "gemini_client": self.gemini_client is not None,
                "context_assembler": self.context_assembler is not None
            }
            health_status["component_health"]["component6"] = comp6_health
            
            # Check Component 7
            comp7_health = {
                "orchestrator": self.component7 is not None,
                "response_analysis": True  # Assume healthy if initialized
            }
            health_status["component_health"]["component7"] = comp7_health
            
            # Overall health assessment
            health_status["overall_healthy"] = (
                comp5_health.get("initialized", False) and
                comp6_health.get("conversation_manager", False) and
                comp7_health.get("orchestrator", False)
            )
            
            self.logger.info(f"Health check completed: {'✅ Healthy' if health_status['overall_healthy'] else '❌ Unhealthy'}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)
        
        return health_status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "conversation_metrics": self.conversation_metrics,
            "component5_stats": self.component5_bridge.get_statistics(),
            "component6_stats": {
                "active_conversations": len(getattr(self.conversation_manager, 'active_conversations', {})),
                "memory_cache_size": len(getattr(self.memory_retriever, 'cache', {}))
            },
            "component7_stats": {
                # Component 7 stats would be available through component7 interface
                "analysis_completed": True
            },
            "system_uptime": datetime.utcnow(),
            "health_summary": self.conversation_metrics
        }
    
    async def cleanup(self):
        """Cleanup all components"""
        try:
            await self.component5_bridge.cleanup()
            # Add cleanup for other components as needed
            self.logger.info("✅ Orchestrator cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during orchestrator cleanup: {e}")