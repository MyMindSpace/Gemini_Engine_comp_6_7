"""
Gemini Engine Orchestrator - Main Integration Layer for Components 6 & 7
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
from shared.mock_interfaces import MockComponent5Interface
from config.settings import settings

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
    """Main orchestrator for Gemini Engine Components 6 & 7"""
    
    def __init__(self, component5_interface: Optional[MockComponent5Interface] = None):
        """Initialize Gemini Engine orchestrator"""
        self.logger = get_logger("gemini_engine_orchestrator")
        
        # Initialize Component 6 modules
        self.conversation_manager = ConversationManager(component5_interface)
        self.memory_retriever = MemoryRetriever(component5_interface)
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
            "user_satisfaction_avg": 0.0
        }
        
        self.logger.info("Gemini Engine Orchestrator initialized successfully")
    
    @log_execution_time
    async def process_conversation(
        self,
        user_id: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        conversation_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process complete conversation flow through Components 6 & 7"""
        correlation_id = generate_correlation_id()
        
        self.logger.info(f"Processing conversation for user {user_id}", extra={
            "correlation_id": correlation_id,
            "conversation_id": conversation_id,
            "message_length": len(user_message)
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Generate AI response using Component 6
            ai_response = await self._generate_ai_response(
                user_id, user_message, conversation_id, emotional_state, conversation_goals
            )
            
            # Step 2: Analyze response using Component 7
            analysis_results = await self._analyze_response(
                ai_response, user_id, conversation_id, {
                    "user_message": user_message,
                    "emotional_state": emotional_state,
                    "conversation_goals": conversation_goals
                }
            )
            
            # Step 3: Generate feedback for continuous improvement
            feedback_results = await self._generate_improvement_feedback(
                analysis_results, conversation_id
            )
            
            # Step 4: Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_performance_metrics(processing_time, True, analysis_results)
            
            # Compile final response
            result = {
                "response": ai_response.enhanced_response,
                "response_id": ai_response.response_id,
                "conversation_id": conversation_id,
                "processing_time_ms": processing_time,
                "quality_analysis": {
                    "overall_quality": analysis_results["analysis"].quality_scores.get("overall_quality", 0.0),
                    "user_satisfaction": analysis_results["satisfaction"].get("satisfaction_score", 0.0),
                    "improvement_suggestions": analysis_results["analysis"].improvement_suggestions
                },
                "metadata": {
                    "correlation_id": correlation_id,
                    "timestamp": datetime.utcnow(),
                    "components_used": ["component6", "component7"],
                    "performance_metrics": self.get_performance_summary()
                }
            }
            
            self.logger.info(f"Conversation processed successfully", extra={
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time,
                "quality_score": result["quality_analysis"]["overall_quality"]
            })
            
            return result
            
        except Exception as e:
            # Update performance metrics for failure
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_performance_metrics(processing_time, False, None)
            
            self.logger.error(f"Failed to process conversation: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "user_id": user_id
            })
            
            # Return error response
            return {
                "error": "Failed to process conversation",
                "error_details": str(e),
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time
            }
    
    async def _generate_ai_response(
        self,
        user_id: str,
        user_message: str,
        conversation_id: Optional[str],
        emotional_state: Optional[Dict[str, Any]],
        conversation_goals: Optional[List[str]]
    ) -> EnhancedResponse:
        """Generate AI response using Component 6"""
        try:
            # Use conversation manager to orchestrate Component 6
            ai_response = await with_timeout(
                self.conversation_manager.process_conversation(
                    user_id=user_id,
                    user_message=user_message,
                    conversation_id=conversation_id,
                    emotional_state=emotional_state,
                    conversation_goals=conversation_goals
                ),
                timeout_seconds=settings.performance.total_response_timeout_ms / 1000,
                default_value=None
            )
            
            if not ai_response:
                raise RuntimeError("Component 6 failed to generate response")
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Component 6 failed to generate response: {e}")
            raise
    
    async def _analyze_response(
        self,
        ai_response: EnhancedResponse,
        user_id: str,
        conversation_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze response using Component 7"""
        try:
            # Prepare context for analysis
            analysis_context = {
                "user_message": context.get("user_message", ""),
                "emotional_state": context.get("emotional_state", {}),
                "conversation_goals": context.get("conversation_goals", []),
                "response_metadata": ai_response.enhancement_metadata
            }
            
            # Run Component 7 analysis
            analysis_results = await with_timeout(
                self.component7.analyze_response(
                    ai_response, analysis_context
                ),
                timeout_seconds=settings.performance.analysis_timeout_ms / 1000,
                default_value=None
            )
            
            if not analysis_results:
                # Create basic analysis results
                from shared.schemas import ResponseAnalysis
                basic_analysis = ResponseAnalysis(
                    analysis_id=generate_correlation_id(),
                    response_id=ai_response.response_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    quality_scores={"overall_quality": 0.7},
                    context_usage={},
                    user_satisfaction=0.7,
                    improvement_suggestions=[]
                )
                
                analysis_results = {
                    "analysis": basic_analysis,
                    "satisfaction": {"satisfaction_score": 0.7}
                }
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Component 7 analysis failed: {e}")
            # Return basic analysis on failure
            from shared.schemas import ResponseAnalysis
            fallback_analysis = ResponseAnalysis(
                analysis_id=generate_correlation_id(),
                response_id=ai_response.response_id,
                conversation_id=conversation_id or "unknown",
                user_id=user_id,
                quality_scores={"overall_quality": 0.5},
                context_usage={},
                user_satisfaction=0.5,
                improvement_suggestions=["Analysis failed, manual review recommended"]
            )
            
            return {
                "analysis": fallback_analysis,
                "satisfaction": {"satisfaction_score": 0.5}
            }
    
    async def _generate_improvement_feedback(
        self,
        analysis_results: Dict[str, Any],
        conversation_id: str
    ) -> Dict[str, Any]:
        """Generate feedback for continuous improvement"""
        try:
            # Generate feedback for Component 5 (Memory Gates)
            component5_feedback = await self.component7.generate_feedback(
                "component5", [analysis_results["analysis"]]
            )
            
            # Generate feedback for Component 6 (Gemini Integration)
            component6_feedback = await self.component7.generate_feedback(
                "component6", [analysis_results["analysis"]]
            )
            
            return {
                "component5_feedback": component5_feedback,
                "component6_feedback": component6_feedback,
                "feedback_generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate improvement feedback: {e}")
            return {"error": "Feedback generation failed"}
    
    async def _update_performance_metrics(
        self,
        processing_time: float,
        success: bool,
        analysis_results: Optional[Dict[str, Any]]
    ) -> None:
        """Update orchestrator performance metrics"""
        self.conversation_metrics["total_conversations"] += 1
        
        # Update average response time
        total_conversations = self.conversation_metrics["total_conversations"]
        current_avg = self.conversation_metrics["avg_response_time_ms"]
        self.conversation_metrics["avg_response_time_ms"] = (
            (current_avg * (total_conversations - 1) + processing_time) / total_conversations
        )
        
        # Update success rate
        if success:
            current_success_rate = self.conversation_metrics["success_rate"]
            self.conversation_metrics["success_rate"] = (
                (current_success_rate * (total_conversations - 1) + 1) / total_conversations
            )
        else:
            current_success_rate = self.conversation_metrics["success_rate"]
            self.conversation_metrics["success_rate"] = (
                (current_success_rate * (total_conversations - 1)) / total_conversations
            )
        
        # Update user satisfaction average
        if analysis_results and "satisfaction" in analysis_results:
            satisfaction_score = analysis_results["satisfaction"].get("satisfaction_score", 0.0)
            current_satisfaction = self.conversation_metrics["user_satisfaction_avg"]
            self.conversation_metrics["user_satisfaction_avg"] = (
                (current_satisfaction * (total_conversations - 1) + satisfaction_score) / total_conversations
            )
    
    async def generate_proactive_message(
        self,
        user_id: str,
        trigger_event: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate proactive message using Component 6"""
        try:
            # Get user profile
            user_profile = await self.memory_retriever.get_user_profile(user_id)
            
            # Generate proactive message
            proactive_message = await self.proactive_engine.generate_proactive_message(
                user_id=user_id,
                trigger_event=trigger_event,
                context=context,
                user_profile=user_profile
            )
            
            if proactive_message:
                return {
                    "message_id": proactive_message.message_id,
                    "content": proactive_message.message_content,
                    "optimal_timing": proactive_message.optimal_timing,
                    "priority": proactive_message.priority_level
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to generate proactive message: {e}")
            return None
    
    async def get_pending_proactive_messages(self, user_id: str) -> List[Dict[str, Any]]:
        """Get pending proactive messages for user"""
        try:
            pending_messages = await self.proactive_engine.get_pending_messages(user_id)
            
            return [
                {
                    "message_id": msg.message_id,
                    "content": msg.message_content,
                    "optimal_timing": msg.optimal_timing,
                    "priority": msg.priority_level
                }
                for msg in pending_messages
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get pending messages: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            return {
                "orchestrator_metrics": self.conversation_metrics,
                "component6_metrics": {
                    "conversation_manager": self.conversation_manager.get_performance_metrics(),
                    "memory_retriever": self.memory_retriever.get_performance_metrics(),
                    "context_assembler": self.context_assembler.get_performance_metrics(),
                    "personality_engine": self.personality_engine.get_performance_metrics(),
                    "gemini_client": self.gemini_client.get_performance_metrics(),
                    "proactive_engine": self.proactive_engine.get_performance_metrics(),
                    "response_processor": self.response_processor.get_performance_metrics()
                },
                "component7_metrics": self.component7.get_performance_summary(),
                "system_health": {
                    "total_conversations": self.conversation_metrics["total_conversations"],
                    "avg_response_time_ms": self.conversation_metrics["avg_response_time_ms"],
                    "success_rate": self.conversation_metrics["success_rate"],
                    "user_satisfaction_avg": self.conversation_metrics["user_satisfaction_avg"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": "Performance summary unavailable"}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "overall_healthy": True,
            "timestamp": datetime.utcnow(),
            "component_health": {}
        }
        
        try:
            # Check Component 6 health
            component6_health = await self.conversation_manager.health_check()
            health_status["component_health"]["component6"] = component6_health
            
            # Check individual Component 6 modules
            health_status["component_health"]["memory_retriever"] = await self.memory_retriever.health_check()
            health_status["component_health"]["gemini_client"] = await self.gemini_client.health_check()
            health_status["component_health"]["personality_engine"] = await self.personality_engine.health_check()
            
            # Overall health assessment
            all_healthy = all(
                status for status in health_status["component_health"].values()
                if isinstance(status, (bool, dict))
            )
            
            health_status["overall_healthy"] = all_healthy
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.conversation_manager.close()
            await self.gemini_client.close()
            self.logger.info("Gemini Engine Orchestrator cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            asyncio.create_task(self.cleanup())
        except:
            pass
