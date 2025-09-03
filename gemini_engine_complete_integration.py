#!/usr/bin/env python3
"""
Gemini Engine Complete Integration - Components 5, 6 & 7
========================================================

This script demonstrates the complete integration of all three Gemini Engine components:
- Component 5: LSTM-Inspired Memory Gates + RL Training
- Component 6: Gemini Brain Integration  
- Component 7: Gemini Response Analysis

Architecture Flow:
Memory Gates (Comp 5) → Context Assembly (Comp 6) → Gemini API → Response Analysis (Comp 7)
     ↑                                                                        ↓
     ←←←←←←←←←←← Feedback Loop for Optimization ←←←←←←←←←←←←←←←←←←←←←←←←←
"""

import sys
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiEngineIntegration:
    """Complete integration of all Gemini Engine components"""
    
    def __init__(self):
        """Initialize the complete Gemini Engine system"""
        self.logger = logging.getLogger("gemini_engine_integration")
        self.components_status = {}
        self.orchestrator = None
        self.mock_component5 = None
        
        # Performance metrics
        self.conversation_history = []
        self.performance_metrics = {
            "total_conversations": 0,
            "avg_response_time_ms": 0.0,
            "success_rate": 0.0,
            "quality_scores": [],
            "satisfaction_scores": []
        }
    
    async def initialize_components(self):
        """Initialize all Gemini Engine components"""
        self.logger.info("🚀 Initializing Gemini Engine Components...")
        
        # Initialize Component 5 (Memory Gates)
        await self._initialize_component5()
        
        # Initialize Component 6 & 7 (Gemini Integration + Response Analysis)
        await self._initialize_main_orchestrator()
        
        # Verify system health
        await self._verify_system_health()
        
        self.logger.info("✅ Gemini Engine initialization complete!")
        return self.components_status
    
    async def _initialize_component5(self):
        """Initialize Component 5 - Memory Gates"""
        try:
            from shared.mock_interfaces import MockComponent5Interface
            from shared.schemas import MemoryContext, MemoryItem, UserProfile, CommunicationStyle
            
            # Initialize mock Component 5 for demonstration
            self.mock_component5 = MockComponent5Interface()
            
            # Create sample memories
            sample_memories = [
                {
                    "id": "mem_001",
                    "user_id": "user_001",
                    "memory_type": "conversation",
                    "content_summary": "User mentioned they love hiking and outdoor activities",
                    "importance_score": 0.8,
                    "emotional_significance": 0.6,
                    "created_at": datetime.now() - timedelta(days=2)
                },
                {
                    "id": "mem_002", 
                    "user_id": "user_001",
                    "memory_type": "emotion",
                    "content_summary": "User felt stressed about work deadlines last week",
                    "importance_score": 0.7,
                    "emotional_significance": 0.9,
                    "created_at": datetime.now() - timedelta(days=7)
                },
                {
                    "id": "mem_003",
                    "user_id": "user_001", 
                    "memory_type": "event",
                    "content_summary": "User has a presentation coming up next Friday",
                    "importance_score": 0.9,
                    "emotional_significance": 0.7,
                    "created_at": datetime.now() - timedelta(days=1)
                }
            ]
            
            # Add memories to mock interface
            for memory in sample_memories:
                await self.mock_component5.add_memory(memory)
            
            # Create sample user profile
            user_profile = UserProfile(
                user_id="user_001",
                communication_style=CommunicationStyle.FRIENDLY,
                preferred_response_length="moderate",
                emotional_matching=True,
                humor_preference=0.7,
                formality_level=0.3,
                conversation_depth=0.8
            )
            
            await self.mock_component5.set_user_profile("user_001", user_profile)
            
            self.components_status["component5"] = {
                "status": "initialized",
                "type": "mock_interface",
                "memories_loaded": len(sample_memories),
                "user_profiles": 1
            }
            
            self.logger.info("✅ Component 5 (Memory Gates) initialized with mock interface")
            
        except Exception as e:
            self.logger.error(f"❌ Component 5 initialization failed: {e}")
            self.components_status["component5"] = {"status": "failed", "error": str(e)}
    
    async def _initialize_main_orchestrator(self):
        """Initialize main orchestrator with Components 6 & 7"""
        try:
            from gemini_engine_orchestrator import GeminiEngineOrchestrator
            
            # Initialize orchestrator with Component 5 interface
            self.orchestrator = GeminiEngineOrchestrator(self.mock_component5)
            
            # Verify orchestrator health
            health_status = await self.orchestrator.health_check()
            
            self.components_status["component6"] = {
                "status": "initialized",
                "health": health_status.get("component_health", {}).get("component6", False)
            }
            
            self.components_status["component7"] = {
                "status": "initialized", 
                "health": True  # Component 7 is integrated in orchestrator
            }
            
            self.components_status["orchestrator"] = {
                "status": "initialized",
                "overall_healthy": health_status.get("overall_healthy", False)
            }
            
            self.logger.info("✅ Main orchestrator (Components 6 & 7) initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Main orchestrator initialization failed: {e}")
            self.components_status["orchestrator"] = {"status": "failed", "error": str(e)}
    
    async def _verify_system_health(self):
        """Verify overall system health"""
        try:
            if self.orchestrator:
                health_check = await self.orchestrator.health_check()
                self.components_status["system_health"] = health_check
                self.logger.info("✅ System health verification complete")
            else:
                self.logger.warning("⚠️ Cannot verify system health - orchestrator not initialized")
                
        except Exception as e:
            self.logger.error(f"❌ System health verification failed: {e}")
    
    async def process_conversation(
        self,
        user_id: str,
        user_message: str,
        conversation_id: Optional[str] = None,
        emotional_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a complete conversation through all components"""
        
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized. Call initialize_components() first.")
        
        start_time = datetime.utcnow()
        
        try:
            # Process conversation through the complete pipeline
            result = await self.orchestrator.process_conversation(
                user_id=user_id,
                user_message=user_message,
                conversation_id=conversation_id,
                emotional_state=emotional_state
            )
            
            # Track performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            quality_score = result.get("quality_analysis", {}).get("overall_quality", 0.0)
            satisfaction_score = result.get("quality_analysis", {}).get("user_satisfaction", 0.0)
            
            self._update_performance_metrics(processing_time, True, quality_score, satisfaction_score)
            
            # Store conversation history
            conversation_record = {
                "timestamp": start_time,
                "user_id": user_id,
                "user_message": user_message,
                "ai_response": result.get("response", ""),
                "processing_time_ms": processing_time,
                "quality_score": quality_score,
                "satisfaction_score": satisfaction_score,
                "conversation_id": result.get("conversation_id")
            }
            
            self.conversation_history.append(conversation_record)
            
            self.logger.info(f"✅ Conversation processed successfully in {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time, False, 0.0, 0.0)
            
            self.logger.error(f"❌ Conversation processing failed: {e}")
            raise
    
    def _update_performance_metrics(
        self,
        processing_time: float,
        success: bool,
        quality_score: float,
        satisfaction_score: float
    ):
        """Update performance tracking metrics"""
        self.performance_metrics["total_conversations"] += 1
        
        # Update average response time
        total = self.performance_metrics["total_conversations"]
        current_avg = self.performance_metrics["avg_response_time_ms"]
        self.performance_metrics["avg_response_time_ms"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update success rate
        if success:
            current_success = self.performance_metrics["success_rate"]
            self.performance_metrics["success_rate"] = (
                (current_success * (total - 1) + 1) / total
            )
        else:
            current_success = self.performance_metrics["success_rate"]
            self.performance_metrics["success_rate"] = (
                current_success * (total - 1) / total
            )
        
        # Track quality and satisfaction scores
        if success:
            self.performance_metrics["quality_scores"].append(quality_score)
            self.performance_metrics["satisfaction_scores"].append(satisfaction_score)
    
    async def demonstrate_proactive_engagement(self, user_id: str) -> List[Dict[str, Any]]:
        """Demonstrate proactive engagement capabilities"""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")
        
        try:
            # Generate proactive message
            proactive_message = await self.orchestrator.generate_proactive_message(
                user_id=user_id,
                trigger_event="stress_pattern_detected",
                context={
                    "recent_stress_indicators": ["work_deadline", "presentation_anxiety"],
                    "suggested_intervention": "relaxation_check_in"
                }
            )
            
            # Get pending messages
            pending_messages = await self.orchestrator.get_pending_proactive_messages(user_id)
            
            self.logger.info(f"✅ Proactive engagement demo complete")
            
            return {
                "generated_message": proactive_message,
                "pending_messages": pending_messages
            }
            
        except Exception as e:
            self.logger.error(f"❌ Proactive engagement demo failed: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                "conversation_metrics": self.performance_metrics.copy(),
                "conversation_history_count": len(self.conversation_history),
                "components_status": self.components_status
            }
            
            # Calculate additional metrics
            if self.performance_metrics["quality_scores"]:
                summary["avg_quality_score"] = sum(self.performance_metrics["quality_scores"]) / len(self.performance_metrics["quality_scores"])
                summary["avg_satisfaction_score"] = sum(self.performance_metrics["satisfaction_scores"]) / len(self.performance_metrics["satisfaction_scores"])
            
            # Get orchestrator performance if available
            if self.orchestrator:
                orchestrator_metrics = self.orchestrator.get_performance_summary()
                summary["orchestrator_metrics"] = orchestrator_metrics
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Performance summary generation failed: {e}")
            return {"error": str(e)}
    
    def print_status_report(self):
        """Print comprehensive status report"""
        print("\n" + "="*80)
        print("🤖 GEMINI ENGINE INTEGRATION STATUS REPORT")
        print("="*80)
        
        # Component status
        print("\n📊 COMPONENT STATUS:")
        for component, status in self.components_status.items():
            status_icon = "✅" if status.get("status") == "initialized" else "❌"
            print(f"  {status_icon} {component.upper()}: {status.get('status', 'unknown')}")
            if status.get("error"):
                print(f"    ❌ Error: {status['error']}")
        
        # Performance metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Total Conversations: {self.performance_metrics['total_conversations']}")
        print(f"  Average Response Time: {self.performance_metrics['avg_response_time_ms']:.2f}ms")
        print(f"  Success Rate: {self.performance_metrics['success_rate']:.2%}")
        
        if self.performance_metrics["quality_scores"]:
            avg_quality = sum(self.performance_metrics["quality_scores"]) / len(self.performance_metrics["quality_scores"])
            avg_satisfaction = sum(self.performance_metrics["satisfaction_scores"]) / len(self.performance_metrics["satisfaction_scores"])
            print(f"  Average Quality Score: {avg_quality:.3f}")
            print(f"  Average Satisfaction Score: {avg_satisfaction:.3f}")
        
        # Recent conversations
        print(f"\n💬 RECENT CONVERSATIONS:")
        recent_conversations = self.conversation_history[-3:] if self.conversation_history else []
        for i, conv in enumerate(recent_conversations, 1):
            print(f"  {i}. [{conv['timestamp'].strftime('%H:%M:%S')}] Quality: {conv['quality_score']:.3f}")
            print(f"     User: {conv['user_message'][:50]}...")
            print(f"     AI: {conv['ai_response'][:50]}...")
        
        print("\n" + "="*80)


async def main():
    """Main demonstration function"""
    print("🚀 Starting Gemini Engine Complete Integration Demo")
    
    # Initialize the integration system
    engine = GeminiEngineIntegration()
    
    try:
        # Initialize all components
        print("\n1️⃣ Initializing Components...")
        await engine.initialize_components()
        
        # Print initial status
        engine.print_status_report()
        
        # Demo conversations
        print("\n2️⃣ Running Conversation Demos...")
        
        demo_conversations = [
            {
                "user_id": "user_001",
                "message": "Hi! I'm feeling a bit stressed about my upcoming presentation. Any advice?",
                "emotional_state": {"primary_emotion": "anxiety", "stress_level": 7, "intensity": 0.7}
            },
            {
                "user_id": "user_001", 
                "message": "Thanks for the advice! I actually love hiking. Do you think a nature walk would help?",
                "emotional_state": {"primary_emotion": "hopeful", "stress_level": 5, "intensity": 0.5}
            },
            {
                "user_id": "user_001",
                "message": "That sounds perfect! I'll plan a hike for this weekend.",
                "emotional_state": {"primary_emotion": "excited", "stress_level": 3, "intensity": 0.8}
            }
        ]
        
        for i, demo in enumerate(demo_conversations, 1):
            print(f"\n  Demo Conversation {i}:")
            print(f"  User: {demo['message']}")
            
            try:
                result = await engine.process_conversation(
                    user_id=demo["user_id"],
                    user_message=demo["message"],
                    emotional_state=demo["emotional_state"]
                )
                
                print(f"  AI Response: {result.get('response', 'No response')}")
                print(f"  Quality Score: {result.get('quality_analysis', {}).get('overall_quality', 0):.3f}")
                print(f"  Processing Time: {result.get('processing_time_ms', 0):.2f}ms")
                
            except Exception as e:
                print(f"  ❌ Conversation failed: {e}")
        
        # Demo proactive engagement
        print("\n3️⃣ Demonstrating Proactive Engagement...")
        try:
            proactive_result = await engine.demonstrate_proactive_engagement("user_001")
            print(f"  Generated proactive message: {proactive_result.get('generated_message')}")
        except Exception as e:
            print(f"  ❌ Proactive engagement demo failed: {e}")
        
        # Final status report
        print("\n4️⃣ Final Status Report:")
        engine.print_status_report()
        
        # Performance summary
        print("\n5️⃣ Detailed Performance Summary:")
        performance = engine.get_performance_summary()
        print(json.dumps(performance, indent=2, default=str))
        
        print("\n✅ Gemini Engine Integration Demo Complete!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Cleanup
        if engine.orchestrator:
            try:
                await engine.orchestrator.cleanup()
                print("🧹 Cleanup completed")
            except:
                pass


if __name__ == "__main__":
    # Run the complete integration demo
    asyncio.run(main())
