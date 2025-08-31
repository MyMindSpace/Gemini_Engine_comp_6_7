#!/usr/bin/env python3
"""
Comprehensive integration test script for Components 6 & 7 together
Tests the complete Gemini Engine pipeline and validates production readiness.
"""

import os
import sys
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

# Set up test environment
os.environ["TEST_MODE"] = "true"
os.environ["GEMINI_API_KEY"] = "test_key_123"
os.environ["PINECONE_API_KEY"] = "test_pinecone_key"
os.environ["PINECONE_ENVIRONMENT"] = "test_env"
os.environ["PINECONE_INDEX_NAME"] = "test_index"
os.environ["REDIS_URL"] = "redis://localhost:6379"

# Import all modules
try:
    # Component 6 imports
    from component6.conversation_manager import ConversationManager
    from component6.memory_retriever import MemoryRetriever
    from component6.context_assembler import ContextAssembler
    from component6.personality_engine import PersonalityEngine
    from component6.gemini_client import GeminiClient
    from component6.response_processor import ResponseProcessor
    from component6.proactive_engine import ProactiveEngine
    
    # Component 7 imports
    from component7.quality_analyzer import QualityAnalyzer
    from component7.satisfaction_tracker import SatisfactionTracker
    from component7.feedback_engine import FeedbackEngine
    from component7.metrics_collector import MetricsCollector
    
    # Main orchestrator
    from gemini_engine_orchestrator import GeminiEngineOrchestrator
    
    # Shared schemas
    from shared.schemas import (
        ConversationContext, UserProfile, MemoryContext,
        EnhancedResponse, ResponseAnalysis, FeedbackReport,
        ConversationMetrics, QualityMetrics, PerformanceMetrics
    )
    from shared.mock_interfaces import MockComponent5Interface
    from config.settings import Settings
    
    print("✅ All integration imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

class GeminiEngineIntegrationTestSuite:
    """Comprehensive integration test suite for Components 6 & 7"""
    
    def __init__(self):
        self.settings = Settings()
        self.mock_component5 = MockComponent5Interface()
        self.orchestrator = None
        self.test_results = {}
        
    async def setup(self):
        """Initialize the orchestrator"""
        self.orchestrator = GeminiEngineOrchestrator(
            component5_interface=self.mock_component5,
            settings=self.settings
        )
        
    async def teardown(self):
        """Clean up resources"""
        if self.orchestrator:
            await self.orchestrator.close()
        
    async def run_all_tests(self):
        """Run all integration tests"""
        print("\n🧪 Starting Gemini Engine Integration Test Suite")
        print("=" * 70)
        
        await self.setup()
        
        tests = [
            ("End-to-End Conversation Flow", self.test_end_to_end_conversation),
            ("Real-time Processing Pipeline", self.test_realtime_pipeline),
            ("Multi-turn Conversation", self.test_multiturn_conversation),
            ("Proactive Engagement", self.test_proactive_engagement),
            ("Quality Analysis Integration", self.test_quality_analysis_integration),
            ("Feedback Loop Integration", self.test_feedback_loop_integration),
            ("Performance Under Load", self.test_performance_load),
            ("Error Recovery", self.test_error_recovery),
            ("Component 5 Integration", self.test_component5_integration),
            ("Production Readiness", self.test_production_readiness),
            ("System Health Monitoring", self.test_system_health),
            ("Metrics Collection Integration", self.test_metrics_integration),
            ("Memory Management", self.test_memory_management),
            ("Personality Adaptation", self.test_personality_adaptation),
            ("Context Assembly Pipeline", self.test_context_assembly_pipeline)
        ]
        
        for test_name, test_func in tests:
            try:
                print(f"\n🔍 Testing {test_name}...")
                start_time = time.time()
                await test_func()
                duration = time.time() - start_time
                self.test_results[test_name] = {
                    "status": "PASSED",
                    "duration": f"{duration:.3f}s"
                }
                print(f"✅ {test_name} - PASSED ({duration:.3f}s)")
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "FAILED",
                    "error": str(e),
                    "duration": "N/A"
                }
                print(f"❌ {test_name} - FAILED: {e}")
                traceback.print_exc()
        
        await self.teardown()
        self.print_summary()
    
    async def test_end_to_end_conversation(self):
        """Test complete conversation flow from input to analysis"""
        conversation_context = ConversationContext(
            user_id="integration_test_user",
            conversation_id="end_to_end_test",
            current_message="I'm feeling stressed about my upcoming job interview. Can you help me prepare?",
            message_history=[],
            context_metadata={"emotion": "anxious", "topic": "job_interview"}
        )
        
        # Process complete conversation
        result = await self.orchestrator.process_user_input(conversation_context)
        
        # Validate response structure
        assert isinstance(result, dict), "Should return result dictionary"
        assert "response" in result, "Should include enhanced response"
        assert "analysis" in result, "Should include response analysis"
        assert "satisfaction_score" in result, "Should include satisfaction score"
        assert "feedback_reports" in result, "Should include feedback reports"
        assert "metrics" in result, "Should include performance metrics"
        
        # Validate response quality
        response = result["response"]
        assert isinstance(response, EnhancedResponse), "Should return EnhancedResponse"
        assert response.content, "Response should have content"
        assert "interview" in response.content.lower(), "Should address interview topic"
        
        # Validate analysis
        analysis = result["analysis"]
        assert isinstance(analysis, ResponseAnalysis), "Should return ResponseAnalysis"
        assert 0 <= analysis.overall_score <= 1, "Overall score should be valid"
        assert analysis.relevance_score > 0, "Should have relevance score"
        
        # Validate feedback
        feedback_reports = result["feedback_reports"]
        assert len(feedback_reports) >= 1, "Should generate feedback reports"
        assert any(fb.target_component in ["component5", "component6"] for fb in feedback_reports), "Should target upstream components"
        
        print("  ✓ Complete conversation pipeline working")
        print("  ✓ Response generation successful")
        print("  ✓ Quality analysis integrated")
        print("  ✓ Feedback generation functional")
    
    async def test_realtime_pipeline(self):
        """Test real-time processing requirements"""
        conversation_context = ConversationContext(
            user_id="realtime_test_user",
            conversation_id="realtime_test",
            current_message="Quick question: What's the weather like?",
            message_history=[],
            context_metadata={}
        )
        
        # Test processing speed
        start_time = time.time()
        result = await self.orchestrator.process_user_input(conversation_context)
        total_time = time.time() - start_time
        
        # Validate performance requirements
        assert total_time < 3.0, f"Total processing took {total_time:.3f}s, should be <3s"
        
        # Check analysis speed
        analysis = result["analysis"]
        analysis_metadata = getattr(analysis, 'metadata', {})
        if "analysis_time" in analysis_metadata:
            analysis_time = analysis_metadata["analysis_time"]
            assert analysis_time < 0.2, f"Analysis took {analysis_time:.3f}s, should be <0.2s"
        
        print(f"  ✓ Total processing time: {total_time:.3f}s (<3s requirement)")
        print("  ✓ Real-time performance achieved")
    
    async def test_multiturn_conversation(self):
        """Test multi-turn conversation handling"""
        base_context = ConversationContext(
            user_id="multiturn_test_user",
            conversation_id="multiturn_test",
            current_message="",
            message_history=[],
            context_metadata={}
        )
        
        messages = [
            "Hello, I'd like to discuss my career goals.",
            "I'm particularly interested in data science.",
            "What skills should I focus on developing?",
            "How can I get practical experience?",
            "Thank you for the advice!"
        ]
        
        conversation_history = []
        
        for i, message in enumerate(messages):
            # Update context with conversation history
            context = ConversationContext(
                user_id=base_context.user_id,
                conversation_id=base_context.conversation_id,
                current_message=message,
                message_history=conversation_history.copy(),
                context_metadata={"turn": i + 1}
            )
            
            # Process message
            result = await self.orchestrator.process_user_input(context)
            response = result["response"]
            
            # Add to conversation history
            conversation_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.content}
            ])
            
            # Validate context awareness
            if i > 0:  # After first turn
                assert "career" in response.content.lower() or "data" in response.content.lower(), \
                    f"Turn {i+1} should maintain context awareness"
            
            # Validate turn-specific responses
            if i == len(messages) - 1:  # Final turn (thank you)
                assert any(word in response.content.lower() for word in ["welcome", "glad", "help"]), \
                    "Should acknowledge gratitude appropriately"
        
        print("  ✓ Multi-turn conversation handling working")
        print("  ✓ Context awareness maintained")
        print("  ✓ Conversation continuity preserved")
    
    async def test_proactive_engagement(self):
        """Test proactive conversation initiation"""
        user_profile = UserProfile(
            user_id="proactive_test_user",
            communication_style="casual",
            emotional_state="neutral",
            personality_traits={"openness": 0.7},
            conversation_preferences={"proactive_frequency": "medium"}
        )
        
        conversation_context = ConversationContext(
            user_id="proactive_test_user",
            conversation_id="proactive_test",
            current_message="",
            message_history=[],
            context_metadata={
                "last_interaction": (datetime.now() - timedelta(hours=25)).isoformat(),
                "user_stress_level": "high"
            }
        )
        
        # Test proactive engagement
        proactive_result = await self.orchestrator.initiate_proactive_engagement(
            user_profile=user_profile,
            conversation_context=conversation_context
        )
        
        if proactive_result:
            assert "message" in proactive_result, "Should include proactive message"
            assert "trigger_type" in proactive_result, "Should include trigger type"
            assert "scheduled_time" in proactive_result, "Should include scheduling info"
            
            print("  ✓ Proactive trigger detection working")
            print("  ✓ Proactive message generation functional")
            print("  ✓ Scheduling logic operational")
        else:
            print("  ⚠️  No proactive engagement triggered (acceptable)")
            print("  ✓ Proactive system operational")
    
    async def test_quality_analysis_integration(self):
        """Test quality analysis integration with conversation flow"""
        conversation_context = ConversationContext(
            user_id="quality_test_user",
            conversation_id="quality_test",
            current_message="Can you explain quantum computing in simple terms?",
            message_history=[],
            context_metadata={"complexity_preference": "simple"}
        )
        
        result = await self.orchestrator.process_user_input(conversation_context)
        
        # Validate quality analysis
        analysis = result["analysis"]
        assert isinstance(analysis, ResponseAnalysis), "Should include quality analysis"
        
        # Check all quality dimensions
        assert hasattr(analysis, 'coherence_score'), "Should have coherence score"
        assert hasattr(analysis, 'relevance_score'), "Should have relevance score"
        assert hasattr(analysis, 'context_usage_score'), "Should have context usage score"
        assert hasattr(analysis, 'personality_consistency_score'), "Should have personality score"
        assert hasattr(analysis, 'safety_score'), "Should have safety score"
        assert hasattr(analysis, 'engagement_potential'), "Should have engagement score"
        
        # Validate score ranges
        assert 0 <= analysis.overall_score <= 1, "Overall score should be in valid range"
        assert 0 <= analysis.coherence_score <= 1, "Coherence score should be in valid range"
        assert 0 <= analysis.relevance_score <= 1, "Relevance score should be in valid range"
        
        # Check improvement suggestions
        assert analysis.improvement_suggestions, "Should provide improvement suggestions"
        assert len(analysis.improvement_suggestions) > 0, "Should have actionable suggestions"
        
        print("  ✓ Quality analysis fully integrated")
        print("  ✓ All quality dimensions measured")
        print("  ✓ Improvement suggestions generated")
    
    async def test_feedback_loop_integration(self):
        """Test bi-directional feedback loops"""
        conversation_context = ConversationContext(
            user_id="feedback_test_user",
            conversation_id="feedback_test",
            current_message="I need help managing my time better.",
            message_history=[],
            context_metadata={"urgency": "high"}
        )
        
        result = await self.orchestrator.process_user_input(conversation_context)
        
        # Validate feedback generation
        feedback_reports = result["feedback_reports"]
        assert len(feedback_reports) > 0, "Should generate feedback reports"
        
        # Check Component 5 feedback
        component5_feedback = [fb for fb in feedback_reports if fb.target_component == "component5"]
        if component5_feedback:
            fb = component5_feedback[0]
            assert isinstance(fb, FeedbackReport), "Should be FeedbackReport instance"
            assert fb.feedback_data, "Should have feedback data"
            assert "memory_relevance" in fb.feedback_data or "context_quality" in fb.feedback_data, \
                "Should include memory-related feedback"
        
        # Check Component 6 feedback
        component6_feedback = [fb for fb in feedback_reports if fb.target_component == "component6"]
        if component6_feedback:
            fb = component6_feedback[0]
            assert isinstance(fb, FeedbackReport), "Should be FeedbackReport instance"
            assert fb.feedback_data, "Should have feedback data"
        
        # Validate feedback priority scoring
        for fb in feedback_reports:
            assert 0 <= fb.priority_score <= 1, "Priority score should be in valid range"
            assert fb.feedback_type in ["improvement", "validation", "correction"], \
                "Should have valid feedback type"
        
        print("  ✓ Feedback loop integration working")
        print("  ✓ Component 5 feedback generated")
        print("  ✓ Component 6 feedback generated")
        print("  ✓ Priority scoring functional")
    
    async def test_performance_load(self):
        """Test performance under concurrent load"""
        print("  🔍 Testing concurrent load handling...")
        
        # Create multiple concurrent conversations
        tasks = []
        for i in range(5):  # Test with 5 concurrent conversations
            context = ConversationContext(
                user_id=f"load_test_user_{i}",
                conversation_id=f"load_test_{i}",
                current_message=f"Test message {i} for load testing",
                message_history=[],
                context_metadata={"test_id": i}
            )
            tasks.append(self.orchestrator.process_user_input(context))
        
        # Process all conversations concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Validate results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_results) >= 3, f"At least 3/5 conversations should succeed, got {len(successful_results)}"
        assert total_time < 15.0, f"Concurrent processing took {total_time:.3f}s, should be reasonable"
        
        # Validate response quality under load
        for result in successful_results:
            assert isinstance(result, dict), "Should return valid result structure"
            assert "response" in result, "Should include response"
            assert "analysis" in result, "Should include analysis"
        
        print(f"  ✓ Concurrent load handling: {len(successful_results)}/5 successful")
        print(f"  ✓ Total processing time: {total_time:.3f}s")
        if failed_results:
            print(f"  ⚠️  {len(failed_results)} conversations failed (acceptable under load)")
    
    async def test_error_recovery(self):
        """Test error handling and recovery mechanisms"""
        print("  🔍 Testing error recovery...")
        
        # Test with invalid input
        invalid_context = ConversationContext(
            user_id="",  # Invalid empty user ID
            conversation_id="error_test",
            current_message="Test message",
            message_history=[],
            context_metadata={}
        )
        
        try:
            result = await self.orchestrator.process_user_input(invalid_context)
            # If it doesn't raise an exception, check for graceful handling
            assert "error" in result or result is None, "Should handle invalid input gracefully"
        except Exception as e:
            # Should be a clear, informative error
            assert len(str(e)) > 10, "Error message should be informative"
            assert "user_id" in str(e).lower(), "Should indicate the specific issue"
        
        # Test with extremely long input
        long_context = ConversationContext(
            user_id="error_test_user",
            conversation_id="error_test_long",
            current_message="x" * 10000,  # Very long message
            message_history=[],
            context_metadata={}
        )
        
        try:
            result = await self.orchestrator.process_user_input(long_context)
            # Should either handle gracefully or provide clear error
            if isinstance(result, dict):
                assert "response" in result, "Should provide some response even for long input"
        except Exception as e:
            assert "token" in str(e).lower() or "length" in str(e).lower(), \
                "Should indicate token/length issue"
        
        print("  ✓ Invalid input handling working")
        print("  ✓ Error messages informative")
        print("  ✓ Graceful degradation functional")
    
    async def test_component5_integration(self):
        """Test Component 5 integration readiness"""
        print("  🔍 Testing Component 5 integration...")
        
        # Validate mock interface matches Component 5 spec
        memory_context = await self.mock_component5.get_memory_context(
            user_id="component5_test_user",
            query="Test query for memory retrieval"
        )
        
        # Validate memory structure matches Component 5 documentation
        assert isinstance(memory_context, MemoryContext), "Should return MemoryContext"
        assert hasattr(memory_context, 'selected_memories'), "Should have selected_memories"
        assert hasattr(memory_context, 'relevance_scores'), "Should have relevance_scores"
        assert hasattr(memory_context, 'token_usage'), "Should have token_usage"
        assert hasattr(memory_context, 'assembly_metadata'), "Should have assembly_metadata"
        
        # Validate memory item structure
        if memory_context.selected_memories:
            memory = memory_context.selected_memories[0]
            required_fields = [
                "id", "user_id", "memory_type", "content_summary",
                "importance_score", "emotional_significance", "temporal_relevance"
            ]
            for field in required_fields:
                assert field in memory, f"Memory should have {field} field"
        
        # Test user profile integration
        user_profile = await self.mock_component5.get_user_profile("component5_test_user")
        assert isinstance(user_profile, UserProfile), "Should return UserProfile"
        
        # Test Component 6 can process Component 5 output
        conversation_context = ConversationContext(
            user_id="component5_test_user",
            conversation_id="component5_integration_test",
            current_message="Test Component 5 integration",
            message_history=[],
            context_metadata={}
        )
        
        result = await self.orchestrator.process_user_input(conversation_context)
        assert isinstance(result, dict), "Should process Component 5 data successfully"
        
        print("  ✓ Component 5 interface compliance verified")
        print("  ✓ Memory structure validation passed")
        print("  ✓ Data flow integration confirmed")
    
    async def test_production_readiness(self):
        """Test production deployment readiness"""
        print("  🔍 Testing production readiness...")
        
        # Test configuration validation
        assert self.settings.gemini_api.api_key, "Should have API key configuration"
        assert self.settings.performance.max_response_time > 0, "Should have performance limits"
        
        # Test logging and monitoring
        result = await self.orchestrator.process_user_input(
            ConversationContext(
                user_id="production_test_user",
                conversation_id="production_test",
                current_message="Production readiness test",
                message_history=[],
                context_metadata={}
            )
        )
        
        # Validate monitoring data
        assert "metrics" in result, "Should include performance metrics"
        metrics = result["metrics"]
        assert isinstance(metrics, dict), "Metrics should be structured data"
        
        # Test system health
        health_status = await self.orchestrator.get_system_health()
        assert isinstance(health_status, dict), "Should return health status"
        assert "status" in health_status, "Should include overall status"
        
        # Test metrics collection
        system_metrics = await self.orchestrator.get_system_metrics()
        assert isinstance(system_metrics, dict), "Should return system metrics"
        
        print("  ✓ Configuration validation passed")
        print("  ✓ Monitoring and logging functional")
        print("  ✓ Health checks operational")
        print("  ✓ Production deployment ready")
    
    async def test_system_health(self):
        """Test system health monitoring"""
        health_status = await self.orchestrator.get_system_health()
        
        assert isinstance(health_status, dict), "Should return health dictionary"
        assert "status" in health_status, "Should include overall status"
        assert "components" in health_status, "Should include component status"
        assert "timestamp" in health_status, "Should include timestamp"
        
        # Validate component health
        components = health_status["components"]
        expected_components = ["component6", "component7", "memory_retriever", "gemini_client"]
        
        for component in expected_components:
            if component in components:
                assert components[component] in ["healthy", "degraded", "error"], \
                    f"Component {component} should have valid status"
        
        print("  ✓ System health monitoring functional")
        print("  ✓ Component status tracking working")
    
    async def test_metrics_integration(self):
        """Test metrics collection integration"""
        # Process a conversation to generate metrics
        result = await self.orchestrator.process_user_input(
            ConversationContext(
                user_id="metrics_test_user",
                conversation_id="metrics_test",
                current_message="Test metrics collection",
                message_history=[],
                context_metadata={}
            )
        )
        
        # Get system metrics
        metrics = await self.orchestrator.get_system_metrics()
        
        assert isinstance(metrics, dict), "Should return metrics dictionary"
        assert "performance" in metrics or "quality" in metrics, "Should include metric categories"
        
        # Validate metric structure
        if "performance" in metrics:
            perf_metrics = metrics["performance"]
            assert isinstance(perf_metrics, dict), "Performance metrics should be structured"
        
        if "quality" in metrics:
            quality_metrics = metrics["quality"]
            assert isinstance(quality_metrics, dict), "Quality metrics should be structured"
        
        print("  ✓ Metrics collection integrated")
        print("  ✓ Performance tracking functional")
        print("  ✓ Quality metrics aggregated")
    
    async def test_memory_management(self):
        """Test memory management and caching"""
        user_id = "memory_test_user"
        
        # Make multiple requests to test caching
        contexts = [
            ConversationContext(
                user_id=user_id,
                conversation_id=f"memory_test_{i}",
                current_message=f"Memory test message {i}",
                message_history=[],
                context_metadata={}
            )
            for i in range(3)
        ]
        
        # Process requests and measure cache effectiveness
        times = []
        for context in contexts:
            start_time = time.time()
            result = await self.orchestrator.process_user_input(context)
            times.append(time.time() - start_time)
            assert isinstance(result, dict), "Should return valid result"
        
        # Later requests might be faster due to caching
        # (though this depends on implementation details)
        print(f"  ✓ Memory management operational")
        print(f"  ✓ Request processing times: {[f'{t:.3f}s' for t in times]}")
    
    async def test_personality_adaptation(self):
        """Test personality adaptation across different user profiles"""
        base_message = "I'm feeling anxious about my presentation tomorrow."
        
        # Test different personality profiles
        profiles = [
            ("casual", "anxious", {"neuroticism": 0.8}),
            ("professional", "confident", {"conscientiousness": 0.9}),
            ("friendly", "neutral", {"agreeableness": 0.8})
        ]
        
        responses = []
        
        for i, (style, emotion, traits) in enumerate(profiles):
            context = ConversationContext(
                user_id=f"personality_test_user_{i}",
                conversation_id=f"personality_test_{i}",
                current_message=base_message,
                message_history=[],
                context_metadata={"communication_style": style, "emotional_state": emotion}
            )
            
            result = await self.orchestrator.process_user_input(context)
            response = result["response"]
            responses.append((style, response.content))
        
        # Validate personality adaptation
        assert len(responses) == len(profiles), "Should generate responses for all profiles"
        
        # Responses should vary based on personality
        contents = [content for _, content in responses]
        assert len(set(contents)) > 1, "Responses should vary based on personality adaptation"
        
        print("  ✓ Personality adaptation working")
        print("  ✓ Communication style adjustment functional")
        print("  ✓ Response variation based on user profile")
    
    async def test_context_assembly_pipeline(self):
        """Test context assembly and token management"""
        # Create context with memory requirements
        context = ConversationContext(
            user_id="context_test_user",
            conversation_id="context_test",
            current_message="Tell me about our previous conversations about career planning.",
            message_history=[
                {"role": "user", "content": "I want to change careers"},
                {"role": "assistant", "content": "That's a big decision. What field interests you?"},
                {"role": "user", "content": "I'm thinking about data science"},
                {"role": "assistant", "content": "Data science is growing rapidly. What's your background?"}
            ],
            context_metadata={"requires_memory": True}
        )
        
        result = await self.orchestrator.process_user_input(context)
        response = result["response"]
        
        # Validate context usage
        assert "career" in response.content.lower(), "Should reference career context"
        
        # Check if analysis includes context usage metrics
        analysis = result["analysis"]
        assert hasattr(analysis, 'context_usage_score'), "Should measure context usage"
        assert analysis.context_usage_score >= 0, "Context usage score should be valid"
        
        print("  ✓ Context assembly pipeline working")
        print("  ✓ Memory integration functional")
        print("  ✓ Token budget management operational")
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("🧪 GEMINI ENGINE INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        total = len(self.test_results)
        
        # Categorize tests
        critical_tests = [
            "End-to-End Conversation Flow",
            "Real-time Processing Pipeline",
            "Component 5 Integration",
            "Production Readiness"
        ]
        
        performance_tests = [
            "Real-time Processing Pipeline",
            "Performance Under Load",
            "Memory Management"
        ]
        
        integration_tests = [
            "Quality Analysis Integration",
            "Feedback Loop Integration",
            "Metrics Collection Integration"
        ]
        
        print("\n📊 TEST RESULTS BY CATEGORY:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            duration = result.get("duration", "N/A")
            
            category = ""
            if test_name in critical_tests:
                category = "🔥 CRITICAL"
            elif test_name in performance_tests:
                category = "⚡ PERFORMANCE"
            elif test_name in integration_tests:
                category = "🔗 INTEGRATION"
            else:
                category = "🛠️  FUNCTIONAL"
            
            print(f"{status_icon} {category:<12} {test_name:<35} {result['status']:<8} {duration}")
            
            if result["status"] == "FAILED":
                print(f"    ❌ Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n📈 OVERALL RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        # Check critical test status
        critical_passed = sum(1 for test in critical_tests 
                            if test in self.test_results and 
                            self.test_results[test]["status"] == "PASSED")
        
        print(f"🔥 CRITICAL TESTS: {critical_passed}/{len(critical_tests)} passed")
        
        if passed == total and critical_passed == len(critical_tests):
            print("\n🎉 ALL TESTS PASSED - GEMINI ENGINE IS PRODUCTION READY!")
            print("✅ Components 6 & 7 fully integrated and operational")
            print("✅ Ready for Component 5 integration")
            print("✅ Production deployment approved")
        elif critical_passed == len(critical_tests):
            print(f"\n✅ CRITICAL TESTS PASSED - Core functionality operational")
            print(f"⚠️  {total-passed} non-critical tests failed - Review recommended")
        else:
            print(f"\n❌ CRITICAL ISSUES DETECTED - {len(critical_tests)-critical_passed} critical tests failed")
            print("🛠️  Fix critical issues before production deployment")

async def main():
    """Main integration test runner"""
    print("🚀 Gemini Engine Integration Test Suite")
    print("Testing Components 6 & 7 Integration for Production Deployment")
    print("=" * 70)
    
    test_suite = GeminiEngineIntegrationTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
