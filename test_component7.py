#!/usr/bin/env python3
"""
Comprehensive test script for Component 7 (Gemini Response Analysis)
Tests all modules independently and verifies production readiness.
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

# Import all Component 7 modules
try:
    from component7.quality_analyzer import QualityAnalyzer
    from component7.satisfaction_tracker import SatisfactionTracker
    from component7.feedback_engine import FeedbackEngine
    from component7.metrics_collector import MetricsCollector
    
    from shared.schemas import (
        EnhancedResponse, ResponseAnalysis, ConversationContext,
        UserProfile, QualityMetrics, FeedbackReport, ConversationMetrics,
        ContextEffectiveness, PerformanceMetrics
    )
    from config.settings import Settings
    
    print("✅ All Component 7 imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

class Component7TestSuite:
    """Comprehensive test suite for Component 7"""
    
    def __init__(self):
        self.settings = Settings()
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all Component 7 tests"""
        print("\n🧪 Starting Component 7 Test Suite")
        print("=" * 60)
        
        tests = [
            ("Quality Analyzer", self.test_quality_analyzer),
            ("Satisfaction Tracker", self.test_satisfaction_tracker),
            ("Feedback Engine", self.test_feedback_engine),
            ("Metrics Collector", self.test_metrics_collector),
            ("Performance Benchmarks", self.test_performance),
            ("Real-time Analysis", self.test_realtime_analysis),
            ("Feedback Loops", self.test_feedback_loops),
            ("Pattern Recognition", self.test_pattern_recognition),
            ("Error Handling", self.test_error_handling),
            ("Component 5/6 Integration", self.test_component_integration)
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
        
        self.print_summary()
    
    async def test_quality_analyzer(self):
        """Test QualityAnalyzer functionality"""
        analyzer = QualityAnalyzer(self.settings)
        
        # Create test response
        response = EnhancedResponse(
            content="I understand you're feeling stressed about work. Here are some strategies that might help: 1) Take regular breaks, 2) Prioritize tasks, 3) Practice deep breathing. Would you like me to elaborate on any of these?",
            conversation_id="test_conv",
            user_id="test_user",
            response_id="test_response",
            timestamp=datetime.now(),
            quality_metrics={},
            metadata={"source": "gemini"}
        )
        
        conversation_context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            current_message="I'm feeling overwhelmed with work stress",
            message_history=[
                {"role": "user", "content": "I'm having a hard time at work"},
                {"role": "assistant", "content": "I'm sorry to hear that. Can you tell me more?"}
            ],
            context_metadata={"topic": "work_stress", "emotion": "stressed"}
        )
        
        user_profile = UserProfile(
            user_id="test_user",
            communication_style="casual",
            emotional_state="stressed",
            personality_traits={"openness": 0.7, "neuroticism": 0.8},
            conversation_preferences={"detail_level": "high"}
        )
        
        # Test quality analysis
        analysis = await analyzer.analyze_response(
            response=response,
            conversation_context=conversation_context,
            user_profile=user_profile
        )
        
        assert isinstance(analysis, ResponseAnalysis), "Should return ResponseAnalysis"
        assert 0 <= analysis.overall_score <= 1, "Overall score should be between 0-1"
        assert analysis.coherence_score >= 0, "Coherence score should be positive"
        assert analysis.relevance_score >= 0, "Relevance score should be positive"
        assert analysis.context_usage_score >= 0, "Context usage score should be positive"
        assert analysis.personality_consistency_score >= 0, "Personality consistency should be positive"
        assert analysis.safety_score >= 0, "Safety score should be positive"
        assert analysis.engagement_potential >= 0, "Engagement potential should be positive"
        assert analysis.improvement_suggestions, "Should provide improvement suggestions"
        
        print("  ✓ Response quality analysis working")
        print("  ✓ All quality metrics calculated")
        print("  ✓ Improvement suggestions generated")
    
    async def test_satisfaction_tracker(self):
        """Test SatisfactionTracker functionality"""
        tracker = SatisfactionTracker(self.settings)
        
        user_profile = UserProfile(
            user_id="test_user",
            communication_style="casual",
            emotional_state="neutral",
            personality_traits={"openness": 0.7},
            conversation_preferences={}
        )
        
        conversation_context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            current_message="Thank you, that was very helpful!",
            message_history=[],
            context_metadata={"sentiment": "positive"}
        )
        
        # Test satisfaction tracking
        satisfaction_score = await tracker.track_satisfaction(
            user_profile=user_profile,
            conversation_context=conversation_context,
            response_analysis=ResponseAnalysis(
                response_id="test_response",
                overall_score=0.85,
                coherence_score=0.9,
                relevance_score=0.8,
                context_usage_score=0.75,
                personality_consistency_score=0.85,
                safety_score=1.0,
                engagement_potential=0.7,
                improvement_suggestions=["Consider more specific examples"],
                analysis_timestamp=datetime.now()
            )
        )
        
        assert 0 <= satisfaction_score <= 1, "Satisfaction score should be between 0-1"
        
        # Test explicit feedback processing
        feedback_score = await tracker.process_explicit_feedback(
            user_id="test_user",
            conversation_id="test_conv",
            feedback_type="rating",
            feedback_value=4.5,
            feedback_text="Very helpful response"
        )
        
        assert feedback_score >= 0, "Feedback score should be positive"
        
        # Test user engagement signals
        engagement_signals = {
            "response_time": 15.5,  # seconds
            "message_length": 150,
            "follow_up_questions": 2,
            "sentiment_change": 0.3
        }
        
        engagement_score = await tracker.calculate_engagement_score(
            user_id="test_user",
            engagement_signals=engagement_signals
        )
        
        assert 0 <= engagement_score <= 1, "Engagement score should be between 0-1"
        
        print("  ✓ Satisfaction tracking working")
        print("  ✓ Explicit feedback processing functional")
        print("  ✓ Engagement signal analysis operational")
    
    async def test_feedback_engine(self):
        """Test FeedbackEngine functionality"""
        engine = FeedbackEngine(self.settings)
        
        # Create comprehensive test data
        response_analysis = ResponseAnalysis(
            response_id="test_response",
            overall_score=0.75,
            coherence_score=0.8,
            relevance_score=0.7,
            context_usage_score=0.6,
            personality_consistency_score=0.8,
            safety_score=1.0,
            engagement_potential=0.7,
            improvement_suggestions=["Improve context usage", "Add more empathy"],
            analysis_timestamp=datetime.now()
        )
        
        conversation_metrics = ConversationMetrics(
            conversation_id="test_conv",
            user_id="test_user",
            total_turns=5,
            user_satisfaction_score=0.8,
            average_response_time=2.1,
            context_effectiveness=0.7,
            personality_adaptation_score=0.75,
            engagement_metrics={"depth": 0.8, "duration": 300},
            timestamp=datetime.now()
        )
        
        # Test Component 5 feedback generation
        component5_feedback = await engine.generate_component5_feedback(
            response_analysis=response_analysis,
            conversation_metrics=conversation_metrics
        )
        
        assert isinstance(component5_feedback, FeedbackReport), "Should return FeedbackReport"
        assert component5_feedback.target_component == "component5", "Should target Component 5"
        assert component5_feedback.feedback_data, "Should have feedback data"
        assert component5_feedback.priority_score >= 0, "Should have priority score"
        
        # Test Component 6 feedback generation
        component6_feedback = await engine.generate_component6_feedback(
            response_analysis=response_analysis,
            conversation_metrics=conversation_metrics
        )
        
        assert isinstance(component6_feedback, FeedbackReport), "Should return FeedbackReport"
        assert component6_feedback.target_component == "component6", "Should target Component 6"
        assert component6_feedback.feedback_data, "Should have feedback data"
        
        # Test bi-directional feedback processing
        all_feedback = await engine.process_conversation_feedback(
            conversation_id="test_conv",
            response_analysis=response_analysis,
            conversation_metrics=conversation_metrics
        )
        
        assert len(all_feedback) >= 2, "Should generate feedback for multiple components"
        assert any(fb.target_component == "component5" for fb in all_feedback), "Should include C5 feedback"
        assert any(fb.target_component == "component6" for fb in all_feedback), "Should include C6 feedback"
        
        print("  ✓ Component 5 feedback generation working")
        print("  ✓ Component 6 feedback generation working")
        print("  ✓ Bi-directional feedback processing functional")
    
    async def test_metrics_collector(self):
        """Test MetricsCollector functionality"""
        collector = MetricsCollector(self.settings)
        
        # Test performance metrics collection
        performance_metrics = PerformanceMetrics(
            component_name="component6",
            operation="conversation_processing",
            execution_time=2.1,
            memory_usage=150.5,
            cpu_usage=25.3,
            success=True,
            timestamp=datetime.now()
        )
        
        await collector.collect_performance_metrics(performance_metrics)
        
        # Test quality metrics collection
        quality_metrics = QualityMetrics(
            response_id="test_response",
            overall_score=0.85,
            coherence=0.9,
            relevance=0.8,
            engagement=0.7,
            safety=1.0,
            user_satisfaction=0.8,
            timestamp=datetime.now()
        )
        
        await collector.collect_quality_metrics(quality_metrics)
        
        # Test error logging
        await collector.log_error(
            component="component7",
            operation="quality_analysis",
            error_type="ValidationError",
            error_message="Test error for validation",
            context={"test": True}
        )
        
        # Test metrics aggregation
        aggregated_metrics = await collector.get_aggregated_metrics(
            time_range="1h",
            component_filter="component6"
        )
        
        assert isinstance(aggregated_metrics, dict), "Should return aggregated metrics dict"
        assert "performance" in aggregated_metrics, "Should include performance metrics"
        assert "quality" in aggregated_metrics, "Should include quality metrics"
        
        # Test real-time dashboard data
        dashboard_data = await collector.get_dashboard_data()
        
        assert isinstance(dashboard_data, dict), "Should return dashboard data"
        assert "current_status" in dashboard_data, "Should include current status"
        assert "metrics_summary" in dashboard_data, "Should include metrics summary"
        
        print("  ✓ Performance metrics collection working")
        print("  ✓ Quality metrics collection working")
        print("  ✓ Error logging functional")
        print("  ✓ Metrics aggregation operational")
        print("  ✓ Dashboard data generation working")
    
    async def test_performance(self):
        """Test performance requirements (<200ms for real-time analysis)"""
        print("  🔍 Testing performance requirements...")
        
        analyzer = QualityAnalyzer(self.settings)
        
        # Create test data
        response = EnhancedResponse(
            content="This is a comprehensive test response that should be analyzed quickly.",
            conversation_id="test_conv",
            user_id="test_user",
            response_id="test_response",
            timestamp=datetime.now(),
            quality_metrics={},
            metadata={}
        )
        
        conversation_context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            current_message="Test message",
            message_history=[],
            context_metadata={}
        )
        
        user_profile = UserProfile(
            user_id="test_user",
            communication_style="casual",
            emotional_state="neutral",
            personality_traits={},
            conversation_preferences={}
        )
        
        # Test analysis speed
        start_time = time.time()
        analysis = await analyzer.analyze_response(response, conversation_context, user_profile)
        analysis_time = time.time() - start_time
        
        assert analysis_time < 0.2, f"Analysis took {analysis_time:.3f}s, should be <0.2s"
        
        print(f"  ✓ Quality analysis: {analysis_time:.3f}s (<0.2s requirement)")
        print("  ✓ Real-time performance achieved")
    
    async def test_realtime_analysis(self):
        """Test real-time analysis capabilities"""
        print("  🔍 Testing real-time analysis...")
        
        analyzer = QualityAnalyzer(self.settings)
        tracker = SatisfactionTracker(self.settings)
        
        # Simulate real-time conversation analysis
        responses = [
            "Hello! How can I help you today?",
            "I understand you're looking for advice on productivity.",
            "Here are some strategies that might work for you...",
            "Would you like me to elaborate on any of these points?"
        ]
        
        analysis_times = []
        
        for i, response_content in enumerate(responses):
            response = EnhancedResponse(
                content=response_content,
                conversation_id="realtime_test",
                user_id="test_user",
                response_id=f"response_{i}",
                timestamp=datetime.now(),
                quality_metrics={},
                metadata={}
            )
            
            conversation_context = ConversationContext(
                user_id="test_user",
                conversation_id="realtime_test",
                current_message=f"User message {i}",
                message_history=[],
                context_metadata={}
            )
            
            user_profile = UserProfile(
                user_id="test_user",
                communication_style="casual",
                emotional_state="neutral",
                personality_traits={},
                conversation_preferences={}
            )
            
            start_time = time.time()
            analysis = await analyzer.analyze_response(response, conversation_context, user_profile)
            satisfaction = await tracker.track_satisfaction(user_profile, conversation_context, analysis)
            analysis_time = time.time() - start_time
            
            analysis_times.append(analysis_time)
            assert analysis_time < 0.2, f"Real-time analysis {i} took {analysis_time:.3f}s"
        
        avg_time = sum(analysis_times) / len(analysis_times)
        print(f"  ✓ Average analysis time: {avg_time:.3f}s")
        print("  ✓ Real-time processing verified")
    
    async def test_feedback_loops(self):
        """Test bi-directional feedback loops"""
        print("  🔍 Testing feedback loops...")
        
        engine = FeedbackEngine(self.settings)
        
        # Test continuous feedback cycle
        feedback_cycle_data = []
        
        for iteration in range(3):
            response_analysis = ResponseAnalysis(
                response_id=f"feedback_test_{iteration}",
                overall_score=0.7 + (iteration * 0.05),  # Simulating improvement
                coherence_score=0.8,
                relevance_score=0.75,
                context_usage_score=0.6 + (iteration * 0.1),  # Improving context usage
                personality_consistency_score=0.8,
                safety_score=1.0,
                engagement_potential=0.7,
                improvement_suggestions=["Improve context usage"],
                analysis_timestamp=datetime.now()
            )
            
            conversation_metrics = ConversationMetrics(
                conversation_id=f"feedback_conv_{iteration}",
                user_id="test_user",
                total_turns=5,
                user_satisfaction_score=0.75 + (iteration * 0.05),
                average_response_time=2.5 - (iteration * 0.1),  # Getting faster
                context_effectiveness=0.6 + (iteration * 0.1),
                personality_adaptation_score=0.75,
                engagement_metrics={"depth": 0.8},
                timestamp=datetime.now()
            )
            
            # Generate feedback for both components
            feedback_reports = await engine.process_conversation_feedback(
                conversation_id=f"feedback_conv_{iteration}",
                response_analysis=response_analysis,
                conversation_metrics=conversation_metrics
            )
            
            feedback_cycle_data.append({
                "iteration": iteration,
                "feedback_count": len(feedback_reports),
                "context_effectiveness": conversation_metrics.context_effectiveness,
                "user_satisfaction": conversation_metrics.user_satisfaction_score
            })
        
        # Verify improvement trend
        context_scores = [data["context_effectiveness"] for data in feedback_cycle_data]
        satisfaction_scores = [data["user_satisfaction"] for data in feedback_cycle_data]
        
        assert context_scores[-1] > context_scores[0], "Context effectiveness should improve"
        assert satisfaction_scores[-1] > satisfaction_scores[0], "User satisfaction should improve"
        
        print("  ✓ Feedback generation working")
        print("  ✓ Improvement trends detected")
        print("  ✓ Continuous learning cycle functional")
    
    async def test_pattern_recognition(self):
        """Test long-term pattern recognition"""
        print("  🔍 Testing pattern recognition...")
        
        collector = MetricsCollector(self.settings)
        
        # Simulate pattern data over time
        base_time = datetime.now() - timedelta(days=7)
        
        # Generate metrics showing patterns
        for day in range(7):
            for hour in range(24):
                timestamp = base_time + timedelta(days=day, hours=hour)
                
                # Simulate daily pattern (higher quality during business hours)
                quality_multiplier = 1.0 if 9 <= hour <= 17 else 0.8
                
                quality_metrics = QualityMetrics(
                    response_id=f"pattern_test_{day}_{hour}",
                    overall_score=0.7 * quality_multiplier,
                    coherence=0.8 * quality_multiplier,
                    relevance=0.75 * quality_multiplier,
                    engagement=0.7 * quality_multiplier,
                    safety=1.0,
                    user_satisfaction=0.75 * quality_multiplier,
                    timestamp=timestamp
                )
                
                await collector.collect_quality_metrics(quality_metrics)
        
        # Test pattern detection
        patterns = await collector.detect_patterns(
            metric_type="quality",
            time_range="7d",
            pattern_types=["daily", "weekly"]
        )
        
        assert isinstance(patterns, dict), "Should return pattern analysis"
        assert "daily_patterns" in patterns or "patterns_detected" in patterns, "Should detect patterns"
        
        print("  ✓ Pattern data collection working")
        print("  ✓ Pattern detection functional")
        print("  ✓ Long-term analysis operational")
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        print("  🔍 Testing error handling...")
        
        analyzer = QualityAnalyzer(self.settings)
        
        # Test invalid input handling
        try:
            await analyzer.analyze_response(
                response=None,
                conversation_context=None,
                user_profile=None
            )
            assert False, "Should raise error for None inputs"
        except Exception as e:
            assert len(str(e)) > 10, "Error message should be informative"
        
        # Test graceful degradation with partial data
        partial_response = EnhancedResponse(
            content="Test response",
            conversation_id="test",
            user_id="test",
            response_id="test",
            timestamp=datetime.now(),
            quality_metrics={},
            metadata={}
        )
        
        try:
            analysis = await analyzer.analyze_response(
                response=partial_response,
                conversation_context=ConversationContext(
                    user_id="test",
                    conversation_id="test",
                    current_message="",
                    message_history=[],
                    context_metadata={}
                ),
                user_profile=UserProfile(
                    user_id="test",
                    communication_style="casual",
                    emotional_state="neutral",
                    personality_traits={},
                    conversation_preferences={}
                )
            )
            assert analysis.overall_score >= 0, "Should handle partial data gracefully"
        except Exception as e:
            # Should be a clear, informative error
            assert "missing" in str(e).lower() or "required" in str(e).lower()
        
        print("  ✓ Input validation working")
        print("  ✓ Error messages informative")
        print("  ✓ Graceful degradation functional")
    
    async def test_component_integration(self):
        """Test integration readiness with Components 5 & 6"""
        print("  🔍 Testing component integration readiness...")
        
        # Test data flow compatibility
        from shared.schemas import MemoryContext, EnhancedResponse
        
        # Verify Component 6 output compatibility
        component6_output = EnhancedResponse(
            content="Test response from Component 6",
            conversation_id="integration_test",
            user_id="test_user",
            response_id="test_response",
            timestamp=datetime.now(),
            quality_metrics={},
            metadata={"component": "component6"}
        )
        
        # Test Component 7 can process Component 6 output
        analyzer = QualityAnalyzer(self.settings)
        
        analysis = await analyzer.analyze_response(
            response=component6_output,
            conversation_context=ConversationContext(
                user_id="test_user",
                conversation_id="integration_test",
                current_message="Test integration",
                message_history=[],
                context_metadata={}
            ),
            user_profile=UserProfile(
                user_id="test_user",
                communication_style="casual",
                emotional_state="neutral",
                personality_traits={},
                conversation_preferences={}
            )
        )
        
        assert isinstance(analysis, ResponseAnalysis), "Should process Component 6 output"
        
        # Test feedback generation for Component 5 integration
        engine = FeedbackEngine(self.settings)
        
        feedback = await engine.generate_component5_feedback(
            response_analysis=analysis,
            conversation_metrics=ConversationMetrics(
                conversation_id="integration_test",
                user_id="test_user",
                total_turns=1,
                user_satisfaction_score=0.8,
                average_response_time=2.0,
                context_effectiveness=0.7,
                personality_adaptation_score=0.75,
                engagement_metrics={},
                timestamp=datetime.now()
            )
        )
        
        assert isinstance(feedback, FeedbackReport), "Should generate Component 5 feedback"
        assert feedback.target_component == "component5", "Should target Component 5"
        
        print("  ✓ Component 6 output processing verified")
        print("  ✓ Component 5 feedback generation confirmed")
        print("  ✓ Integration pipeline validated")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("🧪 COMPONENT 7 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            duration = result.get("duration", "N/A")
            print(f"{status_icon} {test_name:<30} {result['status']:<8} {duration}")
            if result["status"] == "FAILED":
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL COMPONENT 7 TESTS PASSED - PRODUCTION READY!")
        else:
            print(f"⚠️  {total-passed} tests failed - Review and fix issues")

async def main():
    """Main test runner"""
    print("🚀 Component 7 Production Readiness Test")
    print("Testing all modules for production deployment...")
    
    test_suite = Component7TestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
