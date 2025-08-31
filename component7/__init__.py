"""
Component 7: Gemini Response Analysis
Main orchestrator and analysis coordination module.
"""

from .quality_analyzer import QualityAnalyzer
from .satisfaction_tracker import SatisfactionTracker
from .feedback_engine import FeedbackEngine
from .metrics_collector import MetricsCollector

__all__ = [
    "QualityAnalyzer",
    "SatisfactionTracker", 
    "FeedbackEngine",
    "MetricsCollector",
    "Component7Orchestrator"
]


class Component7Orchestrator:
    """Main orchestrator for Component 7 analysis and feedback"""
    
    def __init__(self):
        """Initialize Component 7 orchestrator"""
        self.quality_analyzer = QualityAnalyzer()
        self.satisfaction_tracker = SatisfactionTracker()
        self.feedback_engine = FeedbackEngine()
        self.metrics_collector = MetricsCollector()
    
    async def analyze_response(self, response, context_used, user_satisfaction=None):
        """Complete response analysis pipeline"""
        # Analyze response quality
        analysis = await self.quality_analyzer.analyze_response(
            response, context_used, user_satisfaction
        )
        
        # Track user satisfaction
        satisfaction_data = await self.satisfaction_tracker.track_user_satisfaction(
            response.user_id if hasattr(response, 'user_id') else "unknown",
            analysis
        )
        
        # Collect metrics
        await self.metrics_collector.collect_component_metrics(
            "component7",
            {
                "quality_score": analysis.quality_scores.get("overall_quality", 0.0),
                "satisfaction_score": satisfaction_data.get("satisfaction_score", 0.0),
                "processing_time_ms": 100  # placeholder
            }
        )
        
        return {
            "analysis": analysis,
            "satisfaction": satisfaction_data
        }
    
    async def generate_feedback(self, component_target, analysis_results):
        """Generate feedback for upstream components"""
        return await self.feedback_engine.generate_component_feedback(
            component_target, analysis_results
        )
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        return {
            "quality_analyzer": self.quality_analyzer.get_performance_metrics(),
            "satisfaction_tracker": self.satisfaction_tracker.get_performance_metrics(),
            "feedback_engine": self.feedback_engine.get_performance_metrics(),
            "metrics_collector": self.metrics_collector.get_performance_metrics()
        }