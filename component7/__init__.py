"""
Component 7: Gemini Response Analysis Package.
Monitors and analyzes Gemini response quality, user satisfaction, and conversation effectiveness.
"""

from .quality_analyzer import QualityAnalyzer
from .satisfaction_tracker import SatisfactionTracker
from .feedback_engine import FeedbackEngine
from .metrics_collector import MetricsCollector

__all__ = [
    "QualityAnalyzer",
    "SatisfactionTracker", 
    "FeedbackEngine",
    "MetricsCollector"
]

# Version information
__version__ = "1.0.0"
__author__ = "Gemini Engine Team"
__description__ = "Component 7: Gemini Response Analysis and Quality Monitoring"
