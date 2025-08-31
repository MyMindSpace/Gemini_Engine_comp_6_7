"""
Quality Analyzer Module for Component 7.
Assesses response quality, coherence, and relevance for AI responses.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import json

from shared.schemas import (
    EnhancedResponse, QualityMetrics, ResponseAnalysis
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id,
    extract_keywords, clean_text
)


class QualityAnalyzer:
    """Analyzes response quality and generates improvement suggestions"""
    
    def __init__(self):
        """Initialize quality analyzer"""
        self.logger = get_logger("quality_analyzer")
        
        # Quality assessment models (simplified for production)
        self.coherence_patterns = self._load_coherence_patterns()
        self.safety_patterns = self._load_safety_patterns()
        self.engagement_patterns = self._load_engagement_patterns()
        
        # Performance tracking
        self.analysis_times = []
        self.quality_scores = []
        
        self.logger.info("Quality Analyzer initialized")
    
    @log_execution_time
    async def analyze_response(
        self,
        response: EnhancedResponse,
        context_used: Dict[str, Any],
        user_satisfaction: Optional[float] = None
    ) -> ResponseAnalysis:
        """Analyze response quality and generate comprehensive assessment"""
        correlation_id = generate_correlation_id()
        self.logger.info(f"Analyzing response quality for {response.response_id}", extra={
            "correlation_id": correlation_id,
            "response_length": len(response.enhanced_response)
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                response, context_used
            )
            
            # Analyze context usage
            context_usage = await self._analyze_context_usage(
                response, context_used
            )
            
            # Generate improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(
                quality_metrics, context_usage, response
            )
            
            # Calculate overall user satisfaction
            final_satisfaction = user_satisfaction or self._estimate_user_satisfaction(
                quality_metrics, context_usage
            )
            
            # Create response analysis
            analysis = ResponseAnalysis(
                analysis_id=generate_correlation_id(),
                response_id=response.response_id,
                conversation_id="",  # Will be provided by caller
                user_id="",  # Will be provided by caller
                quality_scores=quality_metrics.__dict__,
                context_usage=context_usage,
                user_satisfaction=final_satisfaction,
                improvement_suggestions=improvement_suggestions
            )
            
            # Update performance metrics
            self._update_performance_metrics(start_time, quality_metrics.overall_quality)
            
            self.logger.info(f"Response quality analysis completed", extra={
                "correlation_id": correlation_id,
                "overall_quality": quality_metrics.overall_quality,
                "suggestions_count": len(improvement_suggestions)
            })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze response quality: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "response_id": response.response_id
            })
            raise
    
    async def _calculate_quality_metrics(
        self,
        response: EnhancedResponse,
        context_used: Dict[str, Any]
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        # Coherence analysis
        coherence_score = await self._analyze_coherence(response.enhanced_response)
        
        # Relevance analysis
        relevance_score = await self._analyze_relevance(
            response.enhanced_response, context_used
        )
        
        # Context usage analysis
        context_usage_score = await self._analyze_context_utilization(
            response, context_used
        )
        
        # Personality consistency
        personality_score = await self._analyze_personality_consistency(
            response, context_used
        )
        
        # Safety analysis
        safety_score = await self._analyze_safety(response.enhanced_response)
        
        # Engagement potential
        engagement_score = await self._analyze_engagement_potential(
            response.enhanced_response
        )
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality([
            coherence_score, relevance_score, context_usage_score,
            personality_score, safety_score, engagement_score
        ])
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            response, context_used
        )
        
        return QualityMetrics(
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            context_usage_score=context_usage_score,
            personality_consistency_score=personality_score,
            safety_score=safety_score,
            engagement_potential=engagement_score,
            overall_quality=overall_quality,
            confidence_level=confidence_level,
            analysis_timestamp=datetime.utcnow()
        )
    
    async def _analyze_coherence(self, response_text: str) -> float:
        """Analyze response coherence and logical flow"""
        if not response_text:
            return 0.0
        
        score = 0.0
        
        # Sentence structure analysis
        sentences = re.split(r'[.!?]+', response_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        # Check sentence completeness
        complete_sentences = 0
        for sentence in sentences:
            if len(sentence.split()) >= 3:  # Minimum 3 words for meaningful sentence
                complete_sentences += 1
        
        sentence_completeness = complete_sentences / len(sentences)
        score += sentence_completeness * 0.3
        
        # Check logical flow indicators
        flow_indicators = ['because', 'therefore', 'however', 'although', 'furthermore', 'moreover']
        flow_count = sum(1 for indicator in flow_indicators if indicator in response_text.lower())
        flow_score = min(1.0, flow_count / 3)  # Normalize to 0-1
        score += flow_score * 0.2
        
        # Check paragraph structure
        paragraphs = response_text.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.2  # Bonus for structured paragraphs
        
        # Check response length appropriateness
        word_count = len(response_text.split())
        if 50 <= word_count <= 500:
            score += 0.3  # Optimal length
        elif 25 <= word_count <= 800:
            score += 0.2  # Acceptable length
        else:
            score += 0.1  # Suboptimal length
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_relevance(
        self,
        response_text: str,
        context_used: Dict[str, Any]
    ) -> float:
        """Analyze response relevance to context and user input"""
        if not response_text or not context_used:
            return 0.5  # Neutral score if no context
        
        score = 0.0
        
        # Extract context keywords
        context_text = ""
        if 'memories' in context_used:
            for memory in context_used['memories']:
                if isinstance(memory, dict) and 'content_summary' in memory:
                    context_text += " " + str(memory['content_summary'])
        
        if 'user_message' in context_used:
            context_text += " " + str(context_used['user_message'])
        
        if not context_text.strip():
            return 0.5
        
        # Calculate keyword overlap
        response_keywords = set(extract_keywords(response_text.lower(), 20))
        context_keywords = set(extract_keywords(context_text.lower(), 20))
        
        if not response_keywords or not context_keywords:
            return 0.5
        
        # Jaccard similarity
        intersection = len(response_keywords & context_keywords)
        union = len(response_keywords | context_keywords)
        
        if union > 0:
            keyword_similarity = intersection / union
            score += keyword_similarity * 0.6
        
        # Check for context-specific references
        context_references = 0
        if 'memories' in context_used:
            for memory in context_used['memories']:
                if isinstance(memory, dict) and 'content_summary' in memory:
                    memory_text = str(memory['content_summary'])
                    if any(keyword in response_text.lower() for keyword in extract_keywords(memory_text, 5)):
                        context_references += 1
        
        if context_references > 0:
            reference_score = min(1.0, context_references / 3)
            score += reference_score * 0.4
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_context_utilization(
        self,
        response: EnhancedResponse,
        context_used: Dict[str, Any]
    ) -> float:
        """Analyze how well the response utilizes provided context"""
        if not context_used or not context_used.get('memories'):
            return 0.5  # Neutral if no context
        
        score = 0.0
        
        # Check memory utilization
        memories = context_used.get('memories', [])
        if not memories:
            return 0.5
        
        utilized_memories = 0
        for memory in memories:
            if isinstance(memory, dict) and 'content_summary' in memory:
                memory_text = str(memory['content_summary'])
                memory_keywords = extract_keywords(memory_text, 10)
                
                # Check if memory keywords appear in response
                if any(keyword in response.enhanced_response.lower() for keyword in memory_keywords):
                    utilized_memories += 1
        
        utilization_rate = utilized_memories / len(memories)
        score += utilization_rate * 0.7
        
        # Check for natural context integration
        natural_integration = 0
        if 'user_message' in context_used:
            user_message = str(context_used['user_message'])
            if any(word in response.enhanced_response.lower() for word in user_message.lower().split()):
                natural_integration += 0.3
        
        score += natural_integration
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_personality_consistency(
        self,
        response: EnhancedResponse,
        context_used: Dict[str, Any]
    ) -> float:
        """Analyze personality consistency across responses"""
        # This would typically compare with previous responses
        # For now, use a simplified approach
        
        score = 0.8  # Base score
        
        # Check for personality indicators
        personality_indicators = {
            'formal': ['therefore', 'furthermore', 'consequently', 'thus'],
            'casual': ['hey', 'cool', 'awesome', 'great'],
            'supportive': ['support', 'help', 'assist', 'encourage'],
            'professional': ['recommend', 'suggest', 'advise', 'consider']
        }
        
        response_lower = response.enhanced_response.lower()
        
        # Count personality indicators
        indicator_counts = {}
        for personality, indicators in personality_indicators.items():
            count = sum(1 for indicator in indicators if indicator in response_lower)
            indicator_counts[personality] = count
        
        # If we have context about preferred style, check alignment
        if 'user_preferences' in context_used:
            user_prefs = context_used['user_preferences']
            if 'communication_style' in user_prefs:
                preferred_style = user_prefs['communication_style'].lower()
                if preferred_style in indicator_counts:
                    # Boost score for style alignment
                    if indicator_counts[preferred_style] > 0:
                        score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_safety(self, response_text: str) -> float:
        """Analyze response safety and appropriateness"""
        if not response_text:
            return 1.0  # Empty response is safe
        
        score = 1.0  # Start with perfect safety score
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(harm|hurt|kill|suicide|self.?harm)\b',
            r'\b(dangerous|illegal|criminal)\b',
            r'\b(hate|racist|sexist|discriminatory)\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, response_text.lower()):
                score -= 0.3  # Penalty for harmful content
        
        # Check for inappropriate language
        inappropriate_words = ['profanity', 'vulgar', 'offensive']
        for word in inappropriate_words:
            if word in response_text.lower():
                score -= 0.2
        
        # Check for medical advice (should be avoided)
        medical_patterns = [
            r'\b(diagnose|diagnosis|treatment|medication|prescription)\b',
            r'\b(doctor|physician|therapist|psychiatrist)\b'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, response_text.lower()):
                score -= 0.1  # Small penalty for medical content
        
        return max(0.0, score)
    
    async def _analyze_engagement_potential(self, response_text: str) -> float:
        """Analyze potential for user engagement and continued conversation"""
        if not response_text:
            return 0.0
        
        score = 0.0
        
        # Check for questions (encourage responses)
        question_count = response_text.count('?')
        if question_count > 0:
            score += min(0.4, question_count * 0.2)
        
        # Check for follow-up suggestions
        follow_up_indicators = [
            'what do you think', 'how do you feel', 'would you like',
            'tell me more', 'what about you', 'your thoughts'
        ]
        
        for indicator in follow_up_indicators:
            if indicator in response_text.lower():
                score += 0.2
                break
        
        # Check for personalization
        personal_indicators = ['you', 'your', 'yourself']
        personal_count = sum(response_text.lower().count(indicator) for indicator in personal_indicators)
        if personal_count > 0:
            score += min(0.2, personal_count * 0.05)
        
        # Check for emotional validation
        emotional_indicators = ['understand', 'feel', 'experience', 'valid']
        emotional_count = sum(1 for indicator in emotional_indicators if indicator in response_text.lower())
        if emotional_count > 0:
            score += min(0.2, emotional_count * 0.1)
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_context_usage(
        self,
        response: EnhancedResponse,
        context_used: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how context was used in the response"""
        analysis = {
            "context_provided": bool(context_used),
            "memories_utilized": 0,
            "context_keywords_used": 0,
            "context_integration_score": 0.0,
            "missing_context": [],
            "context_efficiency": 0.0
        }
        
        if not context_used:
            return analysis
        
        # Analyze memory utilization
        memories = context_used.get('memories', [])
        if memories:
            utilized_memories = 0
            for memory in memories:
                if isinstance(memory, dict) and 'content_summary' in memory:
                    memory_text = str(memory['content_summary'])
                    memory_keywords = extract_keywords(memory_text, 10)
                    
                    if any(keyword in response.enhanced_response.lower() for keyword in memory_keywords):
                        utilized_memories += 1
            
            analysis["memories_utilized"] = utilized_memories
            analysis["context_integration_score"] = utilized_memories / len(memories) if memories else 0.0
        
        # Analyze keyword usage
        if 'user_message' in context_used:
            user_keywords = extract_keywords(str(context_used['user_message']), 10)
            used_keywords = sum(1 for keyword in user_keywords if keyword in response.enhanced_response.lower())
            analysis["context_keywords_used"] = used_keywords
        
        # Calculate context efficiency
        total_context_items = len(memories) + (1 if 'user_message' in context_used else 0)
        if total_context_items > 0:
            analysis["context_efficiency"] = (analysis["memories_utilized"] + analysis["context_keywords_used"]) / total_context_items
        
        return analysis
    
    async def _generate_improvement_suggestions(
        self,
        quality_metrics: QualityMetrics,
        context_usage: Dict[str, Any],
        response: EnhancedResponse
    ) -> List[str]:
        """Generate improvement suggestions based on quality analysis"""
        suggestions = []
        
        # Coherence suggestions
        if quality_metrics.coherence_score < 0.7:
            suggestions.append("Improve sentence structure and logical flow for better coherence")
        
        # Relevance suggestions
        if quality_metrics.relevance_score < 0.7:
            suggestions.append("Ensure response directly addresses user's message and context")
        
        # Context usage suggestions
        if quality_metrics.context_usage_score < 0.6:
            suggestions.append("Better utilize provided memories and context in the response")
        
        # Personality consistency suggestions
        if quality_metrics.personality_consistency_score < 0.8:
            suggestions.append("Maintain consistent communication style throughout the response")
        
        # Safety suggestions
        if quality_metrics.safety_score < 0.9:
            suggestions.append("Review content for safety and appropriateness")
        
        # Engagement suggestions
        if quality_metrics.engagement_potential < 0.6:
            suggestions.append("Include follow-up questions to encourage continued conversation")
        
        # Context efficiency suggestions
        if context_usage.get("context_efficiency", 0) < 0.5:
            suggestions.append("Optimize context usage to improve response relevance")
        
        # Response length suggestions
        response_length = len(response.enhanced_response)
        if response_length < 50:
            suggestions.append("Provide more detailed responses for better user experience")
        elif response_length > 800:
            suggestions.append("Consider more concise responses to maintain user engagement")
        
        # If no specific issues, provide general improvement suggestions
        if not suggestions:
            suggestions.append("Response quality is good, focus on maintaining consistency")
            suggestions.append("Continue monitoring user engagement patterns")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _estimate_user_satisfaction(
        self,
        quality_metrics: QualityMetrics,
        context_usage: Dict[str, Any]
    ) -> float:
        """Estimate user satisfaction based on quality metrics"""
        # Weighted combination of quality factors
        weights = {
            'coherence': 0.2,
            'relevance': 0.25,
            'context_usage': 0.2,
            'personality': 0.15,
            'safety': 0.1,
            'engagement': 0.1
        }
        
        satisfaction = (
            quality_metrics.coherence_score * weights['coherence'] +
            quality_metrics.relevance_score * weights['relevance'] +
            quality_metrics.context_usage_score * weights['context_usage'] +
            quality_metrics.personality_consistency_score * weights['personality'] +
            quality_metrics.safety_score * weights['safety'] +
            quality_metrics.engagement_potential * weights['engagement']
        )
        
        # Adjust based on context usage efficiency
        context_efficiency = context_usage.get("context_efficiency", 0.5)
        satisfaction = satisfaction * 0.8 + context_efficiency * 0.2
        
        return min(1.0, max(0.0, satisfaction))
    
    def _calculate_overall_quality(self, scores: List[float]) -> float:
        """Calculate overall quality score from individual metrics"""
        if not scores:
            return 0.0
        
        # Weighted average with emphasis on core metrics
        weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # Sum to 1.0
        
        # Ensure we have enough weights
        while len(weights) < len(scores):
            weights.append(weights[-1] * 0.8)
        
        # Normalize weights
        total_weight = sum(weights[:len(scores)])
        normalized_weights = [w / total_weight for w in weights[:len(scores)]]
        
        overall_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        return min(1.0, max(0.0, overall_score))
    
    def _calculate_confidence_level(
        self,
        response: EnhancedResponse,
        context_used: Dict[str, Any]
    ) -> float:
        """Calculate confidence level in the quality assessment"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on response characteristics
        if len(response.enhanced_response) > 100:
            confidence += 0.1  # Longer responses provide more data
        
        if context_used and context_used.get('memories'):
            confidence += 0.1  # Context availability improves confidence
        
        # Adjust based on response quality indicators
        if response.quality_score > 0.8:
            confidence += 0.1  # High quality responses are easier to assess
        
        return min(1.0, max(0.0, confidence))
    
    def _load_coherence_patterns(self) -> Dict[str, Any]:
        """Load coherence analysis patterns"""
        return {
            "sentence_indicators": ["because", "therefore", "however", "although"],
            "flow_connectors": ["first", "second", "finally", "in conclusion"],
            "logical_operators": ["and", "or", "but", "if", "then"]
        }
    
    def _load_safety_patterns(self) -> Dict[str, Any]:
        """Load safety analysis patterns"""
        return {
            "harmful_content": ["harm", "hurt", "kill", "suicide"],
            "inappropriate": ["profanity", "vulgar", "offensive"],
            "medical_advice": ["diagnose", "treatment", "prescription"]
        }
    
    def _load_engagement_patterns(self) -> Dict[str, Any]:
        """Load engagement analysis patterns"""
        return {
            "questions": ["what", "how", "why", "when", "where"],
            "personal_pronouns": ["you", "your", "yourself"],
            "emotional_validation": ["understand", "feel", "experience"]
        }
    
    def _update_performance_metrics(self, start_time: datetime, quality_score: float) -> None:
        """Update performance tracking metrics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.analysis_times.append(processing_time)
        self.quality_scores.append(quality_score)
        
        # Keep only last 100 measurements
        if len(self.analysis_times) > 100:
            self.analysis_times = self.analysis_times[-100:]
        if len(self.quality_scores) > 100:
            self.quality_scores = self.quality_scores[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.analysis_times:
            return {
                "avg_analysis_time_ms": 0.0,
                "avg_quality_score": 0.0,
                "total_analyses": 0
            }
        
        return {
            "avg_analysis_time_ms": sum(self.analysis_times) / len(self.analysis_times),
            "min_analysis_time_ms": min(self.analysis_times),
            "max_analysis_time_ms": max(self.analysis_times),
            "avg_quality_score": sum(self.quality_scores) / len(self.quality_scores),
            "total_analyses": len(self.analysis_times)
        }
