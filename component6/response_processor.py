"""
Response Processor Module for Component 6.
Enhances and validates AI responses before delivery.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict

from shared.schemas import (
    EnhancedResponse, ResponseAnalysis, QualityMetrics,
    ConversationContext, UserProfile
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id
)
from config.settings import settings


class ResponseProcessor:
    """Processes and enhances AI responses before delivery"""
    
    def __init__(self):
        """Initialize response processor"""
        self.logger = get_logger("response_processor")
        
        # Response enhancement rules
        self.enhancement_rules = self._initialize_enhancement_rules()
        self.validation_rules = self._initialize_validation_rules()
        
        # Processing history
        self.processing_history = {}
        self.enhancement_stats = defaultdict(int)
        
        # Performance tracking
        self.processing_times = []
        self.enhancement_times = []
        
        self.logger.info("Response Processor initialized")
    
    def _initialize_enhancement_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize response enhancement rules"""
        return {
            "clarity": {
                "enabled": True,
                "priority": "high",
                "rules": [
                    "break_long_sentences",
                    "add_transition_phrases",
                    "clarify_pronouns",
                    "add_context_markers"
                ]
            },
            "engagement": {
                "enabled": True,
                "priority": "medium",
                "rules": [
                    "add_thoughtful_questions",
                    "include_encouragement",
                    "personalize_responses",
                    "add_conversation_hooks"
                ]
            },
            "safety": {
                "enabled": True,
                "priority": "critical",
                "rules": [
                    "filter_inappropriate_content",
                    "add_safety_disclaimers",
                    "moderate_tone",
                    "validate_facts"
                ]
            },
            "accessibility": {
                "enabled": True,
                "priority": "medium",
                "rules": [
                    "simplify_complex_terms",
                    "add_explanations",
                    "use_clear_language",
                    "structure_information"
                ]
            }
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize response validation rules"""
        return {
            "content_validation": {
                "enabled": True,
                "checks": [
                    "check_response_length",
                    "validate_sentence_structure",
                    "check_for_incomplete_thoughts",
                    "validate_context_relevance"
                ]
            },
            "safety_validation": {
                "enabled": True,
                "checks": [
                    "check_for_harmful_content",
                    "validate_tone_appropriateness",
                    "check_for_biases",
                    "validate_fact_claims"
                ]
            },
            "quality_validation": {
                "enabled": True,
                "checks": [
                    "check_coherence",
                    "validate_grammar",
                    "check_readability",
                    "validate_completeness"
                ]
            }
        }
    
    @log_execution_time
    async def process_response(
        self,
        raw_response: str,
        conversation_context: ConversationContext,
        user_profile: Optional[UserProfile] = None,
        personality_profile: Optional[Dict[str, Any]] = None
    ) -> EnhancedResponse:
        """Process and enhance a raw AI response"""
        correlation_id = generate_correlation_id()
        start_time = datetime.utcnow()
        
        self.logger.info("Processing response for enhancement", extra={
            "correlation_id": correlation_id,
            "response_length": len(raw_response)
        })
        
        try:
            # Initialize enhanced response
            enhanced_response = EnhancedResponse(
                response_id=generate_correlation_id(),
                original_response=raw_response,
                enhanced_response=raw_response,
                enhancement_metadata={},
                processing_timestamp=datetime.utcnow(),
                quality_metrics={},
                safety_checks={},
                context_usage={}
            )
            
            # Apply enhancement rules
            enhanced_text = await self._apply_enhancement_rules(
                raw_response, conversation_context, user_profile, personality_profile
            )
            
            # Apply validation rules
            validation_results = await self._apply_validation_rules(
                enhanced_text, conversation_context
            )
            
            # Update enhanced response
            enhanced_response.enhanced_response = enhanced_text
            enhanced_response.enhancement_metadata = {
                "enhancement_rules_applied": list(self.enhancement_rules.keys()),
                "validation_results": validation_results,
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                enhanced_text, raw_response, conversation_context
            )
            enhanced_response.quality_metrics = quality_metrics
            
            # Perform safety checks
            safety_checks = await self._perform_safety_checks(enhanced_text)
            enhanced_response.safety_checks = safety_checks
            
            # Analyze context usage
            context_usage = await self._analyze_context_usage(
                enhanced_text, conversation_context
            )
            enhanced_response.context_usage = context_usage
            
            # Store processing history
            await self._store_processing_history(
                enhanced_response.response_id, enhanced_response, start_time
            )
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.processing_times.append(processing_time)
            
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            
            self.logger.info(f"Response processed successfully in {processing_time:.2f}ms", extra={
                "correlation_id": correlation_id,
                "response_id": enhanced_response.response_id,
                "enhancement_applied": bool(enhanced_text != raw_response)
            })
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Failed to process response: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            # Return minimal enhanced response on error
            return EnhancedResponse(
                response_id=correlation_id,
                original_response=raw_response,
                enhanced_response=raw_response,
                enhancement_metadata={"error": str(e)},
                processing_timestamp=datetime.utcnow(),
                quality_metrics={},
                safety_checks={},
                context_usage={}
            )
    
    async def _apply_enhancement_rules(
        self,
        raw_response: str,
        conversation_context: ConversationContext,
        user_profile: Optional[UserProfile],
        personality_profile: Optional[Dict[str, Any]]
    ) -> str:
        """Apply enhancement rules to improve response quality"""
        enhanced_text = raw_response
        start_time = datetime.utcnow()
        
        # Apply clarity enhancements
        if self.enhancement_rules["clarity"]["enabled"]:
            enhanced_text = await self._enhance_clarity(enhanced_text, conversation_context)
        
        # Apply engagement enhancements
        if self.enhancement_rules["engagement"]["enabled"]:
            enhanced_text = await self._enhance_engagement(
                enhanced_text, conversation_context, user_profile, personality_profile
            )
        
        # Apply safety enhancements
        if self.enhancement_rules["safety"]["enabled"]:
            enhanced_text = await self._enhance_safety(enhanced_text, conversation_context)
        
        # Apply accessibility enhancements
        if self.enhancement_rules["accessibility"]["enabled"]:
            enhanced_text = await self._enhance_accessibility(enhanced_text, user_profile)
        
        # Calculate enhancement time
        enhancement_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.enhancement_times.append(enhancement_time)
        
        if len(self.enhancement_times) > 100:
            self.enhancement_times = self.enhancement_times[-100:]
        
        return enhanced_text
    
    async def _enhance_clarity(
        self,
        text: str,
        conversation_context: ConversationContext
    ) -> str:
        """Enhance response clarity"""
        enhanced_text = text
        
        # Break long sentences
        enhanced_text = self._break_long_sentences(enhanced_text)
        
        # Add transition phrases
        enhanced_text = self._add_transition_phrases(enhanced_text)
        
        # Clarify pronouns
        enhanced_text = self._clarify_pronouns(enhanced_text, conversation_context)
        
        # Add context markers
        enhanced_text = self._add_context_markers(enhanced_text, conversation_context)
        
        self.enhancement_stats["clarity"] += 1
        return enhanced_text
    
    async def _enhance_engagement(
        self,
        text: str,
        conversation_context: ConversationContext,
        user_profile: Optional[UserProfile],
        personality_profile: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance response engagement"""
        enhanced_text = text
        
        # Add thoughtful questions
        enhanced_text = self._add_thoughtful_questions(enhanced_text, conversation_context)
        
        # Include encouragement
        enhanced_text = self._add_encouragement(enhanced_text, personality_profile)
        
        # Personalize responses
        enhanced_text = self._personalize_responses(enhanced_text, user_profile)
        
        # Add conversation hooks
        enhanced_text = self._add_conversation_hooks(enhanced_text, conversation_context)
        
        self.enhancement_stats["engagement"] += 1
        return enhanced_text
    
    async def _enhance_safety(
        self,
        text: str,
        conversation_context: ConversationContext
    ) -> str:
        """Enhance response safety"""
        enhanced_text = text
        
        # Filter inappropriate content
        enhanced_text = self._filter_inappropriate_content(enhanced_text)
        
        # Add safety disclaimers
        enhanced_text = self._add_safety_disclaimers(enhanced_text, conversation_context)
        
        # Moderate tone
        enhanced_text = self._moderate_tone(enhanced_text)
        
        # Validate facts
        enhanced_text = self._validate_facts(enhanced_text)
        
        self.enhancement_stats["safety"] += 1
        return enhanced_text
    
    async def _enhance_accessibility(
        self,
        text: str,
        user_profile: Optional[UserProfile]
    ) -> str:
        """Enhance response accessibility"""
        enhanced_text = text
        
        # Simplify complex terms
        enhanced_text = self._simplify_complex_terms(enhanced_text)
        
        # Add explanations
        enhanced_text = self._add_explanations(enhanced_text)
        
        # Use clear language
        enhanced_text = self._use_clear_language(enhanced_text)
        
        # Structure information
        enhanced_text = self._structure_information(enhanced_text)
        
        self.enhancement_stats["accessibility"] += 1
        return enhanced_text
    
    def _break_long_sentences(self, text: str) -> str:
        """Break long sentences into shorter, clearer ones"""
        sentences = re.split(r'[.!?]+', text)
        enhanced_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 100:  # Long sentence threshold
                # Simple splitting on conjunctions
                parts = re.split(r'\s+(and|or|but|however|therefore|furthermore)\s+', sentence)
                if len(parts) > 1:
                    enhanced_sentences.extend([part.strip() + "." for part in parts if part.strip()])
                else:
                    enhanced_sentences.append(sentence + ".")
            else:
                enhanced_sentences.append(sentence + ".")
        
        return " ".join(enhanced_sentences)
    
    def _add_transition_phrases(self, text: str) -> str:
        """Add transition phrases to improve flow"""
        transitions = [
            "Additionally,", "Furthermore,", "Moreover,", "In addition,",
            "On the other hand,", "However,", "Nevertheless,", "Despite this,",
            "As a result,", "Therefore,", "Consequently,", "Thus,"
        ]
        
        # Simple approach: add transitions to some sentences
        sentences = text.split(". ")
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i > 0 and len(sentence) > 20:  # Add transitions to longer sentences
                transition = transitions[i % len(transitions)]
                enhanced_sentences.append(transition + " " + sentence)
            else:
                enhanced_sentences.append(sentence)
        
        return ". ".join(enhanced_sentences)
    
    def _clarify_pronouns(self, text: str, conversation_context: ConversationContext) -> str:
        """Clarify ambiguous pronouns"""
        # Simple pronoun replacement based on context
        pronoun_mappings = {
            "it": "this topic",
            "this": "this point",
            "that": "that aspect",
            "they": "the team",
            "them": "the team"
        }
        
        enhanced_text = text
        for pronoun, replacement in pronoun_mappings.items():
            # Replace standalone pronouns (with word boundaries)
            pattern = r'\b' + pronoun + r'\b'
            enhanced_text = re.sub(pattern, replacement, enhanced_text, flags=re.IGNORECASE)
        
        return enhanced_text
    
    def _add_context_markers(self, text: str, conversation_context: ConversationContext) -> str:
        """Add context markers to improve understanding"""
        if hasattr(conversation_context, 'conversation_topic'):
            topic = conversation_context.conversation_topic
            if topic and topic.lower() not in text.lower():
                # Add topic context if not already present
                text = f"Regarding {topic}, {text}"
        
        return text
    
    def _add_thoughtful_questions(self, text: str, conversation_context: ConversationContext) -> str:
        """Add thoughtful questions to encourage engagement"""
        questions = [
            "What are your thoughts on this?",
            "How does this relate to your experience?",
            "What questions do you have about this?",
            "How might you apply this information?",
            "What aspects would you like to explore further?"
        ]
        
        # Add a question if the response is substantial
        if len(text) > 100 and not text.endswith("?"):
            question = questions[len(text) % len(questions)]
            text += f"\n\n{question}"
        
        return text
    
    def _add_encouragement(self, text: str, personality_profile: Optional[Dict[str, Any]]) -> str:
        """Add encouragement based on personality profile"""
        encouragements = [
            "You're doing great!",
            "Keep up the excellent work!",
            "This is really insightful!",
            "You're making great progress!",
            "Well done on this!"
        ]
        
        # Add encouragement if personality is supportive
        if personality_profile and personality_profile.get("traits", {}).get("supportiveness", 0) > 0.7:
            if not any(enc in text for enc in encouragements):
                encouragement = encouragements[len(text) % len(encouragements)]
                text += f"\n\n{encouragement}"
        
        return text
    
    def _personalize_responses(self, text: str, user_profile: Optional[UserProfile]) -> str:
        """Personalize responses based on user profile"""
        if not user_profile:
            return text
        
        # Add personal touch if user has a name
        if hasattr(user_profile, 'display_name') and user_profile.display_name:
            name = user_profile.display_name
            # Add name to the beginning if not already present
            if name not in text:
                text = f"Hi {name}! {text}"
        
        return text
    
    def _add_conversation_hooks(self, text: str, conversation_context: ConversationContext) -> str:
        """Add conversation hooks to encourage continued dialogue"""
        hooks = [
            "What do you think about this approach?",
            "Would you like to explore this further?",
            "How does this align with your goals?",
            "What's your perspective on this?",
            "Shall we dive deeper into any particular aspect?"
        ]
        
        # Add hook if response is substantial and doesn't already have one
        if len(text) > 150 and not any(hook.lower() in text.lower() for hook in hooks):
            hook = hooks[len(text) % len(hooks)]
            text += f"\n\n{hook}"
        
        return text
    
    def _filter_inappropriate_content(self, text: str) -> str:
        """Filter out inappropriate content"""
        # Simple content filtering (in production, use more sophisticated methods)
        inappropriate_patterns = [
            r'\b(hate|violence|harm)\b',
            r'\b(discriminatory|offensive|inappropriate)\b'
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, "[content moderated]", text, flags=re.IGNORECASE)
        
        return text
    
    def _add_safety_disclaimers(self, text: str, conversation_context: ConversationContext) -> str:
        """Add safety disclaimers when appropriate"""
        # Add disclaimer for certain topics
        sensitive_topics = ["medical", "legal", "financial", "mental health"]
        
        if hasattr(conversation_context, 'conversation_topic'):
            topic = conversation_context.conversation_topic.lower()
            if any(sensitive in topic for sensitive in sensitive_topics):
                disclaimer = "\n\nNote: This information is for general purposes only. Please consult appropriate professionals for specific advice."
                if disclaimer not in text:
                    text += disclaimer
        
        return text
    
    def _moderate_tone(self, text: str) -> str:
        """Moderate response tone to ensure appropriateness"""
        # Simple tone moderation
        aggressive_patterns = [
            (r'\b(always|never)\b', 'often/rarely'),
            (r'\b(terrible|awful|horrible)\b', 'challenging/difficult'),
            (r'\b(stupid|idiot|fool)\b', 'misguided/incorrect')
        ]
        
        for pattern, replacement in aggressive_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _validate_facts(self, text: str) -> str:
        """Add fact validation disclaimers"""
        # Simple fact validation (in production, use fact-checking APIs)
        fact_indicators = ["research shows", "studies indicate", "scientists say", "experts agree"]
        
        if any(indicator in text.lower() for indicator in fact_indicators):
            if "fact-checking" not in text.lower():
                text += "\n\nNote: Please verify facts independently as information may change over time."
        
        return text
    
    def _simplify_complex_terms(self, text: str) -> str:
        """Simplify complex terminology"""
        # Simple term simplification
        complex_terms = {
            "utilize": "use",
            "facilitate": "help",
            "implement": "put into practice",
            "methodology": "method",
            "paradigm": "approach"
        }
        
        for complex_term, simple_term in complex_terms.items():
            text = re.sub(r'\b' + complex_term + r'\b', simple_term, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_explanations(self, text: str) -> str:
        """Add explanations for complex concepts"""
        # Simple explanation addition
        if len(text) > 200 and "in other words" not in text.lower():
            # Add explanation for longer responses
            text += "\n\nIn other words, the key point is to focus on what matters most to you."
        
        return text
    
    def _use_clear_language(self, text: str) -> str:
        """Ensure language is clear and understandable"""
        # Simple clarity improvements
        unclear_patterns = [
            (r'\b(etc\.|\.\.\.)\b', 'and so on'),
            (r'\b(i\.e\.)\b', 'that is'),
            (r'\b(e\.g\.)\b', 'for example')
        ]
        
        for pattern, replacement in unclear_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _structure_information(self, text: str) -> str:
        """Structure information for better readability"""
        # Simple structuring
        if len(text) > 300 and not text.startswith("Here's"):
            text = "Here's what I can tell you:\n\n" + text
        
        return text
    
    async def _apply_validation_rules(
        self,
        text: str,
        conversation_context: ConversationContext
    ) -> Dict[str, Any]:
        """Apply validation rules to check response quality"""
        validation_results = {}
        
        # Content validation
        if self.validation_rules["content_validation"]["enabled"]:
            validation_results["content"] = await self._validate_content(text, conversation_context)
        
        # Safety validation
        if self.validation_rules["safety_validation"]["enabled"]:
            validation_results["safety"] = await self._validate_safety(text)
        
        # Quality validation
        if self.validation_rules["quality_validation"]["enabled"]:
            validation_results["quality"] = await self._validate_quality(text)
        
        return validation_results
    
    async def _validate_content(
        self,
        text: str,
        conversation_context: ConversationContext
    ) -> Dict[str, Any]:
        """Validate response content"""
        return {
            "length_appropriate": 50 <= len(text) <= 2000,
            "sentence_structure_valid": self._check_sentence_structure(text),
            "complete_thoughts": self._check_complete_thoughts(text),
            "context_relevant": self._check_context_relevance(text, conversation_context)
        }
    
    async def _validate_safety(self, text: str) -> Dict[str, Any]:
        """Validate response safety"""
        return {
            "no_harmful_content": not self._contains_harmful_content(text),
            "tone_appropriate": self._check_tone_appropriateness(text),
            "no_biases": not self._contains_biases(text),
            "fact_claims_valid": self._validate_fact_claims(text)
        }
    
    async def _validate_quality(self, text: str) -> Dict[str, Any]:
        """Validate response quality"""
        return {
            "coherent": self._check_coherence(text),
            "grammar_valid": self._check_grammar(text),
            "readable": self._check_readability(text),
            "complete": self._check_completeness(text)
        }
    
    def _check_sentence_structure(self, text: str) -> bool:
        """Check if sentence structure is valid"""
        sentences = text.split('.')
        return all(len(s.strip()) > 0 for s in sentences if s.strip())
    
    def _check_complete_thoughts(self, text: str) -> bool:
        """Check if response contains complete thoughts"""
        return len(text.split('.')) >= 2
    
    def _check_context_relevance(self, text: str, conversation_context: ConversationContext) -> bool:
        """Check if response is relevant to conversation context"""
        if hasattr(conversation_context, 'conversation_topic'):
            topic = conversation_context.conversation_topic.lower()
            return topic in text.lower() or len(text) < 100
        return True
    
    def _contains_harmful_content(self, text: str) -> bool:
        """Check if text contains harmful content"""
        harmful_patterns = [r'\b(harm|hurt|danger)\b']
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in harmful_patterns)
    
    def _check_tone_appropriateness(self, text: str) -> bool:
        """Check if tone is appropriate"""
        inappropriate_patterns = [r'\b(angry|hostile|aggressive)\b']
        return not any(re.search(pattern, text, re.IGNORECASE) for pattern in inappropriate_patterns)
    
    def _contains_biases(self, text: str) -> bool:
        """Check if text contains biases"""
        bias_patterns = [r'\b(always|never|everyone|nobody)\b']
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in bias_patterns)
    
    def _validate_fact_claims(self, text: str) -> bool:
        """Validate fact claims in text"""
        fact_indicators = ["research shows", "studies prove", "scientists confirm"]
        return not any(indicator in text.lower() for indicator in fact_indicators)
    
    def _check_coherence(self, text: str) -> bool:
        """Check if text is coherent"""
        sentences = text.split('.')
        return len(sentences) <= 1 or all(len(s.strip()) > 10 for s in sentences if s.strip())
    
    def _check_grammar(self, text: str) -> bool:
        """Check basic grammar"""
        # Simple grammar check
        return text.count('(') == text.count(')') and text.count('"') % 2 == 0
    
    def _check_readability(self, text: str) -> bool:
        """Check readability"""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        return avg_word_length <= 8  # Average word length threshold
    
    def _check_completeness(self, text: str) -> bool:
        """Check if response is complete"""
        return text.strip().endswith(('.', '!', '?'))
    
    async def _calculate_quality_metrics(
        self,
        enhanced_text: str,
        original_text: str,
        conversation_context: ConversationContext
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the response"""
        return {
            "response_length": len(enhanced_text),
            "enhancement_ratio": len(enhanced_text) / len(original_text) if original_text else 1.0,
            "sentence_count": len(enhanced_text.split('.')),
            "word_count": len(enhanced_text.split()),
            "readability_score": self._calculate_readability_score(enhanced_text),
            "coherence_score": self._calculate_coherence_score(enhanced_text),
            "context_relevance_score": self._calculate_context_relevance_score(enhanced_text, conversation_context)
        }
    
    async def _perform_safety_checks(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive safety checks"""
        return {
            "content_safety": not self._contains_harmful_content(text),
            "tone_safety": self._check_tone_appropriateness(text),
            "bias_detection": not self._contains_biases(text),
            "fact_safety": self._validate_fact_claims(text),
            "overall_safety_score": 0.9  # Placeholder for more sophisticated scoring
        }
    
    async def _analyze_context_usage(
        self,
        text: str,
        conversation_context: ConversationContext
    ) -> Dict[str, Any]:
        """Analyze how well the response uses conversation context"""
        context_usage = {
            "topic_mentioned": False,
            "context_elements_used": 0,
            "relevance_score": 0.0
        }
        
        if hasattr(conversation_context, 'conversation_topic'):
            topic = conversation_context.conversation_topic
            if topic and topic.lower() in text.lower():
                context_usage["topic_mentioned"] = True
                context_usage["context_elements_used"] += 1
        
        # Calculate relevance score
        context_usage["relevance_score"] = min(1.0, context_usage["context_elements_used"] * 0.5)
        
        return context_usage
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (0-1)"""
        words = text.split()
        if not words:
            return 0.0
        
        # Simple Flesch-like scoring
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len(text.split('.'))
        
        if sentence_count == 0:
            return 0.0
        
        # Normalize to 0-1 scale
        word_score = max(0, 1 - (avg_word_length - 4) / 6)  # 4-10 character range
        sentence_score = max(0, 1 - (sentence_count - 1) / 10)  # 1-11 sentence range
        
        return (word_score + sentence_score) / 2
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score (0-1)"""
        sentences = text.split('.')
        if len(sentences) <= 1:
            return 1.0
        
        # Simple coherence based on sentence length consistency
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.0
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        
        # Normalize variance to 0-1 scale
        coherence = max(0, 1 - (variance / 100))
        return coherence
    
    def _calculate_context_relevance_score(
        self,
        text: str,
        conversation_context: ConversationContext
    ) -> float:
        """Calculate context relevance score (0-1)"""
        if not hasattr(conversation_context, 'conversation_topic'):
            return 0.5
        
        topic = conversation_context.conversation_topic.lower()
        if not topic:
            return 0.5
        
        # Simple relevance based on topic mention
        topic_words = topic.split()
        text_lower = text.lower()
        
        relevance_score = 0.0
        for word in topic_words:
            if len(word) > 3 and word in text_lower:  # Only consider meaningful words
                relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    async def _store_processing_history(
        self,
        response_id: str,
        enhanced_response: EnhancedResponse,
        start_time: datetime
    ) -> None:
        """Store processing history for analysis"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        self.processing_history[response_id] = {
            "timestamp": datetime.utcnow(),
            "processing_time_ms": processing_time,
            "enhancement_applied": enhanced_response.enhanced_response != enhanced_response.original_response,
            "quality_metrics": enhanced_response.quality_metrics,
            "safety_checks": enhanced_response.safety_checks
        }
        
        # Keep only last 1000 entries
        if len(self.processing_history) > 1000:
            oldest_key = min(self.processing_history.keys())
            del self.processing_history[oldest_key]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.processing_times:
            return {
                "avg_processing_time_ms": 0.0,
                "avg_enhancement_time_ms": 0.0,
                "total_responses_processed": 0
            }
        
        return {
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time_ms": min(self.processing_times),
            "max_processing_time_ms": max(self.processing_times),
            "avg_enhancement_time_ms": sum(self.enhancement_times) / len(self.enhancement_times) if self.enhancement_times else 0.0,
            "total_responses_processed": len(self.processing_times),
            "enhancement_stats": dict(self.enhancement_stats)
        }
    
    async def cleanup_old_data(self, max_age_hours: int = 24) -> int:
        """Clean up old processing history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        total_removed = 0
        
        old_keys = [
            key for key, data in self.processing_history.items()
            if data["timestamp"] < cutoff_time
        ]
        
        for key in old_keys:
            del self.processing_history[key]
            total_removed += 1
        
        if total_removed > 0:
            self.logger.info(f"Cleaned up {total_removed} old processing history entries")
        
        return total_removed
