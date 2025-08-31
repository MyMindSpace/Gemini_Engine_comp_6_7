"""
Personality Engine Module for Component 6.
Adapts communication styles based on user preferences and context.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import random
from collections import defaultdict

from shared.schemas import (
    UserProfile, ConversationContext, EnhancedResponse,
    PersonalityProfile, CommunicationStyle, CommunicationStyleDetails
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id
)
from config.settings import settings


class PersonalityEngine:
    """Adapts AI personality and communication style based on user preferences"""
    
    def __init__(self):
        """Initialize personality engine"""
        self.logger = get_logger("personality_engine")
        
        # Personality profiles and styles
        self.personality_profiles = self._initialize_personality_profiles()
        self.communication_styles = self._initialize_communication_styles()
        
        # User personality tracking
        self.user_personality_history = {}
        self.conversation_personality_cache = {}
        
        # Adaptation learning
        self.style_effectiveness = defaultdict(lambda: defaultdict(list))
        self.adaptation_patterns = {}
        
        # Performance tracking
        self.adaptation_times = []
        self.style_selection_times = []
        
        self.logger.info("Personality Engine initialized")
    
    def _initialize_personality_profiles(self) -> Dict[str, PersonalityProfile]:
        """Initialize available personality profiles"""
        profiles = {
            "friendly": PersonalityProfile(
                profile_id="friendly",
                name="Friendly Companion",
                description="Warm, supportive, and encouraging communication style",
                traits={
                    "empathy": 0.9,
                    "enthusiasm": 0.8,
                    "supportiveness": 0.9,
                    "humor": 0.7,
                    "formality": 0.3
                },
                communication_patterns={
                    "greeting_style": "warm",
                    "response_tone": "encouraging",
                    "question_style": "curious",
                    "closing_style": "supportive"
                }
            ),
            "professional": PersonalityProfile(
                profile_id="professional",
                name="Professional Assistant",
                description="Clear, concise, and business-oriented communication",
                traits={
                    "clarity": 0.9,
                    "efficiency": 0.9,
                    "accuracy": 0.9,
                    "formality": 0.8,
                    "empathy": 0.5
                },
                communication_patterns={
                    "greeting_style": "formal",
                    "response_tone": "informative",
                    "question_style": "direct",
                    "closing_style": "professional"
                }
            ),
            "creative": PersonalityProfile(
                profile_id="creative",
                name="Creative Thinker",
                description="Imaginative, innovative, and inspiring communication",
                traits={
                    "creativity": 0.9,
                    "imagination": 0.9,
                    "inspiration": 0.8,
                    "flexibility": 0.8,
                    "formality": 0.4
                },
                communication_patterns={
                    "greeting_style": "inspiring",
                    "response_tone": "creative",
                    "question_style": "exploratory",
                    "closing_style": "motivational"
                }
            ),
            "analytical": PersonalityProfile(
                profile_id="analytical",
                name="Analytical Mind",
                description="Logical, detailed, and systematic communication",
                traits={
                    "logic": 0.9,
                    "precision": 0.9,
                    "thoroughness": 0.8,
                    "objectivity": 0.8,
                    "formality": 0.7
                },
                communication_patterns={
                    "greeting_style": "methodical",
                    "response_tone": "analytical",
                    "question_style": "investigative",
                    "closing_style": "conclusive"
                }
            ),
            "casual": PersonalityProfile(
                profile_id="casual",
                name="Casual Friend",
                description="Relaxed, informal, and conversational communication",
                traits={
                    "casualness": 0.9,
                    "friendliness": 0.8,
                    "humor": 0.8,
                    "approachability": 0.9,
                    "formality": 0.2
                },
                communication_patterns={
                    "greeting_style": "casual",
                    "response_tone": "conversational",
                    "question_style": "relaxed",
                    "closing_style": "friendly"
                }
            )
        }
        
        return profiles
    
    def _initialize_communication_styles(self) -> Dict[str, CommunicationStyleDetails]:
        """Initialize available communication styles"""
        styles = {
            "formal": CommunicationStyleDetails(
                style_id="formal",
                name="Formal",
                characteristics={
                    "vocabulary_level": "advanced",
                    "sentence_structure": "complex",
                    "politeness_level": "high",
                    "abbreviation_usage": "minimal",
                    "emoji_usage": "none"
                },
                language_patterns={
                    "greetings": ["Good day", "Hello there", "Greetings"],
                    "acknowledgments": ["I understand", "I comprehend", "I see"],
                    "transitions": ["Furthermore", "Moreover", "Additionally"],
                    "closings": ["Thank you", "Best regards", "Sincerely"]
                }
            ),
            "casual": CommunicationStyleDetails(
                style_id="casual",
                name="Casual",
                characteristics={
                    "vocabulary_level": "conversational",
                    "sentence_structure": "simple",
                    "politeness_level": "moderate",
                    "abbreviation_usage": "moderate",
                    "emoji_usage": "moderate"
                },
                language_patterns={
                    "greetings": ["Hey", "Hi there", "Hello"],
                    "acknowledgments": ["Got it", "I see", "Makes sense"],
                    "transitions": ["Also", "Plus", "And"],
                    "closings": ["Thanks", "See you", "Take care"]
                }
            ),
            "friendly": CommunicationStyleDetails(
                style_id="friendly",
                name="Friendly",
                characteristics={
                    "vocabulary_level": "conversational",
                    "sentence_structure": "varied",
                    "politeness_level": "high",
                    "abbreviation_usage": "minimal",
                    "emoji_usage": "moderate"
                },
                language_patterns={
                    "greetings": ["Hi there!", "Hello friend!", "Hey there!"],
                    "acknowledgments": ["That's great!", "I love that!", "Wonderful!"],
                    "transitions": ["You know what?", "Here's the thing", "The best part is"],
                    "closings": ["Have a great day!", "Take care!", "See you soon!"]
                }
            ),
            "professional": CommunicationStyleDetails(
                style_id="professional",
                name="Professional",
                characteristics={
                    "vocabulary_level": "business",
                    "sentence_structure": "structured",
                    "politeness_level": "high",
                    "abbreviation_usage": "minimal",
                    "emoji_usage": "none"
                },
                language_patterns={
                    "greetings": ["Good morning", "Hello", "Greetings"],
                    "acknowledgments": ["I understand", "I acknowledge", "I recognize"],
                    "transitions": ["In addition", "Furthermore", "Moreover"],
                    "closings": ["Thank you for your time", "Best regards", "Sincerely"]
                }
            ),
            "creative": CommunicationStyleDetails(
                style_id="creative",
                name="Creative",
                characteristics={
                    "vocabulary_level": "descriptive",
                    "sentence_structure": "varied",
                    "politeness_level": "moderate",
                    "abbreviation_usage": "minimal",
                    "emoji_usage": "moderate"
                },
                language_patterns={
                    "greetings": ["Greetings, creative soul!", "Hello, imagination seeker!", "Welcome, dreamer!"],
                    "acknowledgments": ["What a fascinating idea!", "That's absolutely brilliant!", "I'm inspired!"],
                    "transitions": ["Imagine this", "Picture this scenario", "Consider the possibilities"],
                    "closings": ["Keep dreaming big!", "Stay inspired!", "Keep creating!"]
                }
            )
        }
        
        return styles
    
    @log_execution_time
    async def adapt_personality(
        self,
        user_profile: Optional[UserProfile],
        conversation_context: ConversationContext,
        current_message: str,
        emotional_state: Optional[Dict[str, Any]] = None
    ) -> PersonalityProfile:
        """Adapt personality based on user profile and conversation context"""
        correlation_id = generate_correlation_id()
        start_time = datetime.utcnow()
        
        self.logger.info("Adapting personality for user", extra={
            "correlation_id": correlation_id,
            "user_id": getattr(user_profile, 'user_id', 'unknown') if user_profile else 'unknown'
        })
        
        try:
            # Determine base personality profile
            base_profile = await self._select_base_personality(user_profile, conversation_context)
            
            # Adapt based on conversation context
            adapted_profile = await self._adapt_to_conversation_context(
                base_profile, conversation_context, current_message
            )
            
            # Adapt based on emotional state
            if emotional_state:
                adapted_profile = await self._adapt_to_emotional_state(
                    adapted_profile, emotional_state
                )
            
            # Adapt based on conversation goals
            if hasattr(conversation_context, 'conversation_goals') and conversation_context.conversation_goals:
                adapted_profile = await self._adapt_to_conversation_goals(
                    adapted_profile, conversation_context.conversation_goals
                )
            
            # Store adaptation for learning
            await self._store_adaptation_history(
                user_profile.user_id if user_profile else 'unknown',
                base_profile.profile_id,
                adapted_profile.profile_id,
                conversation_context
            )
            
            # Calculate adaptation time
            adaptation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.adaptation_times.append(adaptation_time)
            
            if len(self.adaptation_times) > 100:
                self.adaptation_times = self.adaptation_times[-100:]
            
            self.logger.info(f"Personality adapted from {base_profile.profile_id} to {adapted_profile.profile_id}", extra={
                "correlation_id": correlation_id,
                "adaptation_time_ms": adaptation_time
            })
            
            return adapted_profile
            
        except Exception as e:
            self.logger.error(f"Failed to adapt personality: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            # Return default friendly profile on error
            return self.personality_profiles["friendly"]
    
    async def _select_base_personality(
        self,
        user_profile: Optional[UserProfile],
        conversation_context: ConversationContext
    ) -> PersonalityProfile:
        """Select base personality profile based on user preferences"""
        if not user_profile:
            return self.personality_profiles["friendly"]
        
        # Check if user has explicit personality preference
        if hasattr(user_profile, 'preferred_personality') and user_profile.preferred_personality:
            preferred = user_profile.preferred_personality
            if preferred in self.personality_profiles:
                return self.personality_profiles[preferred]
        
        # Analyze communication preferences to infer personality
        if hasattr(user_profile, 'communication_preferences'):
            preferences = user_profile.communication_preferences
            
            # Map preferences to personality traits
            personality_scores = defaultdict(float)
            
            if preferences.get("formality_level") == "high":
                personality_scores["professional"] += 0.8
                personality_scores["analytical"] += 0.6
            elif preferences.get("formality_level") == "low":
                personality_scores["casual"] += 0.8
                personality_scores["friendly"] += 0.6
            
            if preferences.get("communication_style") == "direct":
                personality_scores["professional"] += 0.7
                personality_scores["analytical"] += 0.6
            elif preferences.get("communication_style") == "supportive":
                personality_scores["friendly"] += 0.8
                personality_scores["casual"] += 0.5
            
            if preferences.get("detail_level") == "high":
                personality_scores["analytical"] += 0.8
                personality_scores["professional"] += 0.6
            elif preferences.get("detail_level") == "low":
                personality_scores["casual"] += 0.7
                personality_scores["friendly"] += 0.5
            
            # Select personality with highest score
            if personality_scores:
                best_personality = max(personality_scores.items(), key=lambda x: x[1])[0]
                return self.personality_profiles[best_personality]
        
        # Default to friendly if no clear preference
        return self.personality_profiles["friendly"]
    
    async def _adapt_to_conversation_context(
        self,
        base_profile: PersonalityProfile,
        conversation_context: ConversationContext,
        current_message: str
    ) -> PersonalityProfile:
        """Adapt personality based on conversation context"""
        adapted_profile = base_profile
        
        # Analyze conversation topic and adjust accordingly
        if hasattr(conversation_context, 'conversation_topic'):
            topic = conversation_context.conversation_topic.lower()
            
            # Professional topics
            if any(word in topic for word in ["work", "business", "professional", "career"]):
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["professional"], 0.6
                )
            
            # Creative topics
            elif any(word in topic for word in ["creative", "art", "design", "imagination", "ideas"]):
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["creative"], 0.7
                )
            
            # Analytical topics
            elif any(word in topic for word in ["analysis", "data", "research", "study", "investigation"]):
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["analytical"], 0.6
                )
        
        # Analyze conversation tone
        if hasattr(conversation_context, 'conversation_tone'):
            tone = conversation_context.conversation_tone
            
            if tone == "serious":
                adapted_profile = self._blend_personalities(
                    adapted_profile, self.personality_profiles["professional"], 0.5
                )
            elif tone == "casual":
                adapted_profile = self._blend_personalities(
                    adapted_profile, self.personality_profiles["casual"], 0.5
                )
            elif tone == "friendly":
                adapted_profile = self._blend_personalities(
                    adapted_profile, self.personality_profiles["friendly"], 0.5
                )
        
        return adapted_profile
    
    async def _adapt_to_emotional_state(
        self,
        base_profile: PersonalityProfile,
        emotional_state: Dict[str, Any]
    ) -> PersonalityProfile:
        """Adapt personality based on user's emotional state"""
        adapted_profile = base_profile
        
        # Get primary emotion
        primary_emotion = emotional_state.get("primary_emotion", "").lower()
        emotion_intensity = emotional_state.get("intensity", 0.5)
        
        if primary_emotion in ["sad", "depressed", "down"]:
            # Increase empathy and supportiveness
            adapted_profile = self._blend_personalities(
                base_profile, self.personality_profiles["friendly"], 0.6
            )
            # Boost empathy trait
            adapted_profile.traits["empathy"] = min(1.0, adapted_profile.traits.get("empathy", 0.5) + 0.3)
            adapted_profile.traits["supportiveness"] = min(1.0, adapted_profile.traits.get("supportiveness", 0.5) + 0.3)
        
        elif primary_emotion in ["angry", "frustrated", "irritated"]:
            # Increase patience and formality
            adapted_profile = self._blend_personalities(
                base_profile, self.personality_profiles["professional"], 0.5
            )
            # Boost clarity and patience
            adapted_profile.traits["clarity"] = min(1.0, adapted_profile.traits.get("clarity", 0.5) + 0.2)
            adapted_profile.traits["formality"] = min(1.0, adapted_profile.traits.get("formality", 0.5) + 0.2)
        
        elif primary_emotion in ["excited", "happy", "enthusiastic"]:
            # Match enthusiasm and creativity
            adapted_profile = self._blend_personalities(
                base_profile, self.personality_profiles["creative"], 0.5
            )
            # Boost enthusiasm and creativity
            adapted_profile.traits["enthusiasm"] = min(1.0, adapted_profile.traits.get("enthusiasm", 0.5) + 0.3)
            adapted_profile.traits["creativity"] = min(1.0, adapted_profile.traits.get("creativity", 0.5) + 0.2)
        
        elif primary_emotion in ["anxious", "worried", "stressed"]:
            # Increase calmness and analytical thinking
            adapted_profile = self._blend_personalities(
                base_profile, self.personality_profiles["analytical"], 0.5
            )
            # Boost logic and precision
            adapted_profile.traits["logic"] = min(1.0, adapted_profile.traits.get("logic", 0.5) + 0.2)
            adapted_profile.traits["precision"] = min(1.0, adapted_profile.traits.get("precision", 0.5) + 0.2)
        
        return adapted_profile
    
    async def _adapt_to_conversation_goals(
        self,
        base_profile: PersonalityProfile,
        conversation_goals: List[str]
    ) -> PersonalityProfile:
        """Adapt personality based on conversation goals"""
        adapted_profile = base_profile
        
        for goal in conversation_goals:
            goal_lower = goal.lower()
            
            if "learn" in goal_lower or "understand" in goal_lower:
                # Increase analytical and educational traits
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["analytical"], 0.6
                )
            
            elif "create" in goal_lower or "design" in goal_lower:
                # Increase creative traits
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["creative"], 0.7
                )
            
            elif "solve" in goal_lower or "fix" in goal_lower:
                # Increase analytical and professional traits
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["analytical"], 0.5
                )
                adapted_profile = self._blend_personalities(
                    adapted_profile, self.personality_profiles["professional"], 0.4
                )
            
            elif "support" in goal_lower or "help" in goal_lower:
                # Increase friendly and supportive traits
                adapted_profile = self._blend_personalities(
                    base_profile, self.personality_profiles["friendly"], 0.6
                )
        
        return adapted_profile
    
    def _blend_personalities(
        self,
        profile1: PersonalityProfile,
        profile2: PersonalityProfile,
        blend_ratio: float
    ) -> PersonalityProfile:
        """Blend two personality profiles with specified ratio"""
        blended_profile = PersonalityProfile(
            profile_id=f"blended_{profile1.profile_id}_{profile2.profile_id}",
            name=f"Blended: {profile1.name} + {profile2.name}",
            description=f"Adaptive blend of {profile1.name} and {profile2.name}",
            traits={},
            communication_patterns={}
        )
        
        # Blend traits
        all_traits = set(profile1.traits.keys()) | set(profile2.traits.keys())
        for trait in all_traits:
            trait1_value = profile1.traits.get(trait, 0.5)
            trait2_value = profile2.traits.get(trait, 0.5)
            blended_value = trait1_value * (1 - blend_ratio) + trait2_value * blend_ratio
            blended_profile.traits[trait] = round(blended_value, 2)
        
        # Blend communication patterns
        all_patterns = set(profile1.communication_patterns.keys()) | set(profile2.communication_patterns.keys())
        for pattern in all_patterns:
            pattern1_value = profile1.communication_patterns.get(pattern, "neutral")
            pattern2_value = profile2.communication_patterns.get(pattern, "neutral")
            
            # Simple blending for patterns (could be more sophisticated)
            if random.random() < blend_ratio:
                blended_profile.communication_patterns[pattern] = pattern2_value
            else:
                blended_profile.communication_patterns[pattern] = pattern1_value
        
        return blended_profile
    
    async def _store_adaptation_history(
        self,
        user_id: str,
        base_profile_id: str,
        adapted_profile_id: str,
        conversation_context: ConversationContext
    ) -> None:
        """Store personality adaptation history for learning"""
        if user_id not in self.user_personality_history:
            self.user_personality_history[user_id] = []
        
        history_entry = {
            "timestamp": datetime.utcnow(),
            "base_profile": base_profile_id,
            "adapted_profile": adapted_profile_id,
            "conversation_id": getattr(conversation_context, 'conversation_id', 'unknown'),
            "conversation_topic": getattr(conversation_context, 'conversation_topic', 'unknown'),
            "conversation_tone": getattr(conversation_context, 'conversation_tone', 'unknown')
        }
        
        self.user_personality_history[user_id].append(history_entry)
        
        # Keep only last 100 entries per user
        if len(self.user_personality_history[user_id]) > 100:
            self.user_personality_history[user_id] = self.user_personality_history[user_id][-100:]
    
    async def get_communication_style(
        self,
        personality_profile: PersonalityProfile,
        context_type: str = "general"
    ) -> CommunicationStyle:
        """Get appropriate communication style for personality profile"""
        start_time = datetime.utcnow()
        
        # Map personality to communication style
        style_mapping = {
            "friendly": "friendly",
            "professional": "professional",
            "creative": "creative",
            "analytical": "formal",
            "casual": "casual"
        }
        
        # Get base style
        base_style_id = style_mapping.get(personality_profile.profile_id, "friendly")
        base_style = self.communication_styles[base_style_id]
        
        # Adapt style based on context
        if context_type == "formal":
            adapted_style = self.communication_styles["formal"]
        elif context_type == "casual":
            adapted_style = self.communication_styles["casual"]
        else:
            adapted_style = base_style
        
        # Calculate selection time
        selection_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.style_selection_times.append(selection_time)
        
        if len(self.style_selection_times) > 100:
            self.style_selection_times = self.style_selection_times[-100:]
        
        return adapted_style
    
    async def get_personality_insights(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get insights about user's personality adaptation patterns"""
        if user_id not in self.user_personality_history:
            return {"error": "No personality history for user"}
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_history = [
            entry for entry in self.user_personality_history[user_id]
            if entry["timestamp"] > cutoff_date
        ]
        
        if not recent_history:
            return {"message": f"No personality data in last {days} days"}
        
        # Analyze adaptation patterns
        profile_transitions = {}
        topic_preferences = defaultdict(int)
        tone_preferences = defaultdict(int)
        
        for entry in recent_history:
            transition = f"{entry['base_profile']}->{entry['adapted_profile']}"
            profile_transitions[transition] = profile_transitions.get(transition, 0) + 1
            
            topic_preferences[entry['conversation_topic']] += 1
            tone_preferences[entry['conversation_tone']] += 1
        
        # Find most common adaptations
        most_common_transitions = sorted(
            profile_transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_adaptations": len(recent_history),
            "most_common_transitions": most_common_transitions,
            "topic_preferences": dict(topic_preferences),
            "tone_preferences": dict(tone_preferences),
            "adaptation_frequency": len(recent_history) / days
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.adaptation_times:
            return {
                "avg_adaptation_time_ms": 0.0,
                "avg_style_selection_time_ms": 0.0,
                "total_adaptations": 0
            }
        
        return {
            "avg_adaptation_time_ms": sum(self.adaptation_times) / len(self.adaptation_times),
            "min_adaptation_time_ms": min(self.adaptation_times),
            "max_adaptation_time_ms": max(self.adaptation_times),
            "avg_style_selection_time_ms": sum(self.style_selection_times) / len(self.style_selection_times) if self.style_selection_times else 0.0,
            "total_adaptations": len(self.adaptation_times),
            "active_users_tracked": len(self.user_personality_history)
        }
    
    async def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old personality adaptation data"""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        total_removed = 0
        
        for user_id in list(self.user_personality_history.keys()):
            original_count = len(self.user_personality_history[user_id])
            self.user_personality_history[user_id] = [
                entry for entry in self.user_personality_history[user_id]
                if entry["timestamp"] > cutoff_date
            ]
            
            removed = original_count - len(self.user_personality_history[user_id])
            total_removed += removed
            
            # Remove user if no recent data
            if not self.user_personality_history[user_id]:
                del self.user_personality_history[user_id]
        
        if total_removed > 0:
            self.logger.info(f"Cleaned up {total_removed} old personality adaptation entries")
        
        return total_removed
