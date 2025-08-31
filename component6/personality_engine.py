"""
Personality Engine Module for Component 6.
Handles AI personality adaptation and communication style management.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from shared.schemas import UserProfile, ConversationContext
from shared.utils import get_logger, generate_correlation_id, SimpleCache
from config.settings import settings


class PersonalityEngine:
    """Manages AI personality adaptation and communication style"""
    
    def __init__(self):
        """Initialize personality engine"""
        self.logger = get_logger("personality_engine")
        self.communication_styles = self._load_communication_styles()
        self.profile_cache = SimpleCache(default_ttl=7200)
        self.adaptation_times = []
        self.logger.info("Personality Engine initialized")
    
    async def adapt_personality(
        self,
        user_profile: UserProfile,
        conversation_context: ConversationContext,
        current_emotional_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Adapt AI personality based on user profile and context"""
        try:
            # Generate basic adaptation
            adaptation = {
                "primary_style": user_profile.communication_style.value,
                "emotional_tone": self._determine_emotional_tone(user_profile, current_emotional_state),
                "response_guidelines": self._generate_response_guidelines(user_profile),
                "interaction_preferences": {
                    "proactive_engagement": user_profile.proactive_engagement,
                    "follow_up_questions": user_profile.conversation_depth > 0.5,
                    "memory_references": True
                },
                "adaptation_metadata": {
                    "adaptation_timestamp": datetime.utcnow(),
                    "confidence_level": 0.8
                }
            }
            
            return adaptation
            
        except Exception as e:
            self.logger.error(f"Failed to adapt personality: {e}")
            return self._get_default_adaptation()
    
    def _determine_emotional_tone(self, user_profile: UserProfile, emotional_state: Optional[Dict[str, Any]]) -> str:
        """Determine appropriate emotional tone"""
        if user_profile.emotional_matching and emotional_state:
            emotion = emotional_state.get("primary_emotion", "neutral")
            mapping = {
                "happy": "encouraging", "sad": "empathetic", "angry": "calm",
                "anxious": "calm", "excited": "encouraging", "stressed": "calming"
            }
            return mapping.get(emotion, "supportive")
        return "supportive"
    
    def _generate_response_guidelines(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Generate response guidelines"""
        return {
            "response_length": user_profile.preferred_response_length,
            "humor_level": user_profile.humor_preference,
            "formality_level": user_profile.formality_level,
            "depth_level": user_profile.conversation_depth
        }
    
    def _load_communication_styles(self) -> Dict[str, Dict[str, Any]]:
        """Load communication styles"""
        return {
            "friendly": {"tone": "warm", "approach": "supportive"},
            "formal": {"tone": "respectful", "approach": "professional"},
            "casual": {"tone": "relaxed", "approach": "conversational"}
        }
    
    def _get_default_adaptation(self) -> Dict[str, Any]:
        """Get default adaptation"""
        return {
            "primary_style": "friendly",
            "emotional_tone": "supportive",
            "response_guidelines": {"response_length": "moderate"},
            "interaction_preferences": {"follow_up_questions": True},
            "adaptation_metadata": {"confidence_level": 0.5}
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_adaptations": len(self.adaptation_times),
            "cache_size": self.profile_cache.size()
        }
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            test_profile = UserProfile(
                user_id="test_user",
                communication_style="friendly",
                preferred_response_length="moderate"
            )
            
            test_context = ConversationContext(
                user_id="test_user",
                conversation_id="test_conversation",
                personality_profile=test_profile
            )
            
            adaptation = await self.adapt_personality(test_profile, test_context)
            return isinstance(adaptation, dict) and "primary_style" in adaptation
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
