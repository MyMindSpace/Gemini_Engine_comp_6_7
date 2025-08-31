"""
Proactive Engine Module for Component 6.
Handles AI-initiated conversations and proactive engagement.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from shared.schemas import ProactiveMessage, UserProfile
from shared.utils import get_logger, generate_correlation_id
from config.settings import settings


class ProactiveEngine:
    """Manages AI-initiated conversations and proactive engagement"""
    
    def __init__(self):
        """Initialize proactive engine"""
        self.logger = get_logger("proactive_engine")
        self.message_queue = []
        self.scheduled_messages = {}
        self.user_patterns = {}
        self.engagement_success_rates = []
        self.logger.info("Proactive Engine initialized")
    
    async def generate_proactive_message(
        self,
        user_id: str,
        trigger_event: str,
        context: Dict[str, Any],
        user_profile: Optional[UserProfile] = None
    ) -> Optional[ProactiveMessage]:
        """Generate proactive message based on trigger and context"""
        try:
            # Check if proactive messages are enabled
            if user_profile and not user_profile.proactive_engagement:
                return None
            
            # Check daily limits
            if not self._check_daily_limits(user_id):
                return None
            
            # Generate message content
            message_content = await self._generate_message_content(trigger_event, context, user_profile)
            
            if not message_content:
                return None
            
            # Calculate optimal timing
            optimal_timing = await self._calculate_optimal_timing(user_id, trigger_event)
            
            # Create proactive message
            proactive_message = ProactiveMessage(
                message_id=generate_correlation_id(),
                user_id=user_id,
                trigger_event=trigger_event,
                message_content=message_content,
                optimal_timing=optimal_timing,
                expected_response="general_response",
                priority_level="medium"
            )
            
            # Schedule message
            await self._schedule_message(proactive_message)
            
            return proactive_message
            
        except Exception as e:
            self.logger.error(f"Failed to generate proactive message: {e}")
            return None
    
    async def _generate_message_content(
        self,
        trigger_event: str,
        context: Dict[str, Any],
        user_profile: Optional[UserProfile]
    ) -> str:
        """Generate appropriate message content for trigger event"""
        templates = {
            "user_inactive": "Hi there! I was thinking about you. How are you feeling today?",
            "goal_milestone": "Congratulations! I noticed you've made great progress.",
            "stress_pattern": "I've noticed some patterns that might be worth discussing.",
            "follow_up_needed": "I was thinking about our last conversation. How are things going?",
            "weekly_reflection": "It's been a while since we reflected together. What's been on your mind?"
        }
        
        template = templates.get(trigger_event, templates["user_inactive"])
        
        # Basic personalization
        if "{user_name}" in template and "user_name" in context:
            template = template.replace("{user_name}", context["user_name"])
        
        return template
    
    async def _calculate_optimal_timing(self, user_id: str, trigger_event: str) -> datetime:
        """Calculate optimal timing for message delivery"""
        # Default to current time + small delay
        base_time = datetime.utcnow() + timedelta(minutes=30)
        
        # Adjust based on trigger urgency
        urgency_delays = {
            "stress_pattern": timedelta(minutes=30),
            "user_inactive": timedelta(hours=2),
            "follow_up_needed": timedelta(hours=1),
            "weekly_reflection": timedelta(hours=12)
        }
        
        if trigger_event in urgency_delays:
            base_time = datetime.utcnow() + urgency_delays[trigger_event]
        
        return base_time
    
    async def _schedule_message(self, message: ProactiveMessage) -> None:
        """Schedule message for delivery"""
        if message.user_id not in self.scheduled_messages:
            self.scheduled_messages[message.user_id] = []
        
        self.scheduled_messages[message.user_id].append(message)
        self.logger.debug(f"Message scheduled for {message.optimal_timing}")
    
    def _check_daily_limits(self, user_id: str) -> bool:
        """Check if user hasn't exceeded daily proactive message limits"""
        today = datetime.utcnow().date()
        today_messages = 0
        
        for message in self.scheduled_messages.get(user_id, []):
            if message.optimal_timing.date() == today:
                today_messages += 1
        
        max_daily = getattr(settings.personality, 'max_proactive_messages_per_day', 5)
        return today_messages < max_daily
    
    async def get_pending_messages(self, user_id: str) -> List[ProactiveMessage]:
        """Get pending proactive messages for user"""
        current_time = datetime.utcnow()
        pending_messages = []
        
        for message in self.scheduled_messages.get(user_id, []):
            if (message.optimal_timing <= current_time and 
                message.delivery_status == "pending"):
                pending_messages.append(message)
        
        return pending_messages
    
    async def mark_message_delivered(self, message_id: str, user_id: str) -> bool:
        """Mark message as delivered"""
        for message in self.scheduled_messages.get(user_id, []):
            if message.message_id == message_id:
                message.delivery_status = "delivered"
                return True
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "avg_engagement_rate": sum(self.engagement_success_rates) / len(self.engagement_success_rates) if self.engagement_success_rates else 0.0,
            "total_messages_sent": len(self.engagement_success_rates),
            "pending_messages": sum(len(messages) for messages in self.scheduled_messages.values()),
            "active_users": len(self.user_patterns)
        }
