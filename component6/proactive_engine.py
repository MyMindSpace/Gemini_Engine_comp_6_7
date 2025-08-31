"""
Proactive Engine Module for Component 6.
Initiates AI conversations based on context and user patterns.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import random
from collections import defaultdict

from shared.schemas import (
    ProactiveMessage, ConversationContext, UserProfile,
    MemoryContext, EnhancedResponse
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id
)
from config.settings import settings


class ProactiveEngine:
    """Initiates proactive conversations based on context and user patterns"""
    
    def __init__(self):
        """Initialize proactive engine"""
        self.logger = get_logger("proactive_engine")
        
        # Proactive triggers and patterns
        self.trigger_patterns = self._initialize_trigger_patterns()
        self.message_templates = self._initialize_message_templates()
        
        # User interaction tracking
        self.user_interaction_history = {}
        self.proactive_success_history = {}
        self.trigger_effectiveness = defaultdict(list)
        
        # Timing and frequency controls
        self.min_interval_hours = 2  # Minimum time between proactive messages
        self.max_daily_messages = 5  # Maximum proactive messages per day
        
        # Performance tracking
        self.trigger_evaluation_times = []
        self.message_generation_times = []
        
        self.logger.info("Proactive Engine initialized")
    
    def _initialize_trigger_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize proactive trigger patterns"""
        return {
            "time_based": {
                "enabled": True,
                "priority": "medium",
                "triggers": ["daily_checkin", "weekly_review", "optimal_timing"]
            },
            "context_based": {
                "enabled": True,
                "priority": "high",
                "triggers": ["memory_surfacing", "goal_alignment", "emotional_support"]
            },
            "behavior_based": {
                "enabled": True,
                "priority": "medium",
                "triggers": ["inactivity_detection", "engagement_drop", "success_celebration"]
            }
        }
    
    def _initialize_message_templates(self) -> Dict[str, List[str]]:
        """Initialize proactive message templates"""
        return {
            "daily_checkin": [
                "Good morning! How are you feeling today?",
                "Hi there! Ready to start your day?",
                "Morning! What's on your mind today?"
            ],
            "memory_surfacing": [
                "I was thinking about {memory_topic}. How does that relate to what you're working on now?",
                "Remember when {memory_topic}? That might be relevant to your current situation."
            ],
            "goal_alignment": [
                "How are you progressing toward {goal_name}?",
                "Let's check in on your {goal_name} goal. What's the next step?"
            ],
            "emotional_support": [
                "I sense you might be feeling {emotion}. How are you doing?",
                "It seems like you could use some support right now. What's on your mind?"
            ],
            "inactivity_detection": [
                "Haven't heard from you in a while. How are things going?",
                "I miss our conversations! What's new with you?"
            ],
            "success_celebration": [
                "Congratulations on {achievement}! That's fantastic!",
                "Wow! You've accomplished {achievement}. How does it feel?"
            ]
        }
    
    @log_execution_time
    async def evaluate_proactive_triggers(
        self,
        user_id: str,
        current_context: ConversationContext,
        memory_context: Optional[MemoryContext] = None,
        user_profile: Optional[UserProfile] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate all proactive triggers for a user"""
        correlation_id = generate_correlation_id()
        start_time = datetime.utcnow()
        
        try:
            # Check if user is eligible for proactive messages
            if not await self._is_user_eligible(user_id):
                return []
            
            active_triggers = []
            
            # Evaluate time-based triggers
            if self.trigger_patterns["time_based"]["enabled"]:
                time_triggers = await self._evaluate_time_based_triggers(user_id, current_context)
                active_triggers.extend(time_triggers)
            
            # Evaluate context-based triggers
            if self.trigger_patterns["context_based"]["enabled"]:
                context_triggers = await self._evaluate_context_based_triggers(
                    user_id, current_context, memory_context, user_profile
                )
                active_triggers.extend(context_triggers)
            
            # Evaluate behavior-based triggers
            if self.trigger_patterns["behavior_based"]["enabled"]:
                behavior_triggers = await self._evaluate_behavior_based_triggers(user_id, current_context)
                active_triggers.extend(behavior_triggers)
            
            # Sort triggers by priority and score
            active_triggers.sort(key=lambda x: (x["priority_score"], x["trigger_score"]), reverse=True)
            
            # Calculate evaluation time
            evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.trigger_evaluation_times.append(evaluation_time)
            
            if len(self.trigger_evaluation_times) > 100:
                self.trigger_evaluation_times = self.trigger_evaluation_times[-100:]
            
            return active_triggers
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate proactive triggers: {e}", extra={
                "correlation_id": correlation_id,
                "user_id": user_id,
                "error": str(e)
            })
            return []
    
    async def _is_user_eligible(self, user_id: str) -> bool:
        """Check if user is eligible for proactive messages"""
        current_time = datetime.utcnow()
        
        # Check minimum interval
        if user_id in self.user_interaction_history:
            last_interaction = self.user_interaction_history[user_id].get("last_proactive_message")
            if last_interaction:
                time_since_last = current_time - last_interaction
                if time_since_last < timedelta(hours=self.min_interval_hours):
                    return False
        
        # Check daily limit
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        if user_id in self.user_interaction_history:
            today_messages = self.user_interaction_history[user_id].get("daily_messages", [])
            today_messages = [msg for msg in today_messages if msg > today_start]
            if len(today_messages) >= self.max_daily_messages:
                return False
        
        return True
    
    async def _evaluate_time_based_triggers(
        self,
        user_id: str,
        current_context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Evaluate time-based proactive triggers"""
        triggers = []
        current_time = datetime.utcnow()
        
        # Daily checkin trigger
        if user_id in self.user_interaction_history:
            last_checkin = self.user_interaction_history[user_id].get("last_daily_checkin")
            if not last_checkin or (current_time - last_checkin).days >= 1:
                triggers.append({
                    "trigger_type": "daily_checkin",
                    "trigger_score": 0.8,
                    "priority_score": 0.7,
                    "reason": "Daily checkin due",
                    "context": {"time_of_day": current_time.hour}
                })
        
        # Weekly review trigger
        if user_id in self.user_interaction_history:
            last_weekly_review = self.user_interaction_history[user_id].get("last_weekly_review")
            if not last_weekly_review or (current_time - last_weekly_review).days >= 7:
                triggers.append({
                    "trigger_type": "weekly_review",
                    "trigger_score": 0.9,
                    "priority_score": 0.8,
                    "reason": "Weekly review due",
                    "context": {"week_number": current_time.isocalendar()[1]}
                })
        
        return triggers
    
    async def _evaluate_context_based_triggers(
        self,
        user_id: str,
        current_context: ConversationContext,
        memory_context: Optional[MemoryContext],
        user_profile: Optional[UserProfile]
    ) -> List[Dict[str, Any]]:
        """Evaluate context-based proactive triggers"""
        triggers = []
        
        # Memory surfacing trigger
        if memory_context and memory_context.selected_memories:
            relevant_memories = await self._find_relevant_memories(
                memory_context.selected_memories, current_context
            )
            
            if relevant_memories:
                triggers.append({
                    "trigger_type": "memory_surfacing",
                    "trigger_score": 0.8,
                    "priority_score": 0.8,
                    "reason": "Relevant memories available",
                    "context": {
                        "relevant_memories": len(relevant_memories),
                        "memory_topics": [m.get("content_summary", "")[:50] for m in relevant_memories[:3]]
                    }
                })
        
        # Goal alignment trigger
        if user_profile and hasattr(user_profile, 'goals') and user_profile.goals:
            active_goals = [goal for goal in user_profile.goals if goal.get("status") == "active"]
            if active_goals:
                goals_needing_attention = await self._identify_goals_needing_attention(active_goals)
                if goals_needing_attention:
                    triggers.append({
                        "trigger_type": "goal_alignment",
                        "trigger_score": 0.9,
                        "priority_score": 0.9,
                        "reason": "Goals need attention",
                        "context": {
                            "goals_count": len(goals_needing_attention),
                            "goal_names": [goal.get("name", "") for goal in goals_needing_attention[:3]]
                        }
                    })
        
        return triggers
    
    async def _evaluate_behavior_based_triggers(
        self,
        user_id: str,
        current_context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Evaluate behavior-based proactive triggers"""
        triggers = []
        
        if user_id not in self.user_interaction_history:
            return triggers
        
        user_history = self.user_interaction_history[user_id]
        current_time = datetime.utcnow()
        
        # Inactivity detection
        last_interaction = user_history.get("last_interaction")
        if last_interaction:
            time_since_last = current_time - last_interaction
            if time_since_last > timedelta(hours=24):
                triggers.append({
                    "trigger_type": "inactivity_detection",
                    "trigger_score": 0.7,
                    "priority_score": 0.6,
                    "reason": "User inactive for 24+ hours",
                    "context": {"hours_inactive": time_since_last.total_seconds() / 3600}
                })
        
        return triggers
    
    async def _find_relevant_memories(
        self,
        memories: List[Dict[str, Any]],
        current_context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Find memories relevant to current conversation context"""
        relevant_memories = []
        
        if not hasattr(current_context, 'conversation_topic'):
            return relevant_memories
        
        current_topic = current_context.conversation_topic.lower()
        
        for memory in memories:
            memory_content = memory.get("content_summary", "").lower()
            memory_type = memory.get("memory_type", "")
            
            # Simple relevance scoring
            relevance_score = 0
            
            # Check content similarity
            topic_words = set(current_topic.split())
            memory_words = set(memory_content.split())
            word_overlap = len(topic_words.intersection(memory_words))
            
            if word_overlap > 0:
                relevance_score += word_overlap * 0.1
            
            # Boost relevance for certain memory types
            if memory_type in ["insight", "emotion"]:
                relevance_score += 0.2
            
            # Check importance score
            importance = memory.get("importance_score", 0)
            relevance_score += importance * 0.3
            
            if relevance_score > 0.3:
                relevant_memories.append(memory)
        
        return relevant_memories[:5]
    
    async def _identify_goals_needing_attention(
        self,
        active_goals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify goals that need attention or support"""
        goals_needing_attention = []
        
        for goal in active_goals:
            # Check if goal is behind schedule
            if "deadline" in goal and "progress" in goal:
                deadline = goal["deadline"]
                progress = goal["progress"]
                
                if isinstance(deadline, str):
                    try:
                        deadline_date = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
                        days_remaining = (deadline_date - datetime.utcnow()).days
                        
                        # Goal needs attention if less than 30 days remaining and progress < 70%
                        if days_remaining < 30 and progress < 0.7:
                            goals_needing_attention.append(goal)
                    except:
                        pass
        
        return goals_needing_attention
    
    @log_execution_time
    async def generate_proactive_message(
        self,
        trigger: Dict[str, Any],
        user_profile: Optional[UserProfile] = None,
        memory_context: Optional[MemoryContext] = None
    ) -> Optional[ProactiveMessage]:
        """Generate a proactive message based on the trigger"""
        correlation_id = generate_correlation_id()
        start_time = datetime.utcnow()
        
        try:
            trigger_type = trigger["trigger_type"]
            
            # Get appropriate message template
            if trigger_type in self.message_templates:
                template = random.choice(self.message_templates[trigger_type])
            else:
                template = "Hi there! I was thinking about you. How are things going?"
            
            # Personalize the message
            personalized_message = await self._personalize_message(
                template, trigger, user_profile, memory_context
            )
            
            # Create proactive message
            proactive_message = ProactiveMessage(
                message_id=generate_correlation_id(),
                trigger_type=trigger_type,
                trigger_reason=trigger["reason"],
                message_content=personalized_message,
                priority_level=self._get_priority_level(trigger["priority_score"]),
                context_data=trigger.get("context", {}),
                generated_at=datetime.utcnow(),
                estimated_engagement=trigger["trigger_score"]
            )
            
            # Calculate generation time
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.message_generation_times.append(generation_time)
            
            if len(self.message_generation_times) > 100:
                self.message_generation_times = self.message_generation_times[-100:]
            
            return proactive_message
            
        except Exception as e:
            self.logger.error(f"Failed to generate proactive message: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            return None
    
    async def _personalize_message(
        self,
        template: str,
        trigger: Dict[str, Any],
        user_profile: Optional[UserProfile],
        memory_context: Optional[MemoryContext]
    ) -> str:
        """Personalize the message template with user-specific information"""
        personalized_message = template
        
        # Replace placeholders with actual values
        if "{memory_topic}" in personalized_message and memory_context:
            if memory_context.selected_memories:
                memory = memory_context.selected_memories[0]
                topic = memory.get("content_summary", "your experiences")[:50]
                personalized_message = personalized_message.replace("{memory_topic}", topic)
        
        if "{goal_name}" in personalized_message and user_profile:
            if hasattr(user_profile, 'goals') and user_profile.goals:
                active_goals = [goal for goal in user_profile.goals if goal.get("status") == "active"]
                if active_goals:
                    goal_name = active_goals[0].get("name", "your goals")
                    personalized_message = personalized_message.replace("{goal_name}", goal_name)
        
        if "{emotion}" in personalized_message:
            context = trigger.get("context", {})
            emotion = context.get("emotion", "overwhelmed")
            personalized_message = personalized_message.replace("{emotion}", emotion)
        
        if "{achievement}" in personalized_message:
            context = trigger.get("context", {})
            achievement = context.get("achievement", "your progress")
            personalized_message = personalized_message.replace("{achievement}", achievement)
        
        return personalized_message
    
    def _get_priority_level(self, priority_score: float) -> str:
        """Convert priority score to priority level"""
        if priority_score >= 0.8:
            return "high"
        elif priority_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    async def record_user_interaction(
        self,
        user_id: str,
        interaction_type: str,
        engagement_score: Optional[float] = None,
        response_content: Optional[str] = None
    ) -> None:
        """Record user interaction for proactive trigger evaluation"""
        current_time = datetime.utcnow()
        
        if user_id not in self.user_interaction_history:
            self.user_interaction_history[user_id] = {
                "last_interaction": current_time,
                "last_proactive_message": None,
                "last_daily_checkin": None,
                "last_weekly_review": None,
                "recent_interactions": [],
                "daily_messages": [],
                "recent_achievements": []
            }
        
        user_history = self.user_interaction_history[user_id]
        
        # Update last interaction time
        user_history["last_interaction"] = current_time
        
        # Record interaction details
        interaction_record = {
            "timestamp": current_time,
            "type": interaction_type,
            "engagement_score": engagement_score or 0.5,
            "response_content": response_content
        }
        
        user_history["recent_interactions"].append(interaction_record)
        
        # Keep only last 20 interactions
        if len(user_history["recent_interactions"]) > 20:
            user_history["recent_interactions"] = user_history["recent_interactions"][-20:]
        
        # Update daily message count
        user_history["daily_messages"].append(current_time)
        
        # Clean up old daily messages
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        user_history["daily_messages"] = [
            msg for msg in user_history["daily_messages"]
            if msg > today_start
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.trigger_evaluation_times:
            return {
                "avg_trigger_evaluation_ms": 0.0,
                "avg_message_generation_ms": 0.0,
                "total_triggers_evaluated": 0
            }
        
        return {
            "avg_trigger_evaluation_ms": sum(self.trigger_evaluation_times) / len(self.trigger_evaluation_times),
            "min_trigger_evaluation_ms": min(self.trigger_evaluation_times),
            "max_trigger_evaluation_ms": max(self.trigger_evaluation_times),
            "avg_message_generation_ms": sum(self.message_generation_times) / len(self.message_generation_times) if self.message_generation_times else 0.0,
            "total_triggers_evaluated": len(self.trigger_evaluation_times),
            "active_users_tracked": len(self.user_interaction_history)
        }
