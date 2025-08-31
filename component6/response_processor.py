"""Response processing and enhancement module for Component 6."""

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from shared.schemas import EnhancedResponse, ConversationContext
from shared.utils import generate_content_hash, format_timestamp
from config.settings import Settings

logger = logging.getLogger(__name__)

class ResponseProcessor:
    """Processes and enhances raw Gemini responses."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enhancement_cache = {}
        
    async def process_response(
        self,
        raw_response: str,
        conversation_context: ConversationContext,
        response_metadata: Optional[Dict[str, Any]] = None
    ) -> EnhancedResponse:
        """Process and enhance a raw response."""
        
        if not raw_response or not conversation_context:
            raise ValueError("Response and conversation context are required")
        
        try:
            # Clean and normalize the response
            cleaned_response = await self._clean_response(raw_response)
            
            # Enhance the response
            enhanced_content = await self._enhance_response(
                cleaned_response,
                conversation_context
            )
            
            # Generate quality metrics
            quality_metrics = await self._generate_quality_metrics(
                enhanced_content,
                conversation_context
            )
            
            # Create enhanced response
            enhanced_response = EnhancedResponse(
                content=enhanced_content,
                conversation_id=conversation_context.conversation_id,
                user_id=conversation_context.user_id,
                response_id=f"resp_{generate_content_hash(enhanced_content)[:8]}",
                timestamp=datetime.now(),
                quality_metrics=quality_metrics,
                metadata={
                    "original_length": len(raw_response),
                    "enhanced_length": len(enhanced_content),
                    "processing_timestamp": format_timestamp(datetime.now()),
                    **(response_metadata or {})
                }
            )
            
            logger.info(f"Processed response for conversation {conversation_context.conversation_id}")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            raise
    
    async def _clean_response(self, raw_response: str) -> str:
        """Clean and normalize the raw response."""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', raw_response.strip())
        
        # Ensure proper sentence endings
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        # Remove any control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned
    
    async def _enhance_response(
        self,
        cleaned_response: str,
        conversation_context: ConversationContext
    ) -> str:
        """Enhance the response with additional elements."""
        
        enhanced = cleaned_response
        
        # Add follow-up question if appropriate
        if await self._should_add_follow_up(conversation_context):
            follow_up = await self._generate_follow_up_question(conversation_context)
            if follow_up:
                enhanced += f" {follow_up}"
        
        return enhanced
    
    async def _should_add_follow_up(self, conversation_context: ConversationContext) -> bool:
        """Determine if a follow-up question should be added."""
        
        # Don't add follow-up if user message is a thank you or goodbye
        # Handle different ConversationContext schemas
        user_message = ""
        if hasattr(conversation_context, 'current_message'):
            user_message = conversation_context.current_message.lower()
        elif hasattr(conversation_context, 'conversation_topic'):
            user_message = (conversation_context.conversation_topic or "").lower()
        endings = ['thank', 'thanks', 'bye', 'goodbye', 'that\'s all', 'perfect']
        
        if any(ending in user_message for ending in endings):
            return False
        
        # Add follow-up for informational responses
        return len(conversation_context.message_history) < 10
    
    async def _generate_follow_up_question(
        self,
        conversation_context: ConversationContext
    ) -> Optional[str]:
        """Generate an appropriate follow-up question."""
        
        topic_keywords = conversation_context.context_metadata.get('topic', '')
        
        if 'stress' in topic_keywords or 'anxiety' in topic_keywords:
            return "How are you feeling about trying these strategies?"
        elif 'career' in topic_keywords or 'job' in topic_keywords:
            return "What aspect would you like to explore further?"
        elif 'productivity' in topic_keywords:
            return "Which of these approaches resonates most with you?"
        else:
            return "Is there anything specific you'd like me to elaborate on?"
    
    async def _generate_quality_metrics(
        self,
        enhanced_content: str,
        conversation_context: ConversationContext
    ) -> Dict[str, Any]:
        """Generate quality metrics for the response."""
        
        return {
            "content_length": len(enhanced_content),
            "sentence_count": len(re.findall(r'[.!?]+', enhanced_content)),
            "word_count": len(enhanced_content.split()),
            "has_follow_up": "?" in enhanced_content,
            "processing_time": 0.1,  # Placeholder
            "enhancement_applied": True
        }
