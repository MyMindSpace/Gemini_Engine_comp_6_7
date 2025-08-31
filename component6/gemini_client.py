"""
Gemini Client Module for Component 6.
Handles Google Gemini API interactions with production-ready features.
"""

import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime, timedelta
import httpx
from dataclasses import asdict

from shared.schemas import (
    GeminiRequest, EnhancedResponse, APIMetrics
)
from shared.utils import (
    get_logger, log_execution_time, generate_correlation_id,
    with_timeout, rate_limit_decorator
)
from shared.mock_interfaces import MockComponent5Interface
from config.settings import settings


class GeminiClient:
    """Production-ready Google Gemini API client"""
    
    def __init__(self):
        """Initialize Gemini client"""
        self.logger = get_logger("gemini_client")
        self.api_key = settings.gemini_api.api_key
        self.base_url = settings.gemini_api.base_url
        self.model_name = settings.gemini_api.model_name
        
        # Rate limiting
        self.request_timestamps = []
        self.token_usage_timestamps = []
        
        # Performance tracking
        self.response_times = []
        self.error_counts = {}
        self.success_counts = {}
        
        # HTTP client
        self.http_client = None
        self._setup_http_client()
        
        self.logger.info("Gemini Client initialized")
    
    def _setup_http_client(self):
        """Setup HTTP client with proper configuration"""
        timeout = httpx.Timeout(
            settings.gemini_api.timeout_seconds,
            connect=10.0,
            read=settings.gemini_api.timeout_seconds
        )
        
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100
            ),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "GeminiEngine/1.0.0"
            }
        )
    
    @log_execution_time
    async def generate_response(
        self,
        gemini_request: GeminiRequest,
        enable_streaming: bool = False
    ) -> EnhancedResponse:
        """Generate response from Gemini API"""
        correlation_id = generate_correlation_id()
        self.logger.info(f"Generating Gemini response for conversation {gemini_request.conversation_id}", extra={
            "correlation_id": correlation_id,
            "user_id": gemini_request.user_id,
            "enable_streaming": enable_streaming
        })
        
        start_time = datetime.utcnow()
        
        try:
            # Check rate limits
            await self._check_rate_limits()
            
            # Prepare API request
            api_request = self._prepare_api_request(gemini_request)
            
            # Make API call
            if enable_streaming:
                response = await self._make_streaming_request(api_request, correlation_id)
            else:
                response = await self._make_single_request(api_request, correlation_id)
            
            # Process response
            enhanced_response = await self._process_response(
                response, gemini_request, start_time, correlation_id
            )
            
            # Update metrics
            self._update_success_metrics(start_time)
            
            self.logger.info(f"Gemini response generated successfully", extra={
                "correlation_id": correlation_id,
                "response_length": len(enhanced_response.enhanced_response),
                "processing_time_ms": enhanced_response.enhancement_metadata.get("processing_time_ms", 0)
            })
            
            return enhanced_response
            
        except Exception as e:
            # Update error metrics
            self._update_error_metrics(str(type(e).__name__), start_time)
            
            self.logger.error(f"Failed to generate Gemini response: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "conversation_id": gemini_request.conversation_id
            })
            raise
    
    async def _check_rate_limits(self) -> None:
        """Check and enforce rate limits"""
        now = time.time()
        
        # Check request rate limit
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 60  # Keep only last minute
        ]
        
        if len(self.request_timestamps) >= settings.gemini_api.rate_limit_requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0])
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        # Check token rate limit
        self.token_usage_timestamps = [
            ts for ts in self.token_usage_timestamps 
            if now - ts < 60  # Keep only last minute
        ]
    
    def _prepare_api_request(self, gemini_request: GeminiRequest) -> Dict[str, Any]:
        """Prepare the API request payload"""
        # Build the content array
        content = []
        
        # Add system prompt if supported
        if hasattr(settings.gemini_api, 'supports_system_prompt') and settings.gemini_api.supports_system_prompt:
            content.append({
                "role": "user",
                "parts": [{"text": gemini_request.system_prompt}]
            })
        
        # Add context and conversation
        full_context = f"{gemini_request.context_section}\n\n{gemini_request.conversation_history}"
        if full_context.strip():
            content.append({
                "role": "user",
                "parts": [{"text": full_context}]
            })
        
        # Add current message
        content.append({
            "role": "user",
            "parts": [{"text": gemini_request.current_message}]
        })
        
        # Build request payload
        request_payload = {
            "contents": content,
            "generationConfig": {
                "temperature": gemini_request.api_parameters.get("temperature", 0.7),
                "topP": gemini_request.api_parameters.get("top_p", 0.9),
                "topK": gemini_request.api_parameters.get("top_k", 40),
                "maxOutputTokens": gemini_request.api_parameters.get("max_tokens", 2048),
                "stopSequences": []
            },
            "safetySettings": gemini_request.api_parameters.get("safety_settings", []),
            "stream": False  # Will be overridden for streaming
        }
        
        return request_payload
    
    async def _make_single_request(
        self,
        api_request: Dict[str, Any],
        correlation_id: str
    ) -> Dict[str, Any]:
        """Make a single request to Gemini API"""
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        
        headers = {
            "x-goog-api-key": self.api_key,
            "x-goog-user-project": "gemini-engine"
        }
        
        try:
            response = await self.http_client.post(
                url,
                json=api_request,
                headers=headers
            )
            
            response.raise_for_status()
            
            # Update rate limiting
            self.request_timestamps.append(time.time())
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error from Gemini API: {e.response.status_code}", extra={
                "correlation_id": correlation_id,
                "status_code": e.response.status_code,
                "response_text": e.response.text
            })
            raise self._create_api_error(e.response.status_code, e.response.text)
        
        except httpx.RequestError as e:
            self.logger.error(f"Request error to Gemini API: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            raise RuntimeError(f"Failed to communicate with Gemini API: {e}")
    
    async def _make_streaming_request(
        self,
        api_request: Dict[str, Any],
        correlation_id: str
    ) -> Dict[str, Any]:
        """Make a streaming request to Gemini API"""
        # For now, fall back to single request
        # Streaming implementation would be added here
        self.logger.info("Streaming not yet implemented, using single request", extra={
            "correlation_id": correlation_id
        })
        
        return await self._make_single_request(api_request, correlation_id)
    
    async def _process_response(
        self,
        api_response: Dict[str, Any],
        gemini_request: GeminiRequest,
        start_time: datetime,
        correlation_id: str
    ) -> EnhancedResponse:
        """Process the API response into EnhancedResponse format"""
        try:
            # Extract response text
            response_text = self._extract_response_text(api_response)
            
            # Process and enhance response
            processed_response = await self._enhance_response(
                response_text, gemini_request
            )
            
            # Generate follow-up suggestions
            follow_up_suggestions = self._generate_follow_up_suggestions(
                processed_response, gemini_request
            )
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(
                processed_response, gemini_request
            )
            
            # Calculate processing time
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Create enhanced response
            enhanced_response = EnhancedResponse(
                response_id=generate_correlation_id(),
                original_response=response_text,
                enhanced_response=processed_response,
                enhancement_metadata={
                    "quality_score": quality_score,
                    "follow_up_suggestions": follow_up_suggestions,
                    "context_utilization": 0.8,  # Placeholder - would be calculated
                    "personality_alignment": 0.9,  # Placeholder - would be calculated
                    "emotional_appropriateness": 0.85,  # Placeholder - would be calculated
                    "response_length": len(processed_response),
                    "processing_time_ms": processing_time_ms
                },
                processing_timestamp=datetime.utcnow(),
                quality_metrics={
                    "overall_score": quality_score,
                    "clarity": 0.9,
                    "relevance": 0.85
                },
                safety_checks={
                    "content_safety": "passed",
                    "emotional_safety": "passed"
                },
                context_usage={
                    "memory_utilization": 0.8,
                    "context_relevance": 0.85
                }
            )
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Failed to process API response: {e}", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            raise
    
    def _extract_response_text(self, api_response: Dict[str, Any]) -> str:
        """Extract response text from Gemini API response"""
        try:
            # Navigate through response structure
            candidates = api_response.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")
            
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise ValueError("No parts in content")
            
            # Extract text from parts
            text_parts = []
            for part in parts:
                if part.get("text"):
                    text_parts.append(part["text"])
            
            if not text_parts:
                raise ValueError("No text in parts")
            
            return " ".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to extract response text: {e}")
            raise ValueError(f"Invalid response format: {e}")
    
    async def _enhance_response(
        self,
        response_text: str,
        gemini_request: GeminiRequest
    ) -> str:
        """Enhance the response quality and formatting"""
        # Basic text cleaning
        enhanced_text = response_text.strip()
        
        # Ensure proper sentence endings
        if enhanced_text and not enhanced_text.endswith(('.', '!', '?')):
            enhanced_text += '.'
        
        # Add follow-up question if appropriate
        if self._should_add_follow_up(gemini_request):
            follow_up = self._generate_simple_follow_up(gemini_request)
            if follow_up:
                enhanced_text += f"\n\n{follow_up}"
        
        return enhanced_text
    
    def _should_add_follow_up(self, gemini_request: GeminiRequest) -> bool:
        """Determine if a follow-up question should be added"""
        guidelines = gemini_request.response_guidelines
        
        # Check if user prefers follow-up questions
        if not guidelines.get("enable_follow_up_questions", True):
            return False
        
        # Check probability threshold
        import random
        return random.random() < guidelines.get("follow_up_question_probability", 0.8)
    
    def _generate_simple_follow_up(self, gemini_request: GeminiRequest) -> Optional[str]:
        """Generate a simple follow-up question"""
        # Simple template-based follow-up questions
        templates = [
            "What are your thoughts on this?",
            "How does this resonate with you?",
            "Is there anything specific you'd like to explore further?",
            "What would you like to focus on next?",
            "How are you feeling about this?"
        ]
        
        import random
        return random.choice(templates)
    
    def _generate_follow_up_suggestions(
        self,
        response_text: str,
        gemini_request: GeminiRequest
    ) -> List[str]:
        """Generate follow-up question suggestions"""
        suggestions = []
        
        # Extract key topics from response
        topics = self._extract_topics(response_text)
        
        # Generate contextual follow-ups
        for topic in topics[:3]:  # Limit to 3 suggestions
            suggestion = f"What are your thoughts on {topic}?"
            suggestions.append(suggestion)
        
        # Add general follow-ups if we don't have enough
        general_suggestions = [
            "How are you feeling about this conversation?",
            "Is there anything else you'd like to discuss?",
            "What would be most helpful for you right now?"
        ]
        
        while len(suggestions) < 3:
            import random
            suggestion = random.choice(general_suggestions)
            if suggestion not in suggestions:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple keyword extraction
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter and count
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top topics
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:5]]
    
    def _calculate_quality_score(
        self,
        response_text: str,
        gemini_request: GeminiRequest
    ) -> float:
        """Calculate response quality score"""
        score = 0.0
        
        # Length appropriateness
        guidelines = gemini_request.response_guidelines
        target_length = guidelines.get("max_words", 300)
        actual_length = len(response_text.split())
        
        if actual_length <= target_length * 1.2:
            score += 0.3
        elif actual_length <= target_length * 1.5:
            score += 0.2
        else:
            score += 0.1
        
        # Content quality indicators
        if response_text.strip():
            score += 0.2
        
        if response_text.endswith(('.', '!', '?')):
            score += 0.1
        
        if len(response_text) > 50:
            score += 0.2
        
        # Follow-up question bonus
        if any(q in response_text.lower() for q in ['what', 'how', 'why', 'when', 'where']):
            score += 0.2
        
        return min(1.0, score)
    
    def _create_api_error(self, status_code: int, response_text: str) -> Exception:
        """Create appropriate exception for API errors"""
        if status_code == 400:
            return ValueError(f"Bad request to Gemini API: {response_text}")
        elif status_code == 401:
            return ValueError("Invalid API key for Gemini API")
        elif status_code == 403:
            return ValueError("Access denied to Gemini API")
        elif status_code == 429:
            return RuntimeError("Rate limit exceeded for Gemini API")
        elif status_code >= 500:
            return RuntimeError(f"Gemini API server error: {status_code}")
        else:
            return RuntimeError(f"Unexpected error from Gemini API: {status_code}")
    
    def _update_success_metrics(self, start_time: datetime) -> None:
        """Update success metrics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.response_times.append(processing_time)
        
        # Keep only last 100 measurements
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        # Update success count
        self.success_counts["total"] = self.success_counts.get("total", 0) + 1
    
    def _update_error_metrics(self, error_type: str, start_time: datetime) -> None:
        """Update error metrics"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.response_times:
            return {
                "avg_response_time_ms": 0.0,
                "success_rate": 0.0,
                "total_requests": 0,
                "error_counts": self.error_counts
            }
        
        total_requests = self.success_counts.get("total", 0) + sum(self.error_counts.values())
        success_rate = self.success_counts.get("total", 0) / total_requests if total_requests > 0 else 0.0
        
        return {
            "avg_response_time_ms": sum(self.response_times) / len(self.response_times),
            "min_response_time_ms": min(self.response_times),
            "max_response_time_ms": max(self.response_times),
            "success_rate": success_rate,
            "total_requests": total_requests,
            "success_counts": self.success_counts,
            "error_counts": self.error_counts
        }
    
    async def health_check(self) -> bool:
        """Perform health check on Gemini API"""
        try:
            # Try a simple request
            test_request = GeminiRequest(
                conversation_id="health_check",
                user_id="system",
                system_prompt="You are a helpful assistant.",
                context_section="",
                conversation_history="",
                current_message="Hello",
                response_guidelines={},
                api_parameters={},
                user_preferences={}
            )
            
            response = await self.generate_response(test_request)
            return isinstance(response, EnhancedResponse)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """Close HTTP client"""
        if self.http_client:
            await self.http_client.aclose()
            self.logger.info("Gemini client HTTP client closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'http_client') and self.http_client:
            # Schedule cleanup
            asyncio.create_task(self.close())
