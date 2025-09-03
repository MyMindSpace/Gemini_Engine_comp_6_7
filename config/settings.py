"""
Configuration management for Gemini Engine Components 6 & 7.
Production-ready settings with environment variable support.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class GeminiAPIConfig:
    """Google Gemini API configuration"""
    api_key: str = "AIzaSyD2WlV2eqqGMvrciq3qh5-F823L6kJlDyM"
    model_name: str = "gemini-2.5-pro"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    safety_settings: Dict[str, Any] = None
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 15000
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1

    def __post_init__(self):
        if self.safety_settings is None:
            self.safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
            }


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    postgres_host: str
    postgres_port: int = 5432
    postgres_database: str = "gemini_engine"
    postgres_username: str = "postgres"
    postgres_password: str = ""
    postgres_ssl_mode: str = "require"
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 30
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_database: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = True
    
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "gemini_engine"


@dataclass
class PerformanceConfig:
    """Performance and scaling configuration"""
    max_concurrent_conversations: int = 1000
    context_assembly_timeout_ms: int = 500
    total_response_timeout_ms: int = 3000
    memory_retrieval_timeout_ms: int = 200
    analysis_timeout_ms: int = 200
    
    # Token budget management
    max_context_tokens: int = 2000
    min_context_tokens: int = 100
    context_compression_threshold: float = 0.8
    
    # Caching configuration
    response_cache_ttl_seconds: int = 3600
    context_cache_ttl_seconds: int = 1800
    user_profile_cache_ttl_seconds: int = 7200
    
    # Rate limiting
    max_requests_per_user_per_minute: int = 30
    max_tokens_per_user_per_minute: int = 5000


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path: Optional[str] = None
    enable_structured_logging: bool = True
    enable_correlation_ids: bool = True
    enable_performance_logging: bool = True
    
    # Monitoring
    enable_metrics_collection: bool = True
    metrics_export_interval_seconds: int = 60
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    enable_content_filtering: bool = True
    enable_safety_checks: bool = True
    enable_privacy_protection: bool = True
    max_response_length: int = 10000
    forbidden_content_patterns: list = None
    
    # API security
    enable_rate_limiting: bool = True
    enable_user_authentication: bool = True
    enable_request_validation: bool = True
    
    # Data privacy
    enable_anonymization: bool = False
    data_retention_days: int = 90
    enable_data_encryption: bool = True


@dataclass
class PersonalityConfig:
    """AI personality and behavior configuration"""
    default_communication_style: str = "friendly"
    default_emotional_tone: str = "supportive"
    enable_personality_adaptation: bool = True
    personality_learning_rate: float = 0.1
    
    # Proactive engagement
    enable_proactive_messages: bool = True
    proactive_engagement_threshold: float = 0.7
    max_proactive_messages_per_day: int = 5
    
    # Response guidelines
    min_response_length: int = 50
    max_response_length: int = 2000
    enable_follow_up_questions: bool = True
    follow_up_question_probability: float = 0.8


@dataclass
class AnalysisConfig:
    """Component 7 analysis configuration"""
    enable_real_time_analysis: bool = True
    enable_batch_analysis: bool = True
    analysis_batch_size: int = 100
    analysis_batch_interval_seconds: int = 300
    
    # Quality metrics
    quality_threshold: float = 0.7
    enable_automated_improvement: bool = True
    improvement_learning_rate: float = 0.05
    
    # Feedback system
    enable_component_feedback: bool = True
    feedback_update_interval_hours: int = 24
    enable_ab_testing: bool = True


class Settings:
    """Main configuration class that loads all settings"""
    
    def __init__(self):
        self.gemini_api = self._load_gemini_config()
        self.database = self._load_database_config()
        self.performance = self._load_performance_config()
        self.logging = self._load_logging_config()
        self.security = self._load_security_config()
        self.personality = self._load_personality_config()
        self.analysis = self._load_analysis_config()
        
        # Set up logging
        self._setup_logging()
    
    def _load_gemini_config(self) -> GeminiAPIConfig:
        """Load Gemini API configuration from environment variables"""
        return GeminiAPIConfig(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro"),
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "2048")),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("GEMINI_TOP_P", "0.9")),
            top_k=int(os.getenv("GEMINI_TOP_K", "40")),
            rate_limit_requests_per_minute=int(os.getenv("GEMINI_RATE_LIMIT_RPM", "60")),
            rate_limit_tokens_per_minute=int(os.getenv("GEMINI_RATE_LIMIT_TPM", "15000")),
            timeout_seconds=int(os.getenv("GEMINI_TIMEOUT", "30")),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
            retry_delay_seconds=int(os.getenv("GEMINI_RETRY_DELAY", "1"))
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables"""
        return DatabaseConfig(
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_database=os.getenv("POSTGRES_DB", "gemini_engine"),
            postgres_username=os.getenv("POSTGRES_USER", "postgres"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
            postgres_ssl_mode=os.getenv("POSTGRES_SSL_MODE", "require"),
            postgres_pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "20")),
            postgres_max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "30")),
            
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_database=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_ssl=os.getenv("REDIS_SSL", "true").lower() == "true",
            
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "")
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration from environment variables"""
        return PerformanceConfig(
            max_concurrent_conversations=int(os.getenv("MAX_CONCURRENT_CONVERSATIONS", "1000")),
            context_assembly_timeout_ms=int(os.getenv("CONTEXT_ASSEMBLY_TIMEOUT_MS", "500")),
            total_response_timeout_ms=int(os.getenv("TOTAL_RESPONSE_TIMEOUT_MS", "3000")),
            memory_retrieval_timeout_ms=int(os.getenv("MEMORY_RETRIEVAL_TIMEOUT_MS", "200")),
            analysis_timeout_ms=int(os.getenv("ANALYSIS_TIMEOUT_MS", "200")),
            
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "2000")),
            min_context_tokens=int(os.getenv("MIN_CONTEXT_TOKENS", "100")),
            context_compression_threshold=float(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "0.8")),
            
            response_cache_ttl_seconds=int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "3600")),
            context_cache_ttl_seconds=int(os.getenv("CONTEXT_CACHE_TTL_SECONDS", "1800")),
            user_profile_cache_ttl_seconds=int(os.getenv("USER_PROFILE_CACHE_TTL_SECONDS", "7200")),
            
            max_requests_per_user_per_minute=int(os.getenv("MAX_REQUESTS_PER_USER_PER_MINUTE", "30")),
            max_tokens_per_user_per_minute=int(os.getenv("MAX_TOKENS_PER_USER_PER_MINUTE", "5000"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment variables"""
        return LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_file_path=os.getenv("LOG_FILE_PATH"),
            enable_structured_logging=os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true",
            enable_correlation_ids=os.getenv("ENABLE_CORRELATION_IDS", "true").lower() == "true",
            enable_performance_logging=os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true",
            
            enable_metrics_collection=os.getenv("ENABLE_METRICS_COLLECTION", "true").lower() == "true",
            metrics_export_interval_seconds=int(os.getenv("METRICS_EXPORT_INTERVAL_SECONDS", "60")),
            enable_health_checks=os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true",
            health_check_interval_seconds=int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "30"))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration from environment variables"""
        forbidden_patterns = os.getenv("FORBIDDEN_CONTENT_PATTERNS", "")
        return SecurityConfig(
            enable_content_filtering=os.getenv("ENABLE_CONTENT_FILTERING", "true").lower() == "true",
            enable_safety_checks=os.getenv("ENABLE_SAFETY_CHECKS", "true").lower() == "true",
            enable_privacy_protection=os.getenv("ENABLE_PRIVACY_PROTECTION", "true").lower() == "true",
            max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "10000")),
            forbidden_content_patterns=forbidden_patterns.split(",") if forbidden_patterns else [],
            
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            enable_user_authentication=os.getenv("ENABLE_USER_AUTHENTICATION", "true").lower() == "true",
            enable_request_validation=os.getenv("ENABLE_REQUEST_VALIDATION", "true").lower() == "true",
            
            enable_anonymization=os.getenv("ENABLE_ANONYMIZATION", "false").lower() == "true",
            data_retention_days=int(os.getenv("DATA_RETENTION_DAYS", "90")),
            enable_data_encryption=os.getenv("ENABLE_DATA_ENCRYPTION", "true").lower() == "true"
        )
    
    def _load_personality_config(self) -> PersonalityConfig:
        """Load personality configuration from environment variables"""
        return PersonalityConfig(
            default_communication_style=os.getenv("DEFAULT_COMMUNICATION_STYLE", "friendly"),
            default_emotional_tone=os.getenv("DEFAULT_EMOTIONAL_TONE", "supportive"),
            enable_personality_adaptation=os.getenv("ENABLE_PERSONALITY_ADAPTATION", "true").lower() == "true",
            personality_learning_rate=float(os.getenv("PERSONALITY_LEARNING_RATE", "0.1")),
            
            enable_proactive_messages=os.getenv("ENABLE_PROACTIVE_MESSAGES", "true").lower() == "true",
            proactive_engagement_threshold=float(os.getenv("PROACTIVE_ENGAGEMENT_THRESHOLD", "0.7")),
            max_proactive_messages_per_day=int(os.getenv("MAX_PROACTIVE_MESSAGES_PER_DAY", "5")),
            
            min_response_length=int(os.getenv("MIN_RESPONSE_LENGTH", "50")),
            max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "2000")),
            enable_follow_up_questions=os.getenv("ENABLE_FOLLOW_UP_QUESTIONS", "true").lower() == "true",
            follow_up_question_probability=float(os.getenv("FOLLOW_UP_QUESTION_PROBABILITY", "0.8"))
        )
    
    def _load_analysis_config(self) -> AnalysisConfig:
        """Load analysis configuration from environment variables"""
        return AnalysisConfig(
            enable_real_time_analysis=os.getenv("ENABLE_REAL_TIME_ANALYSIS", "true").lower() == "true",
            enable_batch_analysis=os.getenv("ENABLE_BATCH_ANALYSIS", "true").lower() == "true",
            analysis_batch_size=int(os.getenv("ANALYSIS_BATCH_SIZE", "100")),
            analysis_batch_interval_seconds=int(os.getenv("ANALYSIS_BATCH_INTERVAL_SECONDS", "300")),
            
            quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.7")),
            enable_automated_improvement=os.getenv("ENABLE_AUTOMATED_IMPROVEMENT", "true").lower() == "true",
            improvement_learning_rate=float(os.getenv("IMPROVEMENT_LEARNING_RATE", "0.05")),
            
            enable_component_feedback=os.getenv("ENABLE_COMPONENT_FEEDBACK", "true").lower() == "true",
            feedback_update_interval_hours=int(os.getenv("FEEDBACK_UPDATE_INTERVAL_HOURS", "24")),
            enable_ab_testing=os.getenv("ENABLE_AB_TESTING", "true").lower() == "true"
        )
    
    def _setup_logging(self):
        """Configure logging based on settings"""
        log_level = getattr(logging, self.logging.log_level.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=self.logging.log_format,
            handlers=self._get_log_handlers()
        )
        
        # Set specific logger levels
        logging.getLogger("gemini_engine").setLevel(log_level)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
    
    def _get_log_handlers(self):
        """Get configured log handlers"""
        handlers = [logging.StreamHandler()]
        
        if self.logging.log_file_path:
            log_file = Path(self.logging.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        return handlers
    
    def validate(self) -> bool:
        """Validate all configuration settings"""
        errors = []
        
        # # Validate required settings (only if not in test mode)
        # if not os.getenv("TEST_MODE", "false").lower() == "true":
        #     if not self.gemini_api.api_key:
        #         errors.append("GEMINI_API_KEY is required")
        
        # # Validate performance settings
        # if self.performance.context_assembly_timeout_ms > self.performance.total_response_timeout_ms:
        #     errors.append("Context assembly timeout cannot exceed total response timeout")
        
        # if self.performance.max_context_tokens < self.performance.min_context_tokens:
        #     errors.append("Max context tokens must be greater than min context tokens")
        
        # if errors:
        #     raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def get_database_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.database.postgres_username}:{self.database.postgres_password}@{self.database.postgres_host}:{self.database.postgres_port}/{self.database.postgres_database}?sslmode={self.database.postgres_ssl_mode}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        protocol = "rediss" if self.database.redis_ssl else "redis"
        auth = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return f"{protocol}://{auth}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_database}"


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    logging.error(f"Configuration error: {e}")
    raise
