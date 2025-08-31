# Gemini Engine - Components 6 & 7

A production-ready AI conversational system that integrates Google Gemini API with intelligent memory management, personality adaptation, and response analysis.

## 🚀 Quick Start

### 1. **Basic Testing (No API Keys Required)**
```bash
# Test basic functionality
$env:TEST_MODE="true"
python test_basic.py

# Run interactive demo
python demo.py

# Test conversation flow
python demo_conversation.py
```

### 2. **Full Functionality Testing (Requires API Keys)**
```bash
# Copy environment template
copy env.example .env

# Edit .env with your API keys
notepad .env

# Install dependencies
pip install -r requirements.txt

# Run comprehensive tests
$env:TEST_MODE="false"
python -m pytest tests/ -v
```

## 🏗️ Architecture Overview

### **Component 6: Gemini Brain Integration**
- **Memory Retriever**: Interfaces with Component 5 LSTM Memory Gates
- **Context Assembler**: Advanced prompt engineering & token management
- **Personality Engine**: Adaptive communication style adaptation
- **Gemini Client**: Google Gemini API integration with streaming
- **Response Processor**: Quality validation and enhancement
- **Proactive Engine**: AI-initiated conversations
- **Conversation Manager**: Main orchestrator

### **Component 7: Response Analysis**
- **Quality Analyzer**: Automated response quality scoring
- **Satisfaction Tracker**: User engagement monitoring
- **Feedback Engine**: Component feedback system
- **Metrics Collector**: Performance monitoring

## 📁 Project Structure

```
gemini_engine_comp6_7/
├── component6/                 # Gemini Brain Integration
│   ├── memory_retriever.py    # Component 5 interface
│   ├── context_assembler.py   # Prompt engineering
│   ├── personality_engine.py  # Communication adaptation
│   ├── gemini_client.py       # Gemini API client
│   ├── response_processor.py  # Response enhancement
│   ├── proactive_engine.py    # AI-initiated conversations
│   └── conversation_manager.py # Main orchestrator
├── component7/                 # Response Analysis
│   ├── quality_analyzer.py    # Quality assessment
│   ├── satisfaction_tracker.py # User engagement
│   ├── feedback_engine.py     # Feedback system
│   └── metrics_collector.py   # Performance metrics
├── shared/                     # Shared components
│   ├── schemas.py             # Data models
│   ├── mock_interfaces.py     # Mock Component 5
│   └── utils.py               # Utilities
├── config/                     # Configuration
│   ├── settings.py            # Environment settings
│   └── prompts.yaml           # Prompt templates
├── tests/                      # Test suite
├── demo.py                     # Basic functionality demo
├── demo_conversation.py        # Conversation flow demo
├── env.example                 # Environment template
├── TESTING_GUIDE.md           # Testing instructions
└── requirements.txt            # Dependencies
```

## 🔧 Configuration

### **Environment Variables**

Create a `.env` file based on `env.example`:

```bash
# Gemini API
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-pro

# Database
POSTGRES_HOST=localhost
POSTGRES_DATABASE=gemini_engine
REDIS_HOST=localhost

# Vector Database
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# Test Mode
TEST_MODE=false
```

### **Required Services**

1. **Google Gemini API**: Get key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **PostgreSQL**: For structured memory metadata
3. **Redis**: For caching and session management
4. **Pinecone**: For vector similarity search

## 💻 Usage Examples

### **Basic Conversation Flow**

```python
from component6.conversation_manager import ConversationManager
from shared.schemas import ConversationContext, UserProfile, CommunicationStyle

# Initialize manager
manager = ConversationManager()

# Create conversation context
context = ConversationContext(
    user_id="user_123",
    conversation_id="conv_001",
    personality_profile=UserProfile(
        user_id="user_123",
        communication_style=CommunicationStyle.FRIENDLY,
        preferred_response_length="moderate"
    )
)

# Process conversation
response = await manager.process_conversation(context)
print(response.enhanced_response)
```

### **Memory Integration**

```python
from shared.mock_interfaces import MockComponent5Interface

# Mock Component 5 interface
component5 = MockComponent5Interface()

# Get memory context
memories = await component5.get_memory_context(
    user_id="user_123",
    current_message="I'm feeling stressed about work",
    conversation_id="conv_001"
)

print(f"Found {len(memories.selected_memories)} relevant memories")
```

### **Quality Analysis**

```python
from component7.quality_analyzer import QualityAnalyzer
from shared.schemas import EnhancedResponse

# Initialize analyzer
analyzer = QualityAnalyzer()

# Analyze response quality
analysis = await analyzer.analyze_response(response)
print(f"Quality Score: {analysis.quality_scores['overall']}")
```

## 🧪 Testing

### **Test Modes**

- **Test Mode (`TEST_MODE=true`)**: No external dependencies, basic validation
- **Production Mode (`TEST_MODE=false`)**: Full functionality with real APIs

### **Running Tests**

```bash
# Basic tests
python test_basic.py

# All tests
python -m pytest tests/ -v

# Specific component
python -m pytest tests/test_component6.py -v

# Performance tests
python -m pytest tests/test_performance.py -v
```

### **Test Coverage**

- ✅ **Component 6**: All modules implemented and tested
- ✅ **Component 7**: All modules implemented and tested
- ✅ **Shared Schemas**: Complete data model validation
- ✅ **Mock Interfaces**: Component 5 simulation
- ✅ **Configuration**: Environment-based settings
- ✅ **Error Handling**: Comprehensive exception management

## 📊 Performance Requirements

### **Component 6**
- Context assembly: <500ms
- Total response time: <3s
- Token budget: 2000 tokens
- Memory retrieval: <200ms

### **Component 7**
- Response quality analysis: <200ms
- User satisfaction tracking: Real-time
- Feedback generation: <100ms
- Metrics collection: <50ms

## 🔒 Security & Safety

- **Content Filtering**: Gemini safety settings integration
- **Rate Limiting**: API quota management
- **Error Handling**: Graceful degradation
- **Logging**: Structured logging with correlation IDs
- **Environment Variables**: Secure configuration management

## 🚀 Production Deployment

### **Prerequisites**
1. All environment variables configured
2. Database services running
3. API keys validated
4. Performance benchmarks passed

### **Deployment Steps**
```bash
# Install dependencies
pip install -r requirements.txt

# Set production mode
$env:TEST_MODE="false"

# Run health checks
python -m pytest tests/test_integration.py -v

# Start application
python your_app.py
```

### **Monitoring**
- Response time metrics
- Error rate tracking
- Memory usage monitoring
- API quota utilization
- User satisfaction scores

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Set Python path
   $env:PYTHONPATH="."
   ```

2. **API Key Errors**
   ```bash
   # Check environment variables
   Get-ChildItem Env: | Where-Object {$_.Name -like "*GEMINI*"}
   ```

3. **Database Connection Issues**
   ```bash
   # Test connections
   psql -h localhost -U postgres -d gemini_engine
   redis-cli ping
   ```

### **Debug Mode**
```bash
# Enable debug logging
$env:LOG_LEVEL="DEBUG"
python demo.py
```

## 📚 Documentation

- **TESTING_GUIDE.md**: Comprehensive testing instructions
- **env.example**: Environment configuration template
- **Code Comments**: Inline documentation throughout
- **Type Hints**: Comprehensive type annotations

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure all tests pass
5. Follow production code standards

## 📄 License

This project is part of the Gemini Engine implementation for advanced AI journal platforms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the testing guide
3. Examine error logs
4. Verify configuration settings

---

**🎉 The Gemini Engine is production-ready and ready for deployment!**
