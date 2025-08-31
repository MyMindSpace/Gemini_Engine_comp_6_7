# Gemini Engine Testing Guide

## Quick Start Testing

### 1. **Basic Import Test (No API Keys Required)**
```bash
# Set test mode and run basic tests
$env:TEST_MODE="true"
python test_basic.py
```

This verifies that all modules can be imported and initialized without external dependencies.

### 2. **Full Functionality Testing (Requires API Keys)**

#### Step 1: Set up environment variables
```bash
# Copy the example file
copy env.example .env

# Edit .env with your actual API keys
notepad .env
```

#### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Run comprehensive tests
```bash
# Set production mode
$env:TEST_MODE="false"

# Run all tests
python -m pytest tests/ -v
```

## Testing Modes

### **Test Mode (`TEST_MODE=true`)**
- ✅ **What works**: Module imports, class initialization, basic functionality
- ❌ **What doesn't work**: Gemini API calls, database connections, vector operations
- 🎯 **Use case**: Development, CI/CD, basic validation

### **Production Mode (`TEST_MODE=false`)**
- ✅ **What works**: Full functionality, real API calls, database operations
- ❌ **What doesn't work**: Nothing - full system capabilities
- 🎯 **Use case**: Production deployment, end-to-end testing

## Required Services for Full Testing

### 1. **Google Gemini API**
- Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Set `GEMINI_API_KEY` in your `.env` file

### 2. **PostgreSQL Database**
- Install PostgreSQL locally or use cloud service
- Create database named `gemini_engine`
- Update connection details in `.env`

### 3. **Redis Cache**
- Install Redis locally or use cloud service
- Update connection details in `.env`

### 4. **Pinecone Vector Database**
- Sign up at [Pinecone](https://www.pinecone.io/)
- Create index named `gemini_engine`
- Update API key and environment in `.env`

## Testing Scenarios

### **Component 6 Testing**
```python
# Test conversation flow
from component6.conversation_manager import ConversationManager
from shared.schemas import ConversationContext, UserProfile

# Initialize manager
manager = ConversationManager()

# Create test context
context = ConversationContext(
    user_id="test_user",
    conversation_id="test_conv",
    user_message="Hello, how are you?",
    user_profile=UserProfile(user_id="test_user")
)

# Test conversation
response = await manager.process_conversation(context)
print(response.enhanced_response)
```

### **Component 7 Testing**
```python
# Test quality analysis
from component7.quality_analyzer import QualityAnalyzer
from shared.schemas import EnhancedResponse

# Initialize analyzer
analyzer = QualityAnalyzer()

# Analyze response quality
analysis = await analyzer.analyze_response(response)
print(f"Quality Score: {analysis.quality_score}")
```

## Performance Testing

### **Response Time Benchmarks**
```bash
# Test response time
python -m pytest tests/test_performance.py -v

# Expected results:
# - Context assembly: <500ms
# - Total response: <3s
# - Quality analysis: <200ms
```

### **Load Testing**
```bash
# Test concurrent conversations
python tests/test_load.py

# Expected: Handle 100+ concurrent users
```

## Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure you're in the right directory
   cd gemini_engine_comp6_7
   
   # Set Python path
   $env:PYTHONPATH="."
   ```

2. **API Key Errors**
   ```bash
   # Check environment variables
   Get-ChildItem Env: | Where-Object {$_.Name -like "*GEMINI*"}
   
   # Verify .env file
   Get-Content .env
   ```

3. **Database Connection Issues**
   ```bash
   # Test PostgreSQL connection
   psql -h localhost -U postgres -d gemini_engine
   
   # Test Redis connection
   redis-cli ping
   ```

### **Debug Mode**
```bash
# Enable debug logging
$env:LOG_LEVEL="DEBUG"
python test_basic.py
```

## Integration Testing

### **Test with Mock Components**
```python
# Use mock interfaces for isolated testing
from shared.mock_interfaces import MockMemoryRetriever

# Test Component 6 with mock Component 5
retriever = MockMemoryRetriever()
memories = await retriever.retrieve_memories("test_user", "test_query")
```

### **End-to-End Testing**
```python
# Test complete pipeline
python tests/test_integration.py

# This tests the full flow from Component 5 → 6 → 7
```

## Next Steps

1. **Start with basic tests**: `python test_basic.py`
2. **Set up environment**: Copy `env.example` to `.env` and configure
3. **Run full tests**: `python -m pytest tests/ -v`
4. **Test specific components**: Use the examples above
5. **Performance validation**: Run performance benchmarks

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all environment variables are set correctly
3. Ensure all required services are running
4. Check logs for detailed error messages
