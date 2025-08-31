"""
Basic test file to verify all modules can be imported and initialized.
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_component6_imports():
    """Test Component 6 imports"""
    print("Testing Component 6 imports...")
    
    try:
        from component6 import (
            ConversationManager,
            MemoryRetriever,
            ContextAssembler,
            GeminiClient,
            PersonalityEngine,
            ResponseProcessor,
            ProactiveEngine
        )
        print("✓ All Component 6 modules imported successfully")
        
        # Test initialization
        memory_retriever = MemoryRetriever()
        context_assembler = ContextAssembler()
        personality_engine = PersonalityEngine()
        response_processor = ResponseProcessor()
        proactive_engine = ProactiveEngine()
        print("✓ All Component 6 classes initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Component 6 import/initialization failed: {e}")
        return False

async def test_component7_imports():
    """Test Component 7 imports"""
    print("\nTesting Component 7 imports...")
    
    try:
        from component7 import (
            QualityAnalyzer,
            SatisfactionTracker,
            FeedbackEngine,
            MetricsCollector
        )
        print("✓ All Component 7 modules imported successfully")
        
        # Test initialization
        quality_analyzer = QualityAnalyzer()
        satisfaction_tracker = SatisfactionTracker()
        feedback_engine = FeedbackEngine()
        metrics_collector = MetricsCollector()
        print("✓ All Component 7 classes initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Component 7 import/initialization failed: {e}")
        return False

async def test_shared_imports():
    """Test shared module imports"""
    print("\nTesting shared module imports...")
    
    try:
        from shared.schemas import (
            MemoryContext,
            ConversationContext,
            EnhancedResponse,
            ResponseAnalysis
        )
        print("✓ All shared schemas imported successfully")
        
        from shared.utils import (
            get_logger,
            log_execution_time,
            generate_correlation_id
        )
        print("✓ All shared utilities imported successfully")
        
        from shared.mock_interfaces import MockComponent5Interface
        print("✓ Mock interfaces imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Shared module import failed: {e}")
        return False

async def test_config_imports():
    """Test configuration imports"""
    print("\nTesting configuration imports...")
    
    try:
        from config.settings import settings
        print("✓ Settings imported successfully")
        
        # Test that settings has expected attributes
        assert hasattr(settings, 'gemini_api')
        assert hasattr(settings, 'database')
        print("✓ Settings structure validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration import failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Starting basic import and initialization tests...\n")
    
    tests = [
        test_shared_imports(),
        test_config_imports(),
        test_component6_imports(),
        test_component7_imports()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Test {i+1} failed with exception: {result}")
        elif result is True:
            passed += 1
        else:
            print(f"Test {i+1} returned: {result}")
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The Gemini Engine is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
