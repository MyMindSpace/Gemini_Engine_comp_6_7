"""
Simple test to verify basic functionality
"""

import test_config  # Set up test environment
import asyncio

async def simple_test():
    """Test basic components individually"""
    print("Testing Gemini Engine Components...")
    
    try:
        # Test shared utilities
        print("1. Testing shared utilities...")
        from shared.utils import get_logger, generate_correlation_id
        logger = get_logger("test")
        correlation_id = generate_correlation_id()
        print(f"✓ Utilities working: {correlation_id}")
        
        # Test schemas
        print("2. Testing schemas...")
        from shared.schemas import UserProfile, CommunicationStyle
        test_profile = UserProfile(
            user_id="test_user",
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_response_length="moderate"
        )
        print(f"✓ Schemas working: {test_profile.user_id}")
        
        # Test mock interfaces
        print("3. Testing mock interfaces...")
        from shared.mock_interfaces import MockComponent5Interface
        mock = MockComponent5Interface()
        print("✓ Mock interfaces working")
        
        # Test individual Component 6 modules
        print("4. Testing Component 6 modules...")
        
        from component6.personality_engine import PersonalityEngine
        personality = PersonalityEngine()
        print("✓ Personality Engine loaded")
        
        from component6.proactive_engine import ProactiveEngine
        proactive = ProactiveEngine()
        print("✓ Proactive Engine loaded")
        
        from component6.response_processor import ResponseProcessor
        processor = ResponseProcessor()
        print("✓ Response Processor loaded")
        
        # Test Component 7 modules  
        print("5. Testing Component 7 modules...")
        from component7.quality_analyzer import QualityAnalyzer
        analyzer = QualityAnalyzer()
        print("✓ Quality Analyzer loaded")
        
        print("\n🎉 Basic component test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)
