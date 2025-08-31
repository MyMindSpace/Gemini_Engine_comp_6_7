"""
Basic integration test for Gemini Engine Components 6 & 7
"""

import asyncio
import logging
from datetime import datetime

# Import test configuration first to set environment variables
import test_config

async def test_basic_integration():
    """Test basic integration of Components 6 & 7"""
    print("Testing Gemini Engine Components 6 & 7 Integration...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from gemini_engine_orchestrator import GeminiEngineOrchestrator
        from shared.mock_interfaces import MockComponent5Interface
        print("✓ All imports successful")
        
        # Initialize orchestrator
        print("2. Initializing orchestrator...")
        mock_component5 = MockComponent5Interface()
        orchestrator = GeminiEngineOrchestrator(mock_component5)
        print("✓ Orchestrator initialized")
        
        # Test health check
        print("3. Running health check...")
        health_status = await orchestrator.health_check()
        print(f"✓ Health check completed: {health_status.get('overall_healthy', False)}")
        
        # Test basic conversation
        print("4. Testing basic conversation...")
        result = await orchestrator.process_conversation(
            user_id="test_user_001",
            user_message="Hello, how are you today?",
            conversation_id="test_conversation_001"
        )
        
        if "error" in result:
            print(f"⚠ Conversation processing had issues: {result['error']}")
        else:
            print(f"✓ Conversation processed successfully")
            print(f"  - Response: {result['response'][:100]}...")
            print(f"  - Quality Score: {result['quality_analysis']['overall_quality']:.2f}")
            print(f"  - Processing Time: {result['processing_time_ms']:.2f}ms")
        
        # Test performance metrics
        print("5. Testing performance metrics...")
        metrics = orchestrator.get_performance_summary()
        print("✓ Performance metrics retrieved")
        print(f"  - Total Conversations: {metrics['orchestrator_metrics']['total_conversations']}")
        print(f"  - Success Rate: {metrics['orchestrator_metrics']['success_rate']:.2f}")
        
        # Test proactive message generation
        print("6. Testing proactive message generation...")
        proactive_msg = await orchestrator.generate_proactive_message(
            user_id="test_user_001",
            trigger_event="user_inactive",
            context={"user_name": "Test User"}
        )
        
        if proactive_msg:
            print("✓ Proactive message generated")
            print(f"  - Content: {proactive_msg['content']}")
        else:
            print("⚠ No proactive message generated")
        
        # Cleanup
        print("7. Cleaning up...")
        await orchestrator.cleanup()
        print("✓ Cleanup completed")
        
        print("\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_basic_integration())
    exit(0 if success else 1)
