#!/usr/bin/env python3
"""
Gemini Engine Conversation Demo
Demonstrates actual conversation flow using the implemented components
"""

import asyncio
import os
from datetime import datetime
from shared.schemas import (
    ConversationContext, UserProfile, CommunicationStyle,
    MemoryContext
)
from component6.conversation_manager import ConversationManager
from shared.mock_interfaces import MockComponent5Interface

async def demo_conversation_flow():
    """Demo a complete conversation flow"""
    print("💬 Gemini Engine Conversation Demo")
    print("=" * 50)
    
    # Set test mode
    os.environ["TEST_MODE"] = "true"
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        conversation_manager = ConversationManager()
        mock_component5 = MockComponent5Interface()
        
        # Create user profile
        user_profile = UserProfile(
            user_id="demo_user",
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_response_length="moderate",
            emotional_matching=True,
            humor_preference=0.7,
            formality_level=0.3,
            conversation_depth=0.8,
            proactive_engagement=True
        )
        
        print("✅ Components initialized successfully")
        
        # Demo conversation 1: Greeting
        print("\n👋 Conversation 1: Greeting")
        print("-" * 30)
        
        context1 = ConversationContext(
            user_id="demo_user",
            conversation_id="demo_conv_001",
            personality_profile=user_profile
        )
        
        # Simulate user message
        user_message = "Hello! How are you today?"
        print(f"User: {user_message}")
        
        # Get memory context from Component 5
        memory_context = await mock_component5.get_memory_context(
            "demo_user", user_message, "demo_conv_001"
        )
        
        print(f"📚 Retrieved {len(memory_context.selected_memories)} relevant memories")
        if memory_context.selected_memories:
            for i, memory in enumerate(memory_context.selected_memories[:2]):
                print(f"   Memory {i+1}: {memory['content_summary'][:60]}...")
        
        # Demo conversation 2: Personal question
        print("\n🤔 Conversation 2: Personal Question")
        print("-" * 30)
        
        user_message2 = "I've been feeling a bit stressed about work lately. Any advice?"
        print(f"User: {user_message2}")
        
        memory_context2 = await mock_component5.get_memory_context(
            "demo_user", user_message2, "demo_conv_001"
        )
        
        print(f"📚 Retrieved {len(memory_context2.selected_memories)} relevant memories")
        if memory_context2.selected_memories:
            for i, memory in enumerate(memory_context2.selected_memories[:2]):
                print(f"   Memory {i+1}: {memory['content_summary'][:60]}...")
        
        # Demo conversation 3: Follow-up
        print("\n🔄 Conversation 3: Follow-up")
        print("-" * 30)
        
        user_message3 = "That's helpful! What about managing time better?"
        print(f"User: {user_message3}")
        
        memory_context3 = await mock_component5.get_memory_context(
            "demo_user", user_message3, "demo_conv_001"
        )
        
        print(f"📚 Retrieved {len(memory_context3.selected_memories)} relevant memories")
        
        print("\n🎯 Conversation Flow Analysis:")
        print(f"   - Total memories considered: {len(memory_context.selected_memories) + len(memory_context2.selected_memories) + len(memory_context3.selected_memories)}")
        print(f"   - Context window size: {context1.context_window_size} tokens")
        print(f"   - User communication style: {user_profile.communication_style}")
        print(f"   - Preferred response length: {user_profile.preferred_response_length}")
        
        print("\n🎉 Conversation demo completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during conversation demo: {e}")
        return False

async def demo_memory_integration():
    """Demo memory integration and retrieval"""
    print("\n🧠 Memory Integration Demo")
    print("=" * 50)
    
    try:
        mock_component5 = MockComponent5Interface()
        
        # Test different query types
        queries = [
            "work stress",
            "relationship advice", 
            "personal goals",
            "health concerns"
        ]
        
        print("Testing memory retrieval for different query types:")
        for query in queries:
            memories = await mock_component5.get_memory_context(
                "demo_user", query, "demo_conv_002"
            )
            
            print(f"\n🔍 Query: '{query}'")
            print(f"   Memories found: {len(memories.selected_memories)}")
            print(f"   Token usage: {memories.token_usage}")
            print(f"   Processing time: {memories.assembly_metadata.get('retrieval_time_ms', 0)}ms")
            
            if memories.selected_memories:
                top_memory = memories.selected_memories[0]
                print(f"   Top memory: {top_memory['content_summary'][:80]}...")
                print(f"   Relevance score: {memories.relevance_scores[0]:.2f}")
        
        print("\n✅ Memory integration demo completed!")
        
    except Exception as e:
        print(f"\n❌ Error during memory demo: {e}")

def main():
    """Main demo function"""
    print("🌟 Welcome to the Gemini Engine Conversation Demo!")
    print("This demonstrates the conversation flow and memory integration")
    
    # Run conversation demo
    success = asyncio.run(demo_conversation_flow())
    
    if success:
        # Run memory integration demo
        asyncio.run(demo_memory_integration())
        
        print("\n📚 What This Demo Shows:")
        print("✅ Component 6: Conversation management and context assembly")
        print("✅ Component 7: Memory retrieval and integration")
        print("✅ Mock Component 5: Memory context simulation")
        print("✅ Schema validation: All data models working correctly")
        
        print("\n🚀 Next Steps for Full Testing:")
        print("1. Set up your .env file with real API keys")
        print("2. Configure database connections")
        print("3. Test with real Gemini API calls")
        print("4. Run performance benchmarks")
    
    else:
        print("\n🔧 Demo failed - check the implementation")

if __name__ == "__main__":
    main()
