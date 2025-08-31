#!/usr/bin/env python3
"""
Gemini Engine Demo Script
Demonstrates basic functionality without requiring external API keys
"""

import asyncio
import os
from datetime import datetime
from shared.schemas import (
    ConversationContext, UserProfile, PersonalityProfile,
    CommunicationStyle, MemoryContext
)
from component6.conversation_manager import ConversationManager
from component7.quality_analyzer import QualityAnalyzer
from shared.mock_interfaces import MockComponent5Interface

async def demo_basic_functionality():
    """Demo basic functionality with mock data"""
    print("🚀 Gemini Engine Demo - Basic Functionality")
    print("=" * 50)
    
    # Set test mode
    os.environ["TEST_MODE"] = "true"
    
    try:
        # 1. Test Component 6 - Conversation Manager
        print("\n📝 Testing Component 6: Conversation Manager")
        print("-" * 40)
        
        manager = ConversationManager()
        print("✅ Conversation Manager initialized successfully")
        
        # 2. Test Component 7 - Quality Analyzer
        print("\n🔍 Testing Component 7: Quality Analyzer")
        print("-" * 40)
        
        analyzer = QualityAnalyzer()
        print("✅ Quality Analyzer initialized successfully")
        
        # 3. Test Mock Memory Retriever
        print("\n🧠 Testing Mock Memory Retriever")
        print("-" * 40)
        
        retriever = MockComponent5Interface()
        memories = await retriever.get_memory_context("demo_user", "Hello", "demo_conv_001")
        print(f"✅ Retrieved {len(memories.selected_memories)} mock memories")
        
        # 4. Test Schema Creation
        print("\n📊 Testing Schema Creation")
        print("-" * 40)
        
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
        
        context = ConversationContext(
            user_id="demo_user",
            conversation_id="demo_conv_001",
            personality_profile=user_profile
        )
        
        print("✅ User profile and conversation context created successfully")
        print(f"   User ID: {context.user_id}")
        print(f"   Style: {context.personality_profile.communication_style}")
        print(f"   Response Length: {context.personality_profile.preferred_response_length}")
        
        # 5. Test Memory Context Processing
        print("\n💾 Testing Memory Context Processing")
        print("-" * 40)
        
        # Simulate Component 5 output
        memory_context = MemoryContext(
            selected_memories=memories.selected_memories,
            relevance_scores=memories.relevance_scores,
            token_usage=memories.token_usage,
            assembly_metadata=memories.assembly_metadata
        )
        
        print("✅ Memory context processed successfully")
        print(f"   Memories: {len(memory_context.selected_memories)}")
        print(f"   Token usage: {memory_context.token_usage}")
        
        print("\n🎉 All basic functionality tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("This might indicate an issue with the implementation")
        return False
    
    return True

async def demo_advanced_features():
    """Demo advanced features (requires proper environment setup)"""
    print("\n🚀 Gemini Engine Demo - Advanced Features")
    print("=" * 50)
    print("⚠️  Note: This demo requires proper environment configuration")
    print("   Set up your .env file with API keys to test these features")
    
    try:
        # Test personality engine
        from component6.personality_engine import PersonalityEngine
        
        personality_engine = PersonalityEngine()
        print("✅ Personality Engine initialized")
        
        # Test context assembler
        from component6.context_assembler import ContextAssembler
        
        context_assembler = ContextAssembler()
        print("✅ Context Assembler initialized")
        
        # Test proactive engine
        from component6.proactive_engine import ProactiveEngine
        
        proactive_engine = ProactiveEngine()
        print("✅ Proactive Engine initialized")
        
        print("\n🎯 Advanced components are ready for testing!")
        print("   Configure your environment and run full tests")
        
    except Exception as e:
        print(f"\n⚠️  Advanced features demo: {e}")
        print("   This is expected without proper configuration")

def main():
    """Main demo function"""
    print("🌟 Welcome to the Gemini Engine Demo!")
    print("This script demonstrates the basic functionality of Components 6 & 7")
    
    # Run basic demo
    success = asyncio.run(demo_basic_functionality())
    
    if success:
        # Run advanced demo
        asyncio.run(demo_advanced_features())
        
        print("\n📚 Next Steps:")
        print("1. Copy env.example to .env")
        print("2. Configure your API keys and database settings")
        print("3. Run: python -m pytest tests/ -v")
        print("4. Test real conversations with: python demo_conversation.py")
    
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Check that all modules are properly implemented")
        print("2. Verify Python path includes the project directory")
        print("3. Run: python test_basic.py")

if __name__ == "__main__":
    main()
