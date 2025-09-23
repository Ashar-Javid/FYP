"""
Test script for RIS Framework
Simple test to verify all components work together.
"""

import os
import sys

def test_framework_components():
    """Test individual framework components."""
    
    print("🔧 Testing Framework Components...")
    
    try:
        # Test configuration
        print("1. Testing Configuration...")
        from config import FRAMEWORK_CONFIG, ensure_directories
        ensure_directories()
        print("   ✅ Configuration loaded successfully")
        
        # Test tools
        print("2. Testing Tools...")
        from tools import ScenarioTool, AlgorithmTool, PowerControlTool, MemoryTool
        
        scenario_tool = ScenarioTool()
        algorithm_tool = AlgorithmTool()
        power_tool = PowerControlTool()
        memory_tool = MemoryTool()
        print("   ✅ Tools initialized successfully")
        
        # Test scenario loading
        print("3. Testing Scenario Loading...")
        scenario = scenario_tool.get_scenario_5ub()
        print(f"   ✅ Scenario loaded: {scenario['scenario_name']} with {len(scenario['scenario_data']['users'])} users")
        
        # Test RAG system
        print("4. Testing RAG Memory System...")
        from rag_memory import RAGMemorySystem
        rag = RAGMemorySystem()
        print(f"   ✅ RAG system loaded with {len(rag.dataset)} examples")
        
        # Test simple similarity search
        # Create proper structure for RAG system
        scenario_for_rag = {"case_data": scenario["scenario_data"]}
        similar = rag.find_similar_scenarios(scenario_for_rag)
        print(f"   ✅ Found {len(similar)} similar scenarios")
        
        # Test agents (without API calls)
        print("5. Testing Agent Initialization...")
        from agents import CoordinatorAgent, EvaluatorAgent
        
        # Test with fallback mode first
        coordinator = CoordinatorAgent()
        evaluator = EvaluatorAgent()
        print("   ✅ Agents initialized successfully")
        
        print("\n🎉 All component tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_workflow():
    """Test a basic workflow without API calls."""
    
    print("\n🔄 Testing Basic Workflow...")
    
    try:
        from tools import ScenarioTool
        from rag_memory import RAGMemorySystem
        
        # Load scenario
        scenario_tool = ScenarioTool()
        scenario = scenario_tool.get_scenario_5ub()
        
        # Test RAG retrieval
        rag = RAGMemorySystem()
        scenario_for_rag = {"case_data": scenario["scenario_data"]}
        similar_scenarios = rag.find_similar_scenarios(scenario_for_rag)
        
        if similar_scenarios:
            # Test pattern extraction
            algo_patterns = rag.extract_algorithm_patterns(similar_scenarios)
            user_patterns = rag.extract_user_selection_patterns(similar_scenarios)
            
            print(f"   Algorithm recommendation: {algo_patterns['recommended_algorithm']}")
            print(f"   User selection strategy: {user_patterns['selection_strategy']}")
            print("   ✅ Pattern extraction successful")
        else:
            print("   ⚠️  No similar scenarios found - this is expected for first run")
        
        # Test scenario plotting
        plot_path = scenario_tool.plot_scenario(scenario)
        print(f"   ✅ Scenario plot created: {plot_path}")
        
        print("\n✅ Basic workflow test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("🌟 RIS Framework Test Suite")
    print("=" * 50)
    
    # Component tests
    component_test_passed = test_framework_components()
    
    if component_test_passed:
        # Workflow tests
        workflow_test_passed = test_basic_workflow()
        
        if workflow_test_passed:
            print("\n🎯 All tests passed! Framework is ready.")
            print("\nTo run the full framework:")
            print("   python ris_framework.py")
            
            # Ask if user wants to run full framework
            if input("\nRun full framework now? (y/n): ").lower() == 'y':
                print("\n" + "="*60)
                try:
                    from ris_framework import main as run_framework
                    run_framework()
                except Exception as e:
                    print(f"Framework execution failed: {e}")
        else:
            print("\n❌ Workflow tests failed. Please check the configuration.")
    else:
        print("\n❌ Component tests failed. Please check dependencies.")

if __name__ == "__main__":
    main()