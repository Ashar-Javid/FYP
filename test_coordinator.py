"""
Simple test for the coordinator agent to verify it works correctly.
"""

from tools import ScenarioTool
from agents import CoordinatorAgent

def test_coordinator():
    print("Testing Coordinator Agent...")
    
    # Load scenario
    scenario_tool = ScenarioTool()
    scenario = scenario_tool.get_scenario_5ub()
    print(f"Loaded scenario: {scenario['scenario_name']}")
    
    # Test coordinator without API
    coordinator = CoordinatorAgent()
    
    try:
        decision = coordinator.analyze_scenario_and_decide(scenario, [])
        print("Coordinator decision:", decision)
    except Exception as e:
        print(f"Coordinator failed with API. Testing fallback: {e}")
        # Test fallback decision
        fallback = coordinator._get_fallback_decision(scenario)
        print("Fallback decision:", fallback)

if __name__ == "__main__":
    test_coordinator()