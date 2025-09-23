"""
RIS Framework Agents
Implements the Coordinator and Evaluator agents using direct API calls to Cerebras.
Simplified implementation without LangChain dependencies for better compatibility.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional
import urllib.request
import urllib.parse
import urllib.error

from config import CEREBRAS_API_KEY, CEREBRAS_MODEL, COORDINATOR_CONFIG, EVALUATOR_CONFIG
from rag_memory import RAGMemorySystem


class CerebrasClient:
    """
    Simple client for Cerebras API without LangChain dependencies.
    """
    
    def __init__(self, model_name: str = CEREBRAS_MODEL, temperature: float = 0.1, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://api.cerebras.ai/v1/chat/completions"
    
    def call(self, prompt: str, system_prompt: str = "") -> str:
        """Call the Cerebras API with a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            data = json.dumps(payload).encode('utf-8')
            request = urllib.request.Request(self.api_url, data=data, headers=headers)
            
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
                
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP Error {e.code}: {e.reason}"
            try:
                error_details = json.loads(e.read().decode('utf-8'))
                error_msg += f" - {error_details}"
            except:
                pass
            print(f"Cerebras API Error: {error_msg}")
            return f"API Error: {error_msg}"
            
        except Exception as e:
            print(f"Error calling Cerebras API: {e}")
            return f"Error: {str(e)}"


class CoordinatorAgent:
    """
    Coordinator Agent responsible for optimizing QoS in multi-user RIS systems.
    Uses RAG memory to learn patterns and make informed decisions.
    """
    
    def __init__(self):
        self.client = CerebrasClient(
            temperature=COORDINATOR_CONFIG["temperature"],
            max_tokens=COORDINATOR_CONFIG["max_tokens"]
        )
        self.rag_system = RAGMemorySystem()
        self.system_prompt = COORDINATOR_CONFIG["system_prompt"]
        
    def analyze_scenario_and_decide(self, scenario_data: Dict[str, Any], 
                                   iteration_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the scenario and make decisions about user selection, algorithm choice, and power adjustment.
        """
        # Get similar scenarios from RAG memory
        similar_scenarios = self.rag_system.find_similar_scenarios(scenario_data["scenario_data"])
        
        # Extract patterns
        algorithm_patterns = self.rag_system.extract_algorithm_patterns(similar_scenarios)
        user_selection_patterns = self.rag_system.extract_user_selection_patterns(similar_scenarios)
        
        # Prepare context for the LLM
        context = self._prepare_decision_context(
            scenario_data, algorithm_patterns, user_selection_patterns, iteration_history
        )
        
        # Get decision from LLM
        prompt = f"{self.system_prompt}\n\n{context}\n\nProvide your decision as a JSON object with the following structure:\n{{\n  \"selected_users\": [list of user IDs to optimize],\n  \"selected_algorithm\": \"algorithm_name\",\n  \"base_station_power_change\": \"power adjustment in dB (e.g., '+3.0' or '-2.5')\",\n  \"reasoning\": \"explanation of your decisions\"\n}}"
        
        try:
            response = self.client.call(prompt, self.system_prompt)
            decision = self._parse_decision_response(response)
            
            # Add pattern information to decision
            decision["algorithm_patterns"] = algorithm_patterns
            decision["user_selection_patterns"] = user_selection_patterns
            decision["similar_scenarios_count"] = len(similar_scenarios)
            
            return decision
            
        except Exception as e:
            print(f"Error in coordinator decision: {e}")
            return self._get_fallback_decision(scenario_data)
    
    def _prepare_decision_context(self, scenario_data: Dict[str, Any], 
                                algorithm_patterns: Dict[str, Any],
                                user_selection_patterns: Dict[str, Any],
                                iteration_history: List[Dict[str, Any]] = None) -> str:
        """Prepare context information for the coordinator's decision."""
        
        context_parts = []
        
        # Current scenario information
        scenario = scenario_data["scenario_data"]
        context_parts.append(f"CURRENT SCENARIO: {scenario_data.get('scenario_name', 'Unknown')}")
        context_parts.append(f"Number of users: {len(scenario['users'])}")
        context_parts.append(f"Current BS power: {scenario_data['sim_settings']['default_bs_power_dBm']} dBm")
        
        # User details
        context_parts.append("\nUSER DETAILS:")
        for user in scenario["users"]:
            los_status = "LoS" if user.get("csi", {}).get("los") else "NLoS"
            fading = user.get("csi", {}).get("fading", "Unknown")
            k_factor = user.get("csi", {}).get("K_factor_dB", "N/A")
            
            context_parts.append(
                f"User {user['id']}: Pos({user['coord'][0]:.1f}, {user['coord'][1]:.1f}), "
                f"Req={user['req_snr_dB']:.1f}dB, Ach={user['achieved_snr_dB']:.1f}dB, "
                f"Δ={user['delta_snr_dB']:.1f}dB, {los_status}, {fading}"
                + (f", K={k_factor:.1f}dB" if k_factor != "N/A" else "")
            )
        
        # RAG-based algorithm recommendations
        context_parts.append(f"\nALGORITHM PATTERNS FROM SIMILAR SCENARIOS:")
        context_parts.append(f"Recommended: {algorithm_patterns['recommended_algorithm']}")
        context_parts.append(f"Confidence: {algorithm_patterns['confidence']:.2f}")
        context_parts.append(f"Reasoning: {algorithm_patterns['reasoning']}")
        
        # User selection patterns
        context_parts.append(f"\nUSER SELECTION PATTERNS:")
        context_parts.append(f"Strategy: {user_selection_patterns['selection_strategy']}")
        if 'threshold' in user_selection_patterns:
            context_parts.append(f"Suggested threshold: {user_selection_patterns['threshold']:.2f} dB")
        context_parts.append(f"Reasoning: {user_selection_patterns['reasoning']}")
        
        # Iteration history (if available)
        if iteration_history:
            context_parts.append(f"\nITERATION HISTORY (Last {min(3, len(iteration_history))} iterations):")
            for i, iteration in enumerate(iteration_history[-3:], 1):
                context_parts.append(
                    f"Iteration {len(iteration_history)-3+i}: "
                    f"Algorithm={iteration.get('algorithm', 'N/A')}, "
                    f"Users={iteration.get('selected_users', [])}, "
                    f"Success={iteration.get('success', False)}"
                )
        
        return "\n".join(context_parts)
    
    def _parse_decision_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured decision."""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                decision = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["selected_users", "selected_algorithm", "base_station_power_change"]
                for field in required_fields:
                    if field not in decision:
                        raise ValueError(f"Missing required field: {field}")
                
                # Parse power change
                power_change_str = decision["base_station_power_change"]
                if isinstance(power_change_str, str):
                    power_change = float(power_change_str.replace('+', '').replace(' dB', '').replace('dB', ''))
                    decision["power_change_dB"] = power_change
                
                return decision
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing coordinator response: {e}")
            print(f"Raw response: {response}")
            raise
    
    def _get_fallback_decision(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide a fallback decision if LLM fails."""
        scenario = scenario_data["scenario_data"]
        
        # Select users with negative delta SNR
        selected_users = [
            user["id"] for user in scenario["users"] 
            if user["delta_snr_dB"] < 0
        ]
        
        # Default to manifold algorithm
        return {
            "selected_users": selected_users,
            "selected_algorithm": "manifold",
            "base_station_power_change": "+2.0",
            "power_change_dB": 2.0,
            "reasoning": "Fallback decision: selected users with negative delta SNR",
            "is_fallback": True
        }


class EvaluatorAgent:
    """
    Evaluator Agent that assesses optimization performance and provides feedback.
    """
    
    def __init__(self):
        self.client = CerebrasClient(
            temperature=EVALUATOR_CONFIG["temperature"],
            max_tokens=EVALUATOR_CONFIG["max_tokens"]
        )
        self.system_prompt = EVALUATOR_CONFIG["system_prompt"]
    
    def evaluate_iteration(self, before_state: Dict[str, Any], after_state: Dict[str, Any],
                          coordinator_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the results of an optimization iteration.
        """
        # Prepare evaluation context
        context = self._prepare_evaluation_context(before_state, after_state, coordinator_decision)
        
        # Get evaluation from LLM
        prompt = f"{self.system_prompt}\n\n{context}\n\nProvide your evaluation as a JSON object with the following structure:\n{{\n  \"performance_summary\": \"brief summary of changes\",\n  \"power_efficiency_status\": \"efficient/wasteful/appropriate\",\n  \"fairness_assessment\": \"fair/unfair distribution across users\",\n  \"recommendations\": \"specific recommendations for next iteration\",\n  \"success\": true/false,\n  \"overall_score\": 0.0-1.0\n}}"
        
        try:
            response = self.client.call(prompt, self.system_prompt)
            evaluation = self._parse_evaluation_response(response)
            
            # Add quantitative metrics
            evaluation.update(self._calculate_metrics(before_state, after_state))
            
            return evaluation
            
        except Exception as e:
            print(f"Error in evaluator: {e}")
            return self._get_fallback_evaluation(before_state, after_state)
    
    def _prepare_evaluation_context(self, before_state: Dict[str, Any], 
                                  after_state: Dict[str, Any],
                                  coordinator_decision: Dict[str, Any]) -> str:
        """Prepare context for evaluation."""
        
        context_parts = []
        
        # Coordinator's decision
        context_parts.append("COORDINATOR'S DECISION:")
        context_parts.append(f"Selected users: {coordinator_decision.get('selected_users', [])}")
        context_parts.append(f"Algorithm: {coordinator_decision.get('selected_algorithm', 'N/A')}")
        context_parts.append(f"Power change: {coordinator_decision.get('power_change_dB', 0):.1f} dB")
        context_parts.append(f"Reasoning: {coordinator_decision.get('reasoning', 'N/A')}")
        
        # Before optimization
        context_parts.append("\nBEFORE OPTIMIZATION:")
        before_users = before_state.get("scenario_data", {}).get("users", [])
        for user in before_users:
            context_parts.append(
                f"User {user['id']}: Req={user['req_snr_dB']:.1f}dB, "
                f"Ach={user['achieved_snr_dB']:.1f}dB, Δ={user['delta_snr_dB']:.1f}dB"
            )
        
        # After optimization
        context_parts.append("\nAFTER OPTIMIZATION:")
        if "final_snrs" in after_state:
            for user in before_users:
                user_id = user["id"]
                final_snr = after_state["final_snrs"].get(user_id, user["achieved_snr_dB"])
                final_delta = final_snr - user["req_snr_dB"]
                context_parts.append(
                    f"User {user_id}: Req={user['req_snr_dB']:.1f}dB, "
                    f"Ach={final_snr:.1f}dB, Δ={final_delta:.1f}dB"
                )
        
        # Performance metrics
        if "execution_time" in after_state:
            context_parts.append(f"\nExecution time: {after_state['execution_time']:.2f} seconds")
        if "converged" in after_state:
            context_parts.append(f"Algorithm converged: {after_state['converged']}")
        
        return "\n".join(context_parts)
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM evaluation response."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                
                # Ensure required fields exist
                required_fields = ["performance_summary", "power_efficiency_status", 
                                 "fairness_assessment", "recommendations", "success"]
                for field in required_fields:
                    if field not in evaluation:
                        evaluation[field] = "Not specified"
                
                # Ensure success is boolean
                if isinstance(evaluation.get("success"), str):
                    evaluation["success"] = evaluation["success"].lower() in ["true", "yes", "1"]
                
                return evaluation
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing evaluator response: {e}")
            print(f"Raw response: {response}")
            raise
    
    def _calculate_metrics(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quantitative performance metrics."""
        metrics = {}
        
        before_users = before_state.get("scenario_data", {}).get("users", [])
        
        if "final_snrs" in after_state and before_users:
            # Calculate improvements
            before_deltas = [user["delta_snr_dB"] for user in before_users]
            after_deltas = []
            
            for user in before_users:
                final_snr = after_state["final_snrs"].get(user["id"], user["achieved_snr_dB"])
                after_delta = final_snr - user["req_snr_dB"]
                after_deltas.append(after_delta)
            
            # Performance metrics
            metrics["avg_delta_improvement"] = np.mean(after_deltas) - np.mean(before_deltas)
            metrics["worst_user_improvement"] = min(after_deltas) - min(before_deltas)
            metrics["users_satisfied_before"] = sum(1 for d in before_deltas if d >= 0)
            metrics["users_satisfied_after"] = sum(1 for d in after_deltas if d >= 0)
            metrics["satisfaction_improvement"] = metrics["users_satisfied_after"] - metrics["users_satisfied_before"]
            
            # Power efficiency
            power_waste_after = sum(max(0, d - 2) for d in after_deltas)  # Excess beyond 2dB margin
            metrics["power_waste_score"] = power_waste_after
            
            # Fairness (variance in delta SNR)
            metrics["fairness_before"] = np.var(before_deltas)
            metrics["fairness_after"] = np.var(after_deltas)
            metrics["fairness_improvement"] = metrics["fairness_before"] - metrics["fairness_after"]
        
        return metrics
    
    def _get_fallback_evaluation(self, before_state: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback evaluation if LLM fails."""
        return {
            "performance_summary": "Evaluation failed - fallback assessment",
            "power_efficiency_status": "unknown",
            "fairness_assessment": "unknown",
            "recommendations": "Manual review required",
            "success": False,
            "overall_score": 0.5,
            "is_fallback": True
        }


# Export agents
__all__ = ['CoordinatorAgent', 'EvaluatorAgent']