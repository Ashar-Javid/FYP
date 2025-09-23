"""
RIS Framework Agents
Implements the Coordinator and Evaluator agents using the official Cerebras SDK.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_SDK_AVAILABLE = True
except ImportError:
    CEREBRAS_SDK_AVAILABLE = False
    print("⚠️  Cerebras SDK not available. Using fallback mode.")

from config import CEREBRAS_API_KEY, CEREBRAS_MODEL, COORDINATOR_CONFIG, EVALUATOR_CONFIG
from llm_logger import log_llm_response
from rag_memory import RAGMemorySystem


class CerebrasClient:
    """
    Client for Cerebras API using the official SDK with fallback support.
    """
    
    def __init__(self, model_name: str = CEREBRAS_MODEL, temperature: float = 0.1, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if CEREBRAS_SDK_AVAILABLE:
            self.client = Cerebras(api_key=CEREBRAS_API_KEY)
            self.use_sdk = True
        else:
            self.use_sdk = False
    
    def call(self, prompt: str, system_prompt: str = "") -> str:
        """Call the Cerebras API with a prompt using SDK or fallback."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.use_sdk and CEREBRAS_SDK_AVAILABLE:
            try:
                # Use official Cerebras SDK
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"Cerebras SDK Error: {e}")
                return self._fallback_decision(prompt, system_prompt)
        else:
            # Use fallback mode
            return self._fallback_decision(prompt, system_prompt)
    
    def _fallback_decision(self, prompt: str, system_prompt: str = "") -> str:
        """Fallback decision-making when API is not available."""
        # Simple pattern-based responses for testing
        if "algorithm" in prompt.lower():
            return "GD"  # Default to Gradient Descent
        elif "user" in prompt.lower():
            return "users_1_2_3"  # Default user selection
        elif "power" in prompt.lower():
            return "increase_power_2db"  # Conservative power adjustment
        else:
            return "continue_optimization"


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
        # Fix data structure for RAG compatibility
        scenario_for_rag = {"case_data": scenario_data["scenario_data"]}
        similar_scenarios = self.rag_system.find_similar_scenarios(scenario_for_rag)
        
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
            # Log raw response
            log_llm_response(agent="coordinator", phase="decision", prompt=prompt, response=response)
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
        """Parse the LLM response into a structured decision with robust error handling."""
        try:
            # Clean the response first
            response = response.strip()
            
            # Remove markdown code block formatting if present
            if "```json" in response:
                start_marker = "```json"
                end_marker = "```"
                start_idx = response.find(start_marker) + len(start_marker)
                end_idx = response.find(end_marker, start_idx)
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx].strip()
                else:
                    json_str = response
            else:
                # Try to extract JSON from the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                else:
                    raise ValueError("No JSON found in response")
            
            # Repair similar to evaluation parser
            import re
            json_candidate = json_str
            json_candidate = re.sub(r',\s*([}\]])', r'\1', json_candidate)
            # Balance braces heuristic
            def _balanced_dec(s: str) -> bool:
                stack = []
                pairs = {'}':'{', ']':'['}
                for ch in s:
                    if ch in '{[': stack.append(ch)
                    elif ch in '}]':
                        if not stack or stack[-1] != pairs[ch]:
                            return False
                        stack.pop()
                return not stack
            if not _balanced_dec(json_candidate):
                for cut in range(len(json_candidate)-1, max(len(json_candidate)-400, 10), -1):
                    trimmed = json_candidate[:cut]
                    if _balanced_dec(trimmed):
                        json_candidate = trimmed
                        break
            json_str = json_candidate.replace('\n', ' ').replace('\r', ' ')
            decision = json.loads(json_str)
            
            # Validate and fix required fields
            if "selected_users" not in decision:
                # Try to extract users from text
                if "users" in response.lower():
                    decision["selected_users"] = [1, 2, 3]  # Default fallback
                    
            if "selected_algorithm" not in decision:
                if "manifold" in response.lower():
                    decision["selected_algorithm"] = "manifold"
                elif "gradient" in response.lower() or "gd" in response.lower():
                    decision["selected_algorithm"] = "GD"
                else:
                    decision["selected_algorithm"] = "manifold"
            
            if "base_station_power_change" not in decision:
                decision["base_station_power_change"] = "+2.0"
            
            # Parse power change safely
            power_change_str = str(decision["base_station_power_change"])
            try:
                power_change = float(power_change_str.replace('+', '').replace(' dB', '').replace('dB', ''))
                decision["power_change_dB"] = power_change
            except:
                decision["power_change_dB"] = 2.0
            
            return decision
            
        except Exception as e:
            print(f"⚠️  JSON parsing failed, using fallback decision. Error: {e}")
            snippet = (response[:300] + '...') if len(response) > 300 else response
            print(f"Response snippet: {snippet}")
            return self._get_fallback_decision_from_text(response)
    
    def _get_fallback_decision_from_text(self, response: str) -> Dict[str, Any]:
        """Extract decision from text when JSON parsing fails."""
        # Extract users mentioned in text
        users = []
        for i in range(1, 6):
            if f"user {i}" in response.lower() or f"user{i}" in response.lower():
                users.append(i)
        
        if not users:
            users = [1, 2, 3]  # Default
        
        # Extract algorithm
        algorithm = "manifold"  # Default
        if "gradient" in response.lower() or "gd" in response.lower():
            algorithm = "GD"
        elif "alternating" in response.lower() or "ao" in response.lower():
            algorithm = "AO"
        
        # Extract power change
        power_change = 2.0  # Default
        import re
        power_matches = re.findall(r'(\d+\.?\d*)\s*db', response.lower())
        if power_matches:
            try:
                power_change = float(power_matches[0])
            except:
                pass
        
        return {
            "selected_users": users,
            "selected_algorithm": algorithm,
            "base_station_power_change": f"+{power_change}",
            "power_change_dB": power_change,
            "reasoning": "Extracted from text (JSON parsing failed)",
            "is_fallback": True
        }
    
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
            log_llm_response(agent="evaluator", phase="evaluation", prompt=prompt, response=response)
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
        """Parse the LLM evaluation response with robust error handling."""
        try:
            response = response.strip()

            # Quick repair: drop trailing markdown fences and incomplete lines
            if response.count('{') > 0 and response.count('}') == 0:
                # Truncated JSON – fallback
                raise ValueError("Truncated JSON - no closing brace")

            # Extract probable JSON region (first '{' to last '}')
            start_brace = response.find('{')
            end_brace = response.rfind('}')
            if start_brace == -1 or end_brace == -1:
                raise ValueError("No JSON braces detected")

            json_candidate = response[start_brace:end_brace+1]

            # Remove code fences if present
            if '```' in json_candidate:
                json_candidate = json_candidate.replace('```json', '').replace('```', '')

            # Basic repairs: remove trailing commas before closing braces/brackets
            import re
            json_candidate = re.sub(r',\s*([}\]])', r'\1', json_candidate)

            # If quotes broken at end, attempt truncation to last complete key-value
            # Heuristic: ensure matching counts of braces and brackets
            def _balanced(s: str) -> bool:
                stack = []
                pairs = {'}':'{', ']':'['}
                for ch in s:
                    if ch in '{[': stack.append(ch)
                    elif ch in '}]':
                        if not stack or stack[-1] != pairs[ch]:
                            return False
                        stack.pop()
                return not stack

            if not _balanced(json_candidate):
                # Try trimming from end until balanced or too short
                for cut in range(len(json_candidate)-1, max(len(json_candidate)-400, 10), -1):
                    trimmed = json_candidate[:cut]
                    if _balanced(trimmed):
                        json_candidate = trimmed
                        break

            json_str = json_candidate.replace('\n', ' ').replace('\r', ' ')
            evaluation = json.loads(json_str)
            
            # Remove markdown code block formatting if present
            if "```json" in response:
                start_marker = "```json"
                end_marker = "```"
                start_idx = response.find(start_marker) + len(start_marker)
                end_idx = response.find(end_marker, start_idx)
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx].strip()
                else:
                    json_str = response
            else:
                # Try to extract JSON from the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                else:
                    raise ValueError("No JSON found in response")
            
            # Clean common JSON formatting issues
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            
            evaluation = json.loads(json_str)
            
            # Ensure required fields exist with defaults
            defaults = {
                "performance_summary": "Performance assessed",
                "power_efficiency_status": "acceptable",
                "fairness_assessment": "fair",
                "recommendations": "Continue optimization",
                "success": False,
                "overall_score": 0.6
            }
            
            for field, default_value in defaults.items():
                if field not in evaluation:
                    evaluation[field] = default_value
            
            # Ensure success is boolean
            if isinstance(evaluation.get("success"), str):
                evaluation["success"] = evaluation["success"].lower() in ["true", "yes", "successful", "1"]
            
            # Ensure score is numeric
            if "overall_score" in evaluation:
                try:
                    evaluation["overall_score"] = float(evaluation["overall_score"])
                except:
                    evaluation["overall_score"] = 0.6
            
            return evaluation
            
        except Exception as e:
            print(f"⚠️  Evaluation JSON parsing failed, using fallback. Error: {e}")
            snippet = (response[:300] + '...') if len(response) > 300 else response
            print(f"Response snippet: {snippet}")
            return self._get_fallback_evaluation(response)
    
    def _get_fallback_evaluation(self, response: str) -> Dict[str, Any]:
        """Extract evaluation from text when JSON parsing fails."""
        # Simple text-based evaluation
        success = any(word in response.lower() for word in ["success", "good", "improved", "better"])
        
        # Extract score if mentioned
        score = 0.6  # Default
        import re
        score_matches = re.findall(r'score[:\s]*(\d+\.?\d*)', response.lower())
        if score_matches:
            try:
                score = float(score_matches[0])
                if score > 1:  # If it's a percentage
                    score = score / 100
            except:
                pass
        
        return {
            "performance_summary": "Extracted from text analysis",
            "power_efficiency_status": "acceptable",
            "fairness_assessment": "reasonable",
            "recommendations": "Continue optimization based on text analysis",
            "success": success,
            "overall_score": score,
            "is_fallback": True
        }
    
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


# Export agents
__all__ = ['CoordinatorAgent', 'EvaluatorAgent']