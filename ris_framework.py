"""
RIS Multi-User Optimization Framework
Main execution loop that coordinates between scenario selection, coordinator decisions, 
algorithm execution, and evaluator feedback.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import framework components
from config import FRAMEWORK_CONFIG, ensure_directories, load_framework_memory, save_framework_memory
from tools import ScenarioTool, AlgorithmTool, PowerControlTool, MemoryTool, VisualizationTool
from logger import log_event
from agents import CoordinatorAgent, EvaluatorAgent
from logger import MetricsLogger  # unified logger


class RISOptimizationFramework:
    """
    Main framework class that orchestrates the multi-user RIS optimization process.
    """
    
    def __init__(self):
        ensure_directories()
        
        # Initialize tools
        self.scenario_tool = ScenarioTool()
        self.algorithm_tool = AlgorithmTool()
        self.power_tool = PowerControlTool()
        self.memory_tool = MemoryTool()
        self.viz_tool = VisualizationTool()
        
        # Initialize agents
        self.coordinator = CoordinatorAgent()
        self.evaluator = EvaluatorAgent()
        
        # Framework state
        self.current_scenario = None
        self.iteration_history = []
        self.current_power_dBm = 20.0  # Default starting power
        self.max_iterations = FRAMEWORK_CONFIG["max_iterations"]
        
        print("=== RIS Multi-User Optimization Framework Initialized ===")
    
    def run_optimization_session(self) -> Dict[str, Any]:
        """
        Run a complete optimization session on scenario 5U_B.
        Returns comprehensive results of the session.
        """
        print(f"\n🚀 Starting optimization session at {datetime.now()}")
        
        # Initialize session
        session_start_time = time.time()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_logger = MetricsLogger(session_id)

        # Optional: reset framework memory at session start
        try:
            if FRAMEWORK_CONFIG.get("reset_framework_memory", False):
                fresh_memory = {
                    "learned_patterns": [],
                    "iteration_history": [],
                    "successful_strategies": [],
                    "failed_strategies": []
                }
                save_framework_memory(fresh_memory)
                print("🧹 Framework memory reset as per config.")
        except Exception as e:
            print(f"⚠️  Could not reset framework memory: {e}")

        # Load scenario (configurable)
        scenario_name = FRAMEWORK_CONFIG.get("default_scenario", "5U_B")
        if hasattr(self.scenario_tool, "get_scenario"):
            self.current_scenario = self.scenario_tool.get_scenario(scenario_name)
        else:
            # Backward-compat: specific method for 5U_B
            self.current_scenario = self.scenario_tool.get_scenario_5ub() if scenario_name == "5U_B" else self.scenario_tool.get_scenario_5ub()
        print(f"📊 Loaded scenario: {self.current_scenario['scenario_name']}")
        log_event(f"session start: scenario={self.current_scenario['scenario_name']}")

        # Create per-session plots folder and use it for all plots
        per_session_plots_dir = os.path.join(FRAMEWORK_CONFIG["plots_dir"], session_id)
        os.makedirs(per_session_plots_dir, exist_ok=True)

        # Create compact scenario scatter plot (replacing initial two-panel plot)
        scatter_plot_path = self.viz_tool.plot_selected_scenario_scatter(
            self.current_scenario,
            os.path.join(per_session_plots_dir, f"initial_scenario_{session_id}.png")
        )
        print(f"📈 Scenario setup plot saved: {scatter_plot_path}")

        # Initialize power level
        self.current_power_dBm = self.current_scenario["sim_settings"]["default_bs_power_dBm"]
        
    # Run optimization iterations
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🔄 === ITERATION {iteration} ===")
            
            try:
                iteration_result = self._run_single_iteration(iteration)
                self.iteration_history.append(iteration_result)
                # concise logging: actions & key results
                coord = iteration_result.get('coordinator_decision', {})
                evald = iteration_result.get('evaluation', {})
                log_event(
                    f"iter={iteration} users={coord.get('selected_users', [])} algo={coord.get('selected_algorithm')} "
                    f"power={self.current_power_dBm:.1f}dBm success={evald.get('success', False)} score={evald.get('overall_score', 0):.2f}"
                )
                # Log metrics to CSV
                try:
                    metrics_logger.log_iteration(iteration_result)
                except Exception as e:
                    print(f"⚠️  Metrics logging failed: {e}")
                
                # Check stopping criteria
                if self._check_stopping_criteria(iteration_result):
                    # Allow at least N iterations for richer plots; else stop
                    min_iters = FRAMEWORK_CONFIG.get("min_iterations_before_stop", 3)
                    if iteration >= min_iters:
                        print(f"✅ Stopping criteria met after {iteration} iterations!")
                        log_event(f"stopping at iter={iteration} (criteria met)")
                        break
                    else:
                        print(f"ℹ️  Success criteria met early; continuing to gather more iteration data (min {min_iters}).")
                    
                # Update power for next iteration if needed
                if iteration_result.get("power_adjustment_needed", False):
                    new_power = iteration_result.get("recommended_power_dBm", self.current_power_dBm)
                    print(f"⚡ Power adjusted: {self.current_power_dBm:.1f} → {new_power:.1f} dBm")
                    log_event(f"power_adjust: {self.current_power_dBm:.1f}dBm -> {new_power:.1f}dBm")
                    self.current_power_dBm = new_power
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                # Continue with next iteration
                continue
        
        # Generate final results
        session_results = self._generate_session_results(session_id, session_start_time)

        # Post-session: Run algorithm comparison with all users
        try:
            all_user_comparison = self.algorithm_tool.run_algorithm_comparison(
                self.current_scenario, self.current_power_dBm
            )
            # Extract agent's final iteration SNRs
            agent_final_snrs = {}
            if self.iteration_history:
                last_algo_results = self.iteration_history[-1].get('algorithm_results', {})
                agent_final_snrs = last_algo_results.get('final_snrs', {})
            comparison_plot = self.viz_tool.plot_algorithm_comparison(
                self.current_scenario, all_user_comparison,
                os.path.join(per_session_plots_dir, f"algorithm_comparison_{session_id}.png"),
                agent_final_snrs=agent_final_snrs
            )
            session_results.setdefault("plots", {})["algorithm_comparison"] = comparison_plot
            session_results["algorithm_comparison_results"] = all_user_comparison
        except Exception as e:
            print(f"⚠️  Algorithm comparison failed: {e}")
        
        # Save results
        self._save_session_results(session_results, session_id)
        
        print(f"\n🎉 Optimization session completed! Results saved.")
        log_event("session end")
        return session_results
    
    def _run_single_iteration(self, iteration_num: int) -> Dict[str, Any]:
        """Run a single optimization iteration."""
        
        iteration_start_time = time.time()
        
        # 1. Coordinator Analysis and Decision
        print("🧠 Coordinator analyzing scenario...")
        coordinator_decision = self.coordinator.analyze_scenario_and_decide(
            self.current_scenario, 
            self.iteration_history
        )
        
        print(f"📋 Coordinator Decision:")
        print(f"   Selected Users: {coordinator_decision['selected_users']}")
        print(f"   Algorithm: {coordinator_decision['selected_algorithm']}")
        print(f"   Power Change: {coordinator_decision.get('power_change_dB', 0):.1f} dB")
        print(f"   Reasoning: {coordinator_decision.get('reasoning', 'N/A')[:100]}...")
        
        # 2. Apply power adjustment
        if 'power_change_dB' in coordinator_decision:
            self.current_power_dBm += coordinator_decision['power_change_dB']
            self.current_power_dBm = max(20, min(45, self.current_power_dBm))  # Clamp to range
        
        # 3. Execute Algorithm
        print("🔧 Executing optimization algorithm...")
        before_state = self.current_scenario.copy()
        
        algorithm_results = self.algorithm_tool.execute_algorithm(
            coordinator_decision["selected_algorithm"],
            self.current_scenario,
            coordinator_decision["selected_users"],
            self.current_power_dBm
        )
        
        after_state = algorithm_results.copy()
        print(f"⚙️  Algorithm executed in {algorithm_results.get('execution_time', 0):.2f}s")
        print(f"   Converged: {algorithm_results.get('converged', False)}")
        
        # 4. Evaluator Assessment
        print("🔍 Evaluator assessing results...")
        try:
            # Debug: Check if states contain serializable data
            # Convert numpy arrays to lists for JSON serialization
            clean_before_state = self._make_json_serializable(before_state)
            clean_after_state = self._make_json_serializable(after_state)
            
            evaluation = self.evaluator.evaluate_iteration(
                clean_before_state, clean_after_state, coordinator_decision
            )
        except Exception as e:
            print(f"⚠️  Evaluator error: {e}")
            # Provide fallback evaluation
            evaluation = {
                "performance_summary": "Evaluation failed, using fallback",
                "power_efficiency_status": "acceptable",
                "fairness_assessment": "reasonable",
                "recommendations": "Continue optimization",
                "success": False,
                "overall_score": 0.5,
                "is_fallback": True
            }
        
        print(f"📊 Evaluation Results:")
        print(f"   Success: {evaluation.get('success', False)}")
        print(f"   Score: {evaluation.get('overall_score', 0):.2f}")
        print(f"   Power Efficiency: {evaluation.get('power_efficiency_status', 'unknown')}")
        print(f"   Summary: {evaluation.get('performance_summary', 'N/A')[:100]}...")
        
        # 5. Power Control Assessment
        power_recommendation = self._assess_power_needs(evaluation, after_state)
        
        # 6. Compile iteration results
        # Compute per-user final delta SNRs for plotting/metrics
        per_user_final_delta = []
        if "final_snrs" in algorithm_results:
            for user in self.current_scenario["scenario_data"]["users"]:
                uid = user["id"]
                final_snr = algorithm_results["final_snrs"].get(uid, user["achieved_snr_dB"])
                final_delta = final_snr - user["req_snr_dB"]
                per_user_final_delta.append({"user_id": uid, "final_delta_snr_dB": final_delta})
        avg_final_delta = float(sum(u["final_delta_snr_dB"] for u in per_user_final_delta)/len(per_user_final_delta)) if per_user_final_delta else 0.0
        iteration_result = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - iteration_start_time,
            "coordinator_decision": coordinator_decision,
            "algorithm_results": algorithm_results,
            "evaluation": evaluation,
            "power_recommendation": power_recommendation,
            "current_power_dBm": self.current_power_dBm,
            "success": evaluation.get("success", False),
            "per_user_final_delta_snr": per_user_final_delta,
            "avg_final_delta_snr": avg_final_delta
        }
        
        # 7. Store in memory for learning
        try:
            # Clean iteration result for JSON serialization before storing
            clean_iteration_result = self._make_json_serializable(iteration_result)
            self.memory_tool.store_iteration_result(clean_iteration_result)
        except Exception as e:
            print(f"⚠️  Warning: Could not store iteration result in memory: {e}")
        
        return iteration_result
    
    def _assess_power_needs(self, evaluation: Dict[str, Any], after_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if power adjustments are needed based on evaluation."""
        
        power_recommendation = {
            "adjustment_needed": False,
            "recommended_power_dBm": self.current_power_dBm,
            "reasoning": "No adjustment needed"
        }
        
        # Check for power waste (excessive delta SNR)
        if evaluation.get("power_efficiency_status") == "wasteful":
            power_waste = evaluation.get("power_waste_score", 0)
            if power_waste > 5:  # Significant waste
                new_power, reason = self.power_tool.suggest_power_reduction(
                    self.current_power_dBm, [power_waste]
                )
                power_recommendation = {
                    "adjustment_needed": True,
                    "recommended_power_dBm": new_power,
                    "reasoning": f"Power reduction recommended: {reason}"
                }
        
        # Check for insufficient power (users not meeting requirements)
        elif evaluation.get("satisfaction_improvement", 0) <= 0:
            if self.current_power_dBm < 45:  # Can still increase power
                new_power, reason = self.power_tool.calculate_power_adjustment(
                    self.current_power_dBm, self.current_scenario
                )
                if new_power > self.current_power_dBm:
                    power_recommendation = {
                        "adjustment_needed": True,
                        "recommended_power_dBm": new_power,
                        "reasoning": f"Power increase recommended: {reason}"
                    }
        
        return power_recommendation
    
    def _check_stopping_criteria(self, iteration_result: Dict[str, Any]) -> bool:
        """Check if stopping criteria are met."""
        
        evaluation = iteration_result.get("evaluation", {})
        
        # Success criteria: SNR requirements and power efficiency both satisfied
        snr_satisfied = evaluation.get("users_satisfied_after", 0) >= len(
            self.current_scenario["scenario_data"]["users"]
        )
        
        power_ok_values = FRAMEWORK_CONFIG.get("power_efficiency_ok_values", ["efficient", "appropriate"])
        power_efficient = evaluation.get("power_efficiency_status") in power_ok_values
        
        # Overall success indicator
        overall_score_stop_threshold = FRAMEWORK_CONFIG.get("overall_score_stop_threshold", 0.8)
        overall_success = evaluation.get("success", False) and evaluation.get("overall_score", 0) > overall_score_stop_threshold
        
        stopping_conditions = {
            "snr_requirements_met": snr_satisfied,
            "power_efficient": power_efficient,
            "overall_success": overall_success
        }
        
        # Stop if all conditions are met or if we've made significant progress
        require_eff = FRAMEWORK_CONFIG.get("require_power_efficiency_for_stop", True)
        snr_gate = (snr_satisfied and power_efficient) if require_eff else snr_satisfied
        return snr_gate or overall_success
    
    def _generate_session_results(self, session_id: str, start_time: float) -> Dict[str, Any]:
        """Generate comprehensive session results."""
        
        total_time = time.time() - start_time
        # Ensure per-session plots directory exists for this session
        per_session_plots_dir = os.path.join(FRAMEWORK_CONFIG["plots_dir"], session_id)
        os.makedirs(per_session_plots_dir, exist_ok=True)
        
        # Analyze session performance
        final_evaluation = None
        if self.iteration_history:
            final_iteration = self.iteration_history[-1]
            final_evaluation = final_iteration.get("evaluation", {})
        
        # Success metrics
        successful_iterations = sum(1 for it in self.iteration_history if it.get("success", False))
        success_rate = successful_iterations / len(self.iteration_history) if self.iteration_history else 0
        
        # Performance progression
        if len(self.iteration_history) >= 2:
            first_score = self.iteration_history[0].get("evaluation", {}).get("overall_score", 0)
            last_score = self.iteration_history[-1].get("evaluation", {}).get("overall_score", 0)
            improvement = last_score - first_score
        else:
            improvement = 0
        
        # Generate new plots (power/SNR evolution and efficiency)
        power_snr_plot = None
        efficiency_plot = None
        if self.iteration_history:
            try:
                power_snr_plot = self.viz_tool.plot_power_and_snr_evolution(
                    self.iteration_history,
                    os.path.join(per_session_plots_dir, f"power_snr_evolution_{session_id}.png")
                )
            except Exception as e:
                print(f"⚠️  Could not generate power/SNR evolution plot: {e}")
            try:
                efficiency_plot = self.viz_tool.plot_power_efficiency(
                    self.iteration_history,
                    os.path.join(per_session_plots_dir, f"power_efficiency_{session_id}.png")
                )
            except Exception as e:
                print(f"⚠️  Could not generate power efficiency plot: {e}")
        
        session_results = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "scenario_name": self.current_scenario.get("scenario_name", "Unknown"),
            "total_execution_time": total_time,
            "total_iterations": len(self.iteration_history),
            "successful_iterations": successful_iterations,
            "success_rate": success_rate,
            "performance_improvement": improvement,
            "final_power_dBm": self.current_power_dBm,
            "final_evaluation": final_evaluation,
            "iteration_history": self.iteration_history,
            "plots": {
                "power_snr_evolution": power_snr_plot,
                "power_efficiency": efficiency_plot
            },
            "stopping_reason": self._get_stopping_reason()
        }

        # Add final per-user SNR comparison plot if data available
        try:
            final_snr_plot = self.viz_tool.plot_final_snr_comparison(
                self.current_scenario, self.iteration_history,
                os.path.join(per_session_plots_dir, f"final_snr_comparison_{session_id}.png")
            ) if self.iteration_history else None
            if final_snr_plot:
                session_results["plots"]["final_snr_comparison"] = final_snr_plot
        except Exception as e:
            print(f"⚠️  Could not generate final SNR comparison plot: {e}")
        
        return session_results
    
    def _get_stopping_reason(self) -> str:
        """Determine why the session stopped."""
        if not self.iteration_history:
            return "No iterations completed"
        
        if len(self.iteration_history) >= self.max_iterations:
            return f"Maximum iterations ({self.max_iterations}) reached"
        
        last_iteration = self.iteration_history[-1]
        if last_iteration.get("success", False):
            return "Success criteria achieved"
        
        return "Session completed"
    
    def _save_session_results(self, results: Dict[str, Any], session_id: str):
        """Save session results to file."""
        results_file = os.path.join(
            FRAMEWORK_CONFIG["results_dir"], 
            f"session_results_{session_id}.json"
        )
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Session results saved: {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def print_session_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of the session results."""
        print("\n" + "="*60)
        print("🎯 SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {results['session_id']}")
        print(f"Scenario: {results['scenario_name']}")
        print(f"Total Time: {results['total_execution_time']:.2f} seconds")
        print(f"Iterations: {results['total_iterations']}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"Performance Improvement: {results['performance_improvement']:+.3f}")
        print(f"Final Power: {results['final_power_dBm']:.1f} dBm")
        print(f"Stopping Reason: {results['stopping_reason']}")
        
        if results.get('final_evaluation'):
            eval_data = results['final_evaluation']
            print(f"\nFinal Evaluation:")
            print(f"  Overall Score: {eval_data.get('overall_score', 0):.2f}")
            print(f"  Power Efficiency: {eval_data.get('power_efficiency_status', 'Unknown')}")
            print(f"  Success: {eval_data.get('success', False)}")
        
        print("="*60)


def main():
    """Main function to run the RIS optimization framework."""
    
    print("🌟 RIS Multi-User Optimization Framework")
    print("🎯 Optimizing 5G/6G systems with intelligent coordination")
    print("-" * 60)
    
    try:
        # Initialize framework
        framework = RISOptimizationFramework()
        
        # Run optimization session
        results = framework.run_optimization_session()
        
        # Display summary
        framework.print_session_summary(results)
        
        # Success indicator
        if results:
            final_score = results.get('final_evaluation', {}).get('overall_score', 0)
            if final_score > 0.8:
                print("🎉 Optimization session completed successfully!")
            elif final_score > 0.6:
                print("✅ Optimization session completed with good results!")
            else:
                print("⚠️  Optimization session completed with room for improvement.")
        else:
            print("⚠️  Session completed but no results were generated.")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️  Session interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ Framework error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the framework
    results = main()
    
    if results:
        print(f"\n📊 Results saved in: {FRAMEWORK_CONFIG['results_dir']}")
        try:
            _sid = results.get('session_id')
            _plots_dir = os.path.join(FRAMEWORK_CONFIG['plots_dir'], _sid) if _sid else FRAMEWORK_CONFIG['plots_dir']
            print(f"📈 Plots saved in: {_plots_dir}")
        except Exception:
            print(f"📈 Plots saved in: {FRAMEWORK_CONFIG['plots_dir']}")
        
        # Optional: Print additional insights
        if input("\nShow detailed iteration history? (y/n): ").lower() == 'y':
            for i, iteration in enumerate(results.get('iteration_history', []), 1):
                print(f"\nIteration {i}:")
                print(f"  Algorithm: {iteration.get('coordinator_decision', {}).get('selected_algorithm', 'N/A')}")
                print(f"  Users: {iteration.get('coordinator_decision', {}).get('selected_users', [])}")
                print(f"  Success: {iteration.get('success', False)}")
                print(f"  Score: {iteration.get('evaluation', {}).get('overall_score', 0):.2f}")