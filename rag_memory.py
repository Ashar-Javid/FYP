"""
RAG Memory System for RIS Framework
Implements similarity search and pattern matching using simple similarity metrics.
Simplified version without heavy dependencies for better compatibility.
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any


class RAGMemorySystem:
    """
    Simplified RAG-based memory system for finding similar scenarios.
    Uses numerical similarity metrics instead of complex embeddings.
    """
    
    def __init__(self, dataset_path: str = "ris_user_algorithm_dataset.json"):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        
        # Simple configuration
        self.max_results = 5
        self.similarity_threshold = 0.3
        self.feature_weights = {
            "user_locations": 0.3,
            "channel_conditions": 0.3,
            "delta_snr_values": 0.4
        }
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the training dataset."""
        try:
            with open(self.dataset_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Dataset file {self.dataset_path} not found. Using empty dataset.")
            return []
    
    def _extract_numerical_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features for similarity matching."""
        case_data = input_data["case_data"]
        users = case_data["users"]
        
        features = []
        
        # Basic counts
        features.append(len(users))  # Number of users
        features.append(sum(1 for u in users if u.get("csi", {}).get("los", False)))  # LoS count
        
        # SNR statistics - handle both field name formats
        req_snrs = []
        ach_snrs = []
        delta_snrs = []
        
        for u in users:
            # Handle different field name formats
            req_snr = u.get("required_snr_dB", u.get("req_snr_dB", 0))
            ach_snr = u.get("achieved_snr_dB", 0)
            delta_snr = u.get("delta_snr_dB", ach_snr - req_snr)
            
            req_snrs.append(req_snr)
            ach_snrs.append(ach_snr)
            delta_snrs.append(delta_snr)
        
        features.extend([
            np.mean(req_snrs), np.std(req_snrs),
            np.mean(ach_snrs), np.std(ach_snrs),
            np.mean(delta_snrs), np.std(delta_snrs),
            min(delta_snrs), max(delta_snrs)
        ])
        
        # Spatial features
        distances = [np.sqrt(u["coord"][0]**2 + u["coord"][1]**2) for u in users]
        features.extend([np.mean(distances), np.std(distances), min(distances), max(distances)])
        
        # Channel quality indicators
        k_factors = [u.get("csi", {}).get("K_factor_dB", 0) for u in users if "K_factor_dB" in u.get("csi", {})]
        if k_factors:
            features.extend([np.mean(k_factors), np.std(k_factors)])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calculate similarity between two feature vectors using cosine similarity."""
        if len(features1) != len(features2):
            return 0.0
        
        # Convert to numpy arrays
        v1 = np.array(features1)
        v2 = np.array(features2)
        
        # Cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def find_similar_scenarios(self, current_scenario: Dict[str, Any], 
                             max_results: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find scenarios similar to the current one using numerical feature similarity.
        Returns list of (scenario, similarity_score) tuples.
        """
        if not self.dataset:
            return []
        
        if max_results is None:
            max_results = self.max_results
        
        # Extract features from current scenario
        current_features = self._extract_numerical_features(current_scenario)
        
        # Calculate similarities
        similarities = []
        for i, example in enumerate(self.dataset):
            example_features = self._extract_numerical_features(example["input"])
            similarity = self._calculate_similarity(current_features, example_features)
            
            if similarity >= self.similarity_threshold:
                similarities.append((example, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def extract_algorithm_patterns(self, similar_scenarios: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Extract algorithm selection patterns from similar scenarios."""
        if not similar_scenarios:
            return {
                "recommended_algorithm": "manifold", 
                "confidence": 0.0, 
                "reasoning": "No similar scenarios found - using default"
            }
        
        # Count algorithm preferences weighted by similarity
        algorithm_votes = {}
        total_weight = 0
        
        for scenario, similarity in similar_scenarios:
            algorithm = scenario["output"]["best_algorithm"]
            weight = similarity
            
            algorithm_votes[algorithm] = algorithm_votes.get(algorithm, 0) + weight
            total_weight += weight
        
        # Normalize votes
        if total_weight > 0:
            for algo in algorithm_votes:
                algorithm_votes[algo] /= total_weight
        
        # Find best algorithm
        best_algorithm = max(algorithm_votes, key=algorithm_votes.get) if algorithm_votes else "manifold"
        confidence = algorithm_votes.get(best_algorithm, 0.0)
        
        # Generate reasoning
        reasoning_parts = [f"Based on {len(similar_scenarios)} similar scenarios."]
        for algo, vote in sorted(algorithm_votes.items(), key=lambda x: x[1], reverse=True)[:3]:
            reasoning_parts.append(f"{algo}: {vote:.2f}")
        
        return {
            "recommended_algorithm": best_algorithm,
            "confidence": confidence,
            "reasoning": " ".join(reasoning_parts),
            "algorithm_distribution": algorithm_votes
        }
    
    def extract_user_selection_patterns(self, similar_scenarios: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Extract user selection patterns from similar scenarios."""
        if not similar_scenarios:
            return {
                "selection_strategy": "select_negative_delta", 
                "reasoning": "No patterns available - using default strategy"
            }
        
        selection_patterns = []
        
        for scenario, similarity in similar_scenarios:
            input_users = scenario["input"]["case_data"]["users"]
            selected_users = scenario["output"]["selected_users"]
            
            # Analyze selection criteria
            selected_delta_snrs = []
            unselected_delta_snrs = []
            
            for user in input_users:
                if user["id"] in selected_users:
                    selected_delta_snrs.append(user["delta_snr_dB"])
                else:
                    unselected_delta_snrs.append(user["delta_snr_dB"])
            
            if selected_delta_snrs:
                pattern = {
                    "similarity": similarity,
                    "selected_delta_range": (min(selected_delta_snrs), max(selected_delta_snrs)),
                    "selection_threshold": max(selected_delta_snrs),
                    "avg_selected_delta": np.mean(selected_delta_snrs)
                }
                selection_patterns.append(pattern)
        
        # Analyze patterns
        if selection_patterns:
            thresholds = [p["selection_threshold"] for p in selection_patterns]
            avg_threshold = np.mean(thresholds)
            
            reasoning = f"Historical patterns suggest selecting users with delta SNR below {avg_threshold:.1f} dB based on {len(selection_patterns)} similar cases"
        else:
            avg_threshold = 0.0
            reasoning = "Using fallback strategy: select users with negative delta SNR"
        
        return {
            "selection_strategy": "learned_threshold",
            "threshold": avg_threshold,
            "reasoning": reasoning,
            "pattern_confidence": len(selection_patterns) / len(similar_scenarios) if similar_scenarios else 0.0
        }

    def learn_from_successful_iteration(self, scenario_data: Dict[str, Any], 
                                       algorithm_used: str, 
                                       final_snr_deltas: List[float],
                                       success_metrics: Dict[str, float] = None) -> bool:
        """
        Learn from a successful iteration by adding it to the dataset.
        
        Args:
            scenario_data: The scenario that was successful
            algorithm_used: The algorithm that worked well
            final_snr_deltas: The final SNR delta values achieved
            success_metrics: Optional success metrics (RSWS, PSD, etc.)
        
        Returns:
            bool: True if the learning was successful
        """
        try:
            # Create a new entry for the dataset
            new_entry = {
                "input": scenario_data,
                "output": {
                    "best_algorithm": algorithm_used,
                    "final_delta_snrs": final_snr_deltas,
                    "success_metrics": success_metrics or {},
                    "timestamp": time.time()
                }
            }
            
            # Add to in-memory dataset
            self.dataset.append(new_entry)
            
            # Save updated dataset to file
            with open(self.dataset_path, 'w') as f:
                json.dump(self.dataset, f, indent=2)
            
            print(f"RAG learned from successful iteration. Dataset now has {len(self.dataset)} entries.")
            return True
            
        except Exception as e:
            print(f"Error learning from successful iteration: {e}")
            return False

    def get_dataset_size(self) -> int:
        """Get the current size of the dataset."""
        return len(self.dataset)


# Export the RAG system
__all__ = ['RAGMemorySystem']