"""
RAG Memory System for RIS Framework
Implements similarity search and pattern matching using embeddings and vector search.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler
from config import RAG_CONFIG


class RAGMemorySystem:
    """
    RAG-based memory system for finding similar scenarios and learning patterns.
    Uses both semantic embeddings and numerical feature matching.
    """
    
    def __init__(self, dataset_path: str = "ris_user_algorithm_dataset.json"):
        self.dataset_path = dataset_path
        self.embedding_model = SentenceTransformer(RAG_CONFIG["embedding_model"])
        self.dataset = self._load_dataset()
        self.feature_scaler = StandardScaler()
        
        # Initialize vector stores
        self.semantic_index = None
        self.numerical_index = None
        self._build_indices()
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the training dataset."""
        try:
            with open(self.dataset_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Dataset file {self.dataset_path} not found. Using empty dataset.")
            return []
    
    def _build_indices(self):
        """Build FAISS indices for both semantic and numerical similarity search."""
        if not self.dataset:
            return
        
        # Build semantic index
        scenario_descriptions = []
        numerical_features = []
        
        for example in self.dataset:
            # Create semantic description
            desc = self._create_scenario_description(example["input"])
            scenario_descriptions.append(desc)
            
            # Extract numerical features
            features = self._extract_numerical_features(example["input"])
            numerical_features.append(features)
        
        # Create semantic embeddings
        embeddings = self.embedding_model.encode(scenario_descriptions)
        
        # Build semantic FAISS index
        self.semantic_index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.semantic_index.add(embeddings.astype('float32'))
        
        # Build numerical FAISS index
        numerical_features = np.array(numerical_features)
        self.feature_scaler.fit(numerical_features)
        normalized_features = self.feature_scaler.transform(numerical_features)
        
        self.numerical_index = faiss.IndexFlatL2(normalized_features.shape[1])
        self.numerical_index.add(normalized_features.astype('float32'))
    
    def _create_scenario_description(self, input_data: Dict[str, Any]) -> str:
        """Create textual description of a scenario for semantic embedding."""
        case_data = input_data["case_data"]
        users = case_data["users"]
        
        # Basic scenario info
        desc_parts = [f"Scenario with {len(users)} users."]
        
        # User characteristics
        los_count = sum(1 for u in users if u.get("csi", {}).get("los", False))
        nlos_count = len(users) - los_count
        desc_parts.append(f"{los_count} users with LoS, {nlos_count} users with NLoS.")
        
        # SNR requirements and achievements
        req_snrs = [u["required_snr_dB"] for u in users]
        ach_snrs = [u["achieved_snr_dB"] for u in users]
        delta_snrs = [u["delta_snr_dB"] for u in users]
        
        desc_parts.append(f"Required SNR range: {min(req_snrs):.1f} to {max(req_snrs):.1f} dB.")
        desc_parts.append(f"Achieved SNR range: {min(ach_snrs):.1f} to {max(ach_snrs):.1f} dB.")
        desc_parts.append(f"Delta SNR range: {min(delta_snrs):.1f} to {max(delta_snrs):.1f} dB.")
        
        # User distribution
        distances = [np.sqrt(u["coord"][0]**2 + u["coord"][1]**2) for u in users]
        desc_parts.append(f"User distances: {min(distances):.1f} to {max(distances):.1f} meters.")
        
        # Channel conditions
        fading_types = [u.get("csi", {}).get("fading", "unknown") for u in users]
        rician_count = fading_types.count("Rician")
        rayleigh_count = fading_types.count("Rayleigh")
        desc_parts.append(f"{rician_count} Rician channels, {rayleigh_count} Rayleigh channels.")
        
        return " ".join(desc_parts)
    
    def _extract_numerical_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features for similarity matching."""
        case_data = input_data["case_data"]
        users = case_data["users"]
        
        features = []
        
        # Basic counts
        features.append(len(users))  # Number of users
        features.append(sum(1 for u in users if u.get("csi", {}).get("los", False)))  # LoS count
        
        # SNR statistics
        req_snrs = [u["required_snr_dB"] for u in users]
        ach_snrs = [u["achieved_snr_dB"] for u in users]
        delta_snrs = [u["delta_snr_dB"] for u in users]
        
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
    
    def find_similar_scenarios(self, current_scenario: Dict[str, Any], 
                             max_results: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find scenarios similar to the current one using hybrid similarity search.
        Returns list of (scenario, similarity_score) tuples.
        """
        if not self.dataset or self.semantic_index is None:
            return []
        
        if max_results is None:
            max_results = RAG_CONFIG["max_retrieved_scenarios"]
        
        # Get semantic similarity
        current_desc = self._create_scenario_description(current_scenario)
        current_embedding = self.embedding_model.encode([current_desc])
        faiss.normalize_L2(current_embedding)
        
        semantic_scores, semantic_indices = self.semantic_index.search(
            current_embedding.astype('float32'), max_results
        )
        
        # Get numerical similarity
        current_features = np.array([self._extract_numerical_features(current_scenario)])
        normalized_features = self.feature_scaler.transform(current_features)
        
        numerical_distances, numerical_indices = self.numerical_index.search(
            normalized_features.astype('float32'), max_results
        )
        
        # Combine similarities with weights
        combined_results = {}
        weights = RAG_CONFIG["feature_weights"]
        
        # Process semantic results
        for score, idx in zip(semantic_scores[0], semantic_indices[0]):
            if idx < len(self.dataset):
                combined_score = weights["channel_conditions"] * score
                combined_results[idx] = combined_results.get(idx, 0) + combined_score
        
        # Process numerical results (convert distance to similarity)
        max_distance = max(numerical_distances[0]) if len(numerical_distances[0]) > 0 else 1.0
        for distance, idx in zip(numerical_distances[0], numerical_indices[0]):
            if idx < len(self.dataset):
                similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
                location_weight = weights["user_locations"] + weights["delta_snr_values"]
                combined_score = location_weight * similarity
                combined_results[idx] = combined_results.get(idx, 0) + combined_score
        
        # Sort by combined score and return top results
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_results[:max_results]:
            if score >= RAG_CONFIG["similarity_threshold"]:
                results.append((self.dataset[idx], score))
        
        return results
    
    def extract_algorithm_patterns(self, similar_scenarios: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Extract algorithm selection patterns from similar scenarios."""
        if not similar_scenarios:
            return {"recommended_algorithm": "manifold", "confidence": 0.0, "reasoning": "No similar scenarios found"}
        
        # Count algorithm preferences
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
        reasoning_parts = [f"Based on {len(similar_scenarios)} similar scenarios:"]
        for algo, vote in sorted(algorithm_votes.items(), key=lambda x: x[1], reverse=True):
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
            return {"selection_strategy": "select_all_negative", "reasoning": "No patterns available"}
        
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
            
            pattern = {
                "similarity": similarity,
                "selected_delta_range": (min(selected_delta_snrs), max(selected_delta_snrs)) if selected_delta_snrs else None,
                "unselected_delta_range": (min(unselected_delta_snrs), max(unselected_delta_snrs)) if unselected_delta_snrs else None,
                "selection_threshold": max(selected_delta_snrs) if selected_delta_snrs else None
            }
            selection_patterns.append(pattern)
        
        # Analyze patterns
        thresholds = [p["selection_threshold"] for p in selection_patterns if p["selection_threshold"] is not None]
        avg_threshold = np.mean(thresholds) if thresholds else 0.0
        
        reasoning = f"Historical patterns suggest selecting users with delta SNR below {avg_threshold:.1f} dB"
        
        return {
            "selection_strategy": "learned_threshold",
            "threshold": avg_threshold,
            "reasoning": reasoning,
            "pattern_confidence": len(thresholds) / len(similar_scenarios) if similar_scenarios else 0.0
        }


# Export the RAG system
__all__ = ['RAGMemorySystem']