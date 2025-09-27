#!/usr/bin/env python3
"""
Required SNR Generator for RIS Framework
Calculates realistic SNR values for all scenarios using channel conditions.
Usage: python generate_required_snr.py
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Import the required modules
from multiuserdatasetgenerator import RISDatasetGenerator
from scenario import CASES
from config import FRAMEWORK_CONFIG

class RequiredSNRGenerator:
    """Generator for calculating and managing required SNR values."""
    
    def __init__(self):
        self.dataset_generator = RISDatasetGenerator()
        self.sim_settings = FRAMEWORK_CONFIG.get("sim_settings", {})
        self.applications = FRAMEWORK_CONFIG.get("snr_calculation", {}).get("applications", {})
        self.generated_values = {}
        
    def calculate_all_scenario_snrs(self) -> Dict[str, Dict]:
        """Calculate required SNR for all users in all scenarios."""
        print("=== Calculating Required SNR Values for All Scenarios ===\n")
        
        results = {}
        
        for scenario_name, scenario_data in CASES.items():
            print(f"Processing scenario: {scenario_name}")
            scenario_results = {
                "num_users": scenario_data["num_users"],
                "users": []
            }
            
            for user in scenario_data["users"]:
                user_id = user["id"]
                user_coord = user["coord"]
                user_csi = user["csi"]
                original_app = user.get("app", "web_browsing")  # fallback if no app
                
                # Map the application name to standardized format
                app_type = self._map_application_name(original_app)
                
                try:
                    # Calculate required SNR using the function from multiuserdatasetgenerator
                    calculated_snr = self.dataset_generator.calculate_required_snr(
                        app_type=app_type,
                        user_coord=user_coord,
                        csi=user_csi
                    )
                    
                    user_result = {
                        "id": user_id,
                        "coord": user_coord,
                        "app": original_app,
                        "app_mapped": app_type,
                        "csi": user_csi,
                        "original_req_snr_dB": user.get("req_snr_dB", None),
                        "calculated_req_snr_dB": round(calculated_snr, 2),
                        "base_snr_dB": self.applications[app_type]["base_snr_dB"],
                        "additional_margin_dB": round(calculated_snr - self.applications[app_type]["base_snr_dB"], 2)
                    }
                    
                    scenario_results["users"].append(user_result)
                    
                    print(f"  User {user_id}: {original_app} -> {calculated_snr:.2f} dB "
                          f"(base: {self.applications[app_type]['base_snr_dB']} dB + "
                          f"margin: {calculated_snr - self.applications[app_type]['base_snr_dB']:.2f} dB)")
                    
                except Exception as e:
                    print(f"  Error calculating SNR for User {user_id}: {e}")
                    # Use original value as fallback
                    user_result = {
                        "id": user_id,
                        "coord": user_coord,
                        "app": original_app,
                        "app_mapped": app_type,
                        "csi": user_csi,
                        "original_req_snr_dB": user.get("req_snr_dB", None),
                        "calculated_req_snr_dB": user.get("req_snr_dB", 10),  # fallback
                        "error": str(e)
                    }
                    scenario_results["users"].append(user_result)
            
            results[scenario_name] = scenario_results
            print(f"  Completed {scenario_name} with {len(scenario_results['users'])} users\n")
        
        self.generated_values = results
        return results
    
    def _map_application_name(self, app_name: str) -> str:
        """Map scenario application names to standardized APPLICATIONS keys."""
        app_mapping = {
            # Direct matches
            "web_browsing": "web_browsing",
            "video_call": "video_call", 
            "voice call": "video_call",  # treat voice as video call
            "hd_streaming": "hd_streaming",
            "hd video": "hd_streaming",
            "online_gaming": "online_gaming",
            "4k_streaming": "4k_streaming",
            "4k video": "4k_streaming",
            "ar_vr": "ar_vr",
            "ar/vr": "ar_vr",
            
            # Additional mappings
            "audio streaming": "web_browsing",  # treat as low-demand
            "voice": "video_call"
        }
        
        # Normalize the input (lowercase, strip spaces)
        normalized_app = app_name.lower().strip()
        
        # Return mapped application or default to web_browsing
        return app_mapping.get(normalized_app, "web_browsing")
    
    def save_snr_values(self, filename: str = "required_snr_values.json"):
        """Save calculated SNR values to a JSON file."""
        if not self.generated_values:
            print("No SNR values to save. Run calculate_all_scenario_snrs() first.")
            return
        
        # Add metadata to the saved file
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_scenarios": len(self.generated_values),
                "framework_config_used": {
                    "sim_settings": self.sim_settings,
                    "applications": self.applications
                }
            },
            "scenarios": self.generated_values
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ SNR values saved to {filename}")
        except Exception as e:
            print(f"✗ Error saving SNR values: {e}")
    
    def update_config_file(self, config_filename: str = "config.py"):
        """Update the config.py file with calculated required SNR values."""
        if not self.generated_values:
            print("No SNR values to update. Run calculate_all_scenario_snrs() first.")
            return
        
        try:
            # Read the current config file
            with open(config_filename, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Create backup
            backup_filename = f"{config_filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(backup_filename, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print(f"✓ Config backup saved to {backup_filename}")
            
            # Create the required SNR configuration section
            snr_config_section = self._generate_snr_config_section()
            
            # Find the insertion point (before the closing brace of FRAMEWORK_CONFIG)
            # Look for the last closing brace in the FRAMEWORK_CONFIG section
            insert_point = config_content.rfind('}')
            
            if insert_point == -1:
                print("✗ Could not find insertion point in config file")
                return
            
            # Insert the SNR config section
            updated_content = (
                config_content[:insert_point] + 
                snr_config_section + 
                config_content[insert_point:]
            )
            
            # Write the updated config
            with open(config_filename, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ Config file {config_filename} updated with required SNR values")
            
        except Exception as e:
            print(f"✗ Error updating config file: {e}")
    
    def _generate_snr_config_section(self) -> str:
        """Generate the required SNR configuration section as a string."""
        config_lines = [
            '    # Required SNR values calculated using calculate_required_snr function',
            f'    # Generated on: {datetime.now().isoformat()}',
            '    "required_snr_values": {'
        ]
        
        for scenario_name, scenario_data in self.generated_values.items():
            config_lines.append(f'        "{scenario_name}": {{')
            config_lines.append(f'            "num_users": {scenario_data["num_users"]},')
            config_lines.append('            "users": {')
            
            for user in scenario_data["users"]:
                user_id = user["id"]
                req_snr = user["calculated_req_snr_dB"]
                config_lines.append(f'                {user_id}: {req_snr},')
            
            config_lines.append('            }')
            config_lines.append('        },')
        
        config_lines.append('    },')
        config_lines.append('')
        
        return '\n'.join(config_lines)
    
    def print_summary(self):
        """Print a summary of calculated SNR values."""
        if not self.generated_values:
            print("No SNR values calculated yet.")
            return
        
        print("\n=== Required SNR Calculation Summary ===")
        
        total_users = 0
        snr_ranges = []
        
        for scenario_name, scenario_data in self.generated_values.items():
            users = scenario_data["users"]
            total_users += len(users)
            
            snr_values = [u["calculated_req_snr_dB"] for u in users]
            snr_ranges.extend(snr_values)
            
            min_snr = min(snr_values)
            max_snr = max(snr_values)
            avg_snr = sum(snr_values) / len(snr_values)
            
            print(f"{scenario_name:8s}: {len(users)} users, "
                  f"SNR range: {min_snr:5.1f} - {max_snr:5.1f} dB, "
                  f"avg: {avg_snr:5.1f} dB")
        
        overall_min = min(snr_ranges)
        overall_max = max(snr_ranges)
        overall_avg = sum(snr_ranges) / len(snr_ranges)
        
        print(f"\nOverall: {total_users} users, "
              f"SNR range: {overall_min:.1f} - {overall_max:.1f} dB, "
              f"avg: {overall_avg:.1f} dB")
        
        # Show application distribution
        app_counts = {}
        for scenario_data in self.generated_values.values():
            for user in scenario_data["users"]:
                app = user["app_mapped"]
                app_counts[app] = app_counts.get(app, 0) + 1
        
        print(f"\nApplication distribution:")
        for app, count in sorted(app_counts.items()):
            percentage = count / total_users * 100
            print(f"  {app:15s}: {count:2d} users ({percentage:4.1f}%)")


def main():
    """Main function to run the SNR calculation process."""
    print("RIS Framework - Required SNR Generator")
    print("=" * 45)
    
    # Initialize the generator
    generator = RequiredSNRGenerator()
    
    # Calculate SNR values for all scenarios
    results = generator.calculate_all_scenario_snrs()
    
    # Print summary
    generator.print_summary()
    
    # Save results to JSON file
    generator.save_snr_values("required_snr_values.json")
    
    print(f"\n✓ Required SNR calculation completed!")
    print(f"✓ Results saved to required_snr_values.json")
    print(f"✓ Scenarios will automatically use the updated JSON values")
    
    return results


if __name__ == "__main__":
    main()