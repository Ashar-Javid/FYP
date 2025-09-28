import numpy as np
import matplotlib.pyplot as plt
import json
import os
from config import FRAMEWORK_CONFIG

def get_required_snr_from_json(scenario_name, user_id, json_file="required_snr_values.json"):
    """
    Get required SNR for a user from the JSON file.
    
    Args:
        scenario_name: Name of the scenario (e.g., "5U_A")
        user_id: ID of the user (integer)
        json_file: Path to the JSON file with SNR values
    
    Returns:
        Required SNR in dB, or None if not found
    """
    try:
        if not os.path.exists(json_file):
            return None
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        scenarios = data.get("scenarios", {})
        scenario_data = scenarios.get(scenario_name, {})
        users = scenario_data.get("users", [])
        
        # Find user by ID
        for user in users:
            if user.get("id") == user_id:
                return user.get("calculated_req_snr_dB", None)
        
        return None
    except Exception as e:
        print(f"Warning: Could not get required SNR for {scenario_name} user {user_id}: {e}")
        return None

def compute_distance(u_coord, bs_coord):
    """Compute 3D distance between BS and user."""
    return np.linalg.norm(np.array(u_coord) - np.array(bs_coord))

def pathloss(d, PL0_dB=30, gamma=3.5):
    """Pathloss model in dB given distance d (m)."""
    return PL0_dB + 10 * gamma * np.log10(d)

def generate_csi(user, bs_coord, ris_coord, sim_settings):
    """
    Generate instantaneous CSI including LoS/NLoS, blockage, fading, RIS phase effect.
    """
    d_bs = compute_distance(user["coord"], bs_coord)  # BS-user distance
    d_ris = compute_distance(user["coord"], ris_coord)  # RIS-user distance
    
    # Pathloss values
    pl_bs = pathloss(d_bs, sim_settings["PL0_dB"], sim_settings["gamma"])
    pl_ris = pathloss(d_ris, sim_settings["PL0_dB"], sim_settings["gamma"])
    
    # Fading model
    if user["csi"]["fading"] == "Rician":
        K_linear = 10**(user["csi"]["K_factor_dB"]/10)
        # LoS component
        h_los = np.exp(1j * 2 * np.pi * np.random.rand())
        # NLoS component
        h_nlos = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
        h = np.sqrt(K_linear/(K_linear+1))*h_los + np.sqrt(1/(K_linear+1))*h_nlos
    else:  # Rayleigh
        h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
    
    # Blockage handling
    if user["csi"]["blockage"] == "blocked":
        h *= 0.1  # attenuate by 10x for blockage
    
    # RIS phase shift effect (placeholder: random phase)
    ris_phase = np.exp(1j * 2 * np.pi * np.random.rand(sim_settings["ris_elements"]))
    h_ris = np.sum(ris_phase) / np.sqrt(sim_settings["ris_elements"])  # effective RIS gain
    
    # Combined channel gain
    h_eff = h * 10**(-pl_bs/20) + h_ris * 10**(-pl_ris/20)
    
    return {
        "h_eff": h_eff,
        "h_magnitude_dB": 20*np.log10(abs(h_eff) + 1e-9),
        "pathloss_bs_dB": pl_bs,
        "pathloss_ris_dB": pl_ris
    }

def calculate_snr(h_eff, bs_power_dBm, noise_power_dBm):
    """
    Compute instantaneous SNR given effective channel and BS power.
    """
    bs_power_linear = 10 ** ((bs_power_dBm - 30) / 10)  # dBm to Watts
    noise_linear = 10 ** ((noise_power_dBm - 30) / 10)  # dBm to Watts
    snr = (abs(h_eff)**2 * bs_power_linear) / noise_linear
    return 10 * np.log10(snr + 1e-9)


# ==========================
# CASE 1: 3 Users (Mixed LoS/NLoS, Different Apps)
# ==========================
CASE_3U = {
    "num_users": 3,
    "users": [
        {"id": 1, "coord": (30, 20, 1.5), "req_snr_dB": get_required_snr_from_json("3U", 1) or 15, "app": "HD video",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 10}},
        {"id": 2, "coord": (70, -10, 1.5), "req_snr_dB": get_required_snr_from_json("3U", 2) or 3, "app": "Voice call",
         "csi": {"blockage": "non_blocked", "los": "NLoS", "fading": "Rician", "K_factor_dB": 6}},
        {"id": 3, "coord": (120, 40, 1.5), "req_snr_dB": get_required_snr_from_json("3U", 3) or 20, "app": "4K video",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}}
    ]
}

# ==========================
# CASE 2: 4 Users (Clustered vs. Distant, Different Apps)
# ==========================
CASE_4U = {
    "num_users": 4,
    "users": [
        {"id": 1, "coord": (60, 20, 1.5), "req_snr_dB": get_required_snr_from_json("4U", 1) or 10, "app": "Online gaming",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 8}},
        {"id": 2, "coord": (65, 25, 1.5), "req_snr_dB": get_required_snr_from_json("4U", 2) or 5, "app": "Audio streaming",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 8}},
        {"id": 3, "coord": (200, -40, 1.5), "req_snr_dB": get_required_snr_from_json("4U", 3) or 25, "app": "AR/VR",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 4, "coord": (220, -60, 1.5), "req_snr_dB": get_required_snr_from_json("4U", 4) or 7, "app": "Web browsing",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}}
    ]
}

# ==========================
# CASE 3: 5 Users (2 Near BS, 3 Far, Mixed Demands)
# ==========================
CASE_5U_A = {
    "num_users": 5,
    "users": [
        {"id": 1, "coord": (20, 5, 1.5), "req_snr_dB": get_required_snr_from_json("5U_A", 1) or 25, "app": "AR/VR",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 12}},
        {"id": 2, "coord": (30, -10, 1.5), "req_snr_dB": get_required_snr_from_json("5U_A", 2) or 3, "app": "Voice call",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 12}},
        {"id": 3, "coord": (200, 40, 1.5), "req_snr_dB": get_required_snr_from_json("5U_A", 3) or 20, "app": "4K video",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 4, "coord": (250, -60, 1.5), "req_snr_dB": get_required_snr_from_json("5U_A", 4) or 15, "app": "HD video",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 5, "coord": (300, 30, 1.5), "req_snr_dB": get_required_snr_from_json("5U_A", 5) or 5, "app": "Audio streaming",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}}
    ]
}

# ==========================
# CASE 4: 5 Users (Mobility Effects, Mixed Apps)
# ==========================
CASE_5U_B = {
    "num_users": 5,
    "users": [
        {"id": 1, "coord": (50, 10, 1.5), "req_snr_dB": get_required_snr_from_json("5U_B", 1) or 7, "app": "Web browsing",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 7, "temporal_autocorr": 0.9}},
        {"id": 2, "coord": (80, -20, 1.5), "req_snr_dB": get_required_snr_from_json("5U_B", 2) or 15, "app": "HD video",
         "csi": {"blockage": "non_blocked", "los": "LoS", "fading": "Rician", "K_factor_dB": 7, "temporal_autocorr": 0.9}},
        {"id": 3, "coord": (120, 30, 1.5), "req_snr_dB": get_required_snr_from_json("5U_B", 3) or 10, "app": "Online gaming",
         "csi": {"blockage": "non_blocked", "los": "NLoS", "fading": "Rician", "K_factor_dB": 4, "temporal_autocorr": 0.6}},
        {"id": 4, "coord": (160, -40, 1.5), "req_snr_dB": get_required_snr_from_json("5U_B", 4) or 20, "app": "4K video",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh", "temporal_autocorr": 0.4}},
        {"id": 5, "coord": (200, 50, 1.5), "req_snr_dB": get_required_snr_from_json("5U_B", 5) or 25, "app": "AR/VR",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh", "temporal_autocorr": 0.4}}
    ]
}

# ==========================
# CASE 5: 5 Users (Cell-edge, Demanding Apps)
# ==========================
CASE_5U_C = {
    "num_users": 5,
    "users": [
        {"id": 1, "coord": (250, 0, 1.5), "req_snr_dB": get_required_snr_from_json("5U_C", 1) or 20, "app": "4K video",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 2, "coord": (300, 20, 1.5), "req_snr_dB": get_required_snr_from_json("5U_C", 2) or 25, "app": "AR/VR",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 3, "coord": (320, -30, 1.5), "req_snr_dB": get_required_snr_from_json("5U_C", 3) or 10, "app": "Online gaming",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 4, "coord": (350, -50, 1.5), "req_snr_dB": get_required_snr_from_json("5U_C", 4) or 15, "app": "HD video",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}},
        {"id": 5, "coord": (380, 60, 1.5), "req_snr_dB": get_required_snr_from_json("5U_C", 5) or 5, "app": "Audio streaming",
         "csi": {"blockage": "blocked", "los": "NLoS", "fading": "Rayleigh"}}
    ]
}



SIM_SETTINGS = {
    **FRAMEWORK_CONFIG.get("sim_settings", {}),
    "base_station_power_dBm_range": FRAMEWORK_CONFIG.get("power_range_dB", [20,45]),
    "default_bs_power_dBm": FRAMEWORK_CONFIG.get("default_bs_power_dBm", 20),
}


CASES = {
    "3U": CASE_3U,
    "4U": CASE_4U,
    "5U_A": CASE_5U_A,
    "5U_B": CASE_5U_B,
    "5U_C": CASE_5U_C,
}
def run_cases_and_plot(cases, sim_settings):
    np.random.seed(sim_settings["seed"])
    
    for case_name, case in cases.items():
        user_ids = []
        snr_vals = []
        req_snrs = []
        apps = []
        user_coords = []
        csi_conditions = []
        
        for user in case["users"]:
            csi_out = generate_csi(user, sim_settings["bs_coord"], sim_settings["ris_coord"], sim_settings)
            snr_out = calculate_snr(csi_out["h_eff"], sim_settings["default_bs_power_dBm"], sim_settings["noise_power_dBm"])
            
            user_ids.append(f"U{user['id']} ({user['app']})")
            snr_vals.append(snr_out)
            req_snrs.append(user["req_snr_dB"])
            apps.append(user["app"])
            
            # Extract 2D coordinates for plotting
            user_coords.append((user["coord"][0], user["coord"][1]))
            csi_conditions.append(user["csi"]["los"])
            
            print(f"{case_name} | User {user['id']} ({user['app']}) | "
                  f"Req = {user['req_snr_dB']} dB | Achieved = {snr_out:.2f} dB")
        
        # Plot bar chart for SNR comparison
        x = np.arange(len(user_ids))
        plt.figure(figsize=(8, 5))
        plt.bar(x - 0.2, snr_vals, width=0.4, label="Achieved SNR")
        plt.bar(x + 0.2, req_snrs, width=0.4, label="Required SNR")
        plt.xticks(x, user_ids)
        plt.ylabel("SNR (dB)")
        plt.title(f"Case {case_name}: Required vs Achieved SNR")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"plots/{case_name}_SNR_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot user setup with channel conditions
        bs_pos = sim_settings["bs_coord"][:2]  # 2D coordinates
        ris_pos = sim_settings["ris_coord"][:2]  # 2D coordinates
        plot_user_setup(bs_pos, ris_pos, user_coords, req_snrs, snr_vals, csi_conditions, 
                        f"Case {case_name}: User Placement and Channel Conditions", case_name)

def plot_user_setup(bs_pos, ris_pos, users, snr_req, snr_ach, csi_conditions, title, case_name):
    plt.figure(figsize=(8, 6))

    # Plot BS and RIS
    plt.scatter(*bs_pos, marker='s', s=200, c='black', label='Base Station')
    plt.scatter(*ris_pos, marker='^', s=200, c='blue', label='RIS')

    # Plot Users
    for i, user in enumerate(users):
        color = 'green' if snr_ach[i] >= snr_req[i] else ('orange' if snr_ach[i] >= snr_req[i]-2 else 'red')
        plt.scatter(*user, c=color, s=100, label=f'User {i+1}' if i==0 else None)

        # Annotate with SNR info
        plt.text(user[0]+1, user[1]+1, f"Req={snr_req[i]} dB\nAch={snr_ach[i]} dB", fontsize=8, color=color)

        # Plot channel condition line (BS-user)
        if csi_conditions[i] == "LoS":
            plt.plot([bs_pos[0], user[0]], [bs_pos[1], user[1]], 'g-', linewidth=1.5)
        elif csi_conditions[i] == "NLoS":
            plt.plot([bs_pos[0], user[0]], [bs_pos[1], user[1]], 'b-', linewidth=1.2)
        else:  # Blocked
            plt.plot([bs_pos[0], user[0]], [bs_pos[1], user[1]], 'r--', linewidth=1.2)        # Plot RIS-user (always adjustable, so dotted)
        plt.plot([ris_pos[0], user[0]], [ris_pos[1], user[1]], 'k:', linewidth=1)
    
    plt.title(title)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{case_name}_user_setup.png", dpi=300, bbox_inches='tight')
    plt.close()

#run_cases_and_plot(CASES, SIM_SETTINGS)