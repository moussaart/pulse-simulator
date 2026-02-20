import csv
import os
from typing import Dict, List, Optional
from collections import defaultdict
from src.core.uwb.uwb_types import SVModelParams, EnvironmentConfig

def load_channel_configs(csv_path: str) -> Dict[str, EnvironmentConfig]:
    """
    Load UWB channel configurations from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        A dictionary mapping environment names (e.g., "Residential") to EnvironmentConfig objects
        containing both LOS and NLOS parameters.
    """
    env_groups = defaultdict(dict)
    
    if not os.path.exists(csv_path):
        print(f"Warning: Config file not found at {csv_path}")
        return {}
        
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # Parse identifiers
                    env_name = row.get('Environment', '').strip()
                    condition = row.get('Condition', '').strip().upper() # LOS or NLOS
                    
                    if not env_name:
                        continue
                        
                    # Parse numerical values
                    # 1. Cluster Arrival Rate (Lambda) - input in 1/ns -> convert to 1/s
                    # stored as 'Lambda_cluster_1_per_ns'
                    lambda_cluster_ns = float(row.get('Lambda_cluster_1_per_ns', 0))
                    cluster_arrival_rate = lambda_cluster_ns * 1e9
                    
                    # 2. Ray Arrival Rate (lambda1) - input in 1/ns -> convert to 1/s
                    lambda1_ns = float(row.get('lambda1_1_per_ns', 0))
                    ray_arrival_rate = lambda1_ns * 1e9
                    
                    # 3. Ray Decay (gamma) - input is lambda2 (1/ns), so gamma = 1/lambda2
                    # stored as 'lambda2_1_per_ns'
                    lambda2_ns = float(row.get('lambda2_1_per_ns', 0))
                    if lambda2_ns > 1e-6:
                        # gamma (ns) = 1 / lambda2
                        # gamma (s) = gamma (ns) * 1e-9
                        ray_decay = (1.0 / lambda2_ns) * 1e-9
                    else:
                        ray_decay = 50e-9 # Default large decay if rate is 0
                        
                    # 4. Cluster Decay (Gamma) - input in ns -> convert to s
                    gamma_ns = float(row.get('Gamma_ns', 0))
                    cluster_decay = gamma_ns * 1e-9
                    
                    # 5. Path Loss Exponent (n)
                    n = float(row.get('n', 2.0))
                    
                    # 6. Shadow Fading (SigmaS) - dB
                    sigma_s = float(row.get('SigmaS_dB', 0))
                    
                    # 7. RMS Delay Spread (estimated based on condition if not present)
                    # Use standard IEEE defaults if not specified
                    if condition == 'LOS':
                        rms_delay = 5e-9 
                    else:
                        rms_delay = 20e-9
                    
                    # Create param object
                    params = SVModelParams(
                        cluster_arrival_rate=cluster_arrival_rate,
                        ray_arrival_rate=ray_arrival_rate,
                        cluster_decay=cluster_decay,
                        ray_decay=ray_decay,
                        path_loss_exponent=n,
                        shadow_fading_std=sigma_s,
                        rms_delay_spread=rms_delay
                    )
                    
                    # Store in groups
                    env_groups[env_name][condition] = params
                    
                except ValueError as e:
                    print(f"Skipping invalid row in channel config: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error loading channel configs: {e}")
        return {}
        
    # Construct EnvironmentConfig objects
    final_configs = {}
    
    for env_name, variants in env_groups.items():
        # We need both LOS and NLOS to make a valid EnvironmentConfig
        # If one is missing, we can duplicate the other or use defaults, 
        # but ideally we have both.
        
        los_params = variants.get('LOS')
        nlos_params = variants.get('NLOS')
        
        if los_params and nlos_params:
            config = EnvironmentConfig(
                name=env_name,
                los_params=los_params,
                nlos_params=nlos_params
            )
            final_configs[env_name] = config
        elif los_params:
            # Only LOS exists, assume NLOS is similar but worse? 
            # Or just duplicate for safety to avoid crashes
            # User request "Only LOS case" might imply some environments only have LOS in DB?
            # But checking CSV, all seem to have pairs.
            print(f"Warning: Environment '{env_name}' missing NLOS params. Using LOS for both.")
            config = EnvironmentConfig(
                name=env_name,
                los_params=los_params,
                nlos_params=los_params # Duplicate
            )
            final_configs[env_name] = config
        elif nlos_params:
            print(f"Warning: Environment '{env_name}' missing LOS params. Using NLOS for both.")
            config = EnvironmentConfig(
                name=env_name,
                los_params=nlos_params, # Duplicate
                nlos_params=nlos_params
            )
            final_configs[env_name] = config
            
    return final_configs
