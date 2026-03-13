import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class UWBHardwareProfile:
    name: str
    tx_current_mA: float
    rx_current_mA: float
    idle_current_mA: float
    notes: str = ""

class HardwareProfileManager:
    _profiles: Dict[str, UWBHardwareProfile] = {}

    @classmethod
    def load_profiles(cls, json_path: Optional[str] = None) -> Dict[str, UWBHardwareProfile]:
        if not json_path:
            json_path = os.path.join(os.path.dirname(__file__), "uwb_hardware_profiles.json")
        
        if not os.path.exists(json_path):
            return cls._profiles
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        cls._profiles.clear()
        for name, params in data.items():
            cls._profiles[name] = UWBHardwareProfile(
                name=name,
                tx_current_mA=params.get("tx_current_mA", 70.0),
                rx_current_mA=params.get("rx_current_mA", 110.0),
                idle_current_mA=params.get("idle_current_mA", 12.0),
                notes=params.get("notes", "")
            )
        return cls._profiles

    @classmethod
    def get_profile(cls, name: str) -> Optional[UWBHardwareProfile]:
        if not cls._profiles:
            cls.load_profiles()
        return cls._profiles.get(name)

    @classmethod
    def get_all_profile_names(cls) -> List[str]:
        if not cls._profiles:
            cls.load_profiles()
        return list(cls._profiles.keys())
