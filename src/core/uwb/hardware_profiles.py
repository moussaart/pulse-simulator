import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class UWBProfile:
    name: str
    energy_tx_uJ: float
    energy_rx_uJ: float
    power_idle_mW: float
    power_sleep_mW: float
    notes: str = ""

@dataclass
class IMUProfile:
    name: str
    energy_active_uJ_per_sample: float
    power_sleep_mW: float
    sample_rate_hz: float = 100.0
    notes: str = ""

class DeviceProfileManager:
    _uwb_profiles: Dict[str, UWBProfile] = {}
    _imu_profiles: Dict[str, IMUProfile] = {}
    _json_path: str = ""

    @classmethod
    def load_profiles(cls, json_path: Optional[str] = None) -> None:
        if not json_path:
            json_path = os.path.join(os.path.dirname(__file__), "device_profiles.json")
        cls._json_path = json_path
        
        if not os.path.exists(json_path):
            return
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        cls._uwb_profiles.clear()
        for name, params in data.get("uwb_profiles", {}).items():
            cls._uwb_profiles[name] = UWBProfile(
                name=name,
                energy_tx_uJ=params.get("energy_tx_uJ", 46.2),
                energy_rx_uJ=params.get("energy_rx_uJ", 108.9),
                power_idle_mW=params.get("power_idle_mW", 19.8),
                power_sleep_mW=params.get("power_sleep_mW", 0.00033),
                notes=params.get("notes", "")
            )

        cls._imu_profiles.clear()
        for name, params in data.get("imu_profiles", {}).items():
            cls._imu_profiles[name] = IMUProfile(
                name=name,
                energy_active_uJ_per_sample=params.get("energy_active_uJ_per_sample", 34.6),
                power_sleep_mW=params.get("power_sleep_mW", 0.0198),
                sample_rate_hz=params.get("sample_rate_hz", 100.0),
                notes=params.get("notes", "")
            )

    @classmethod
    def save_profiles(cls, json_path: Optional[str] = None) -> None:
        if not json_path:
            json_path = cls._json_path
        if not json_path:
            json_path = os.path.join(os.path.dirname(__file__), "device_profiles.json")
            
        data = {
            "uwb_profiles": {},
            "imu_profiles": {}
        }
        
        for name, profile in cls._uwb_profiles.items():
            data["uwb_profiles"][name] = {
                "energy_tx_uJ": profile.energy_tx_uJ,
                "energy_rx_uJ": profile.energy_rx_uJ,
                "power_idle_mW": profile.power_idle_mW,
                "power_sleep_mW": profile.power_sleep_mW,
                "notes": profile.notes
            }
            
        for name, profile in cls._imu_profiles.items():
            data["imu_profiles"][name] = {
                "energy_active_uJ_per_sample": profile.energy_active_uJ_per_sample,
                "power_sleep_mW": profile.power_sleep_mW,
                "sample_rate_hz": profile.sample_rate_hz,
                "notes": profile.notes
            }
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def get_uwb_profile(cls, name: str) -> Optional[UWBProfile]:
        if not cls._uwb_profiles:
            cls.load_profiles()
        return cls._uwb_profiles.get(name)

    @classmethod
    def get_imu_profile(cls, name: str) -> Optional[IMUProfile]:
        if not cls._imu_profiles:
            cls.load_profiles()
        return cls._imu_profiles.get(name)

    @classmethod
    def get_all_uwb_names(cls) -> List[str]:
        if not cls._uwb_profiles:
            cls.load_profiles()
        return list(cls._uwb_profiles.keys())

    @classmethod
    def get_all_imu_names(cls) -> List[str]:
        if not cls._imu_profiles:
            cls.load_profiles()
        return list(cls._imu_profiles.keys())
        
    @classmethod
    def add_uwb_profile(cls, profile: UWBProfile) -> None:
        if not cls._uwb_profiles:
            cls.load_profiles()
        cls._uwb_profiles[profile.name] = profile
        cls.save_profiles()
        
    @classmethod
    def add_imu_profile(cls, profile: IMUProfile) -> None:
        if not cls._imu_profiles:
            cls.load_profiles()
        cls._imu_profiles[profile.name] = profile
        cls.save_profiles()
        
    @classmethod
    def delete_uwb_profile(cls, name: str) -> bool:
        if not cls._uwb_profiles:
            cls.load_profiles()
        if name in cls._uwb_profiles:
            del cls._uwb_profiles[name]
            cls.save_profiles()
            return True
        return False

    @classmethod
    def delete_imu_profile(cls, name: str) -> bool:
        if not cls._imu_profiles:
            cls.load_profiles()
        if name in cls._imu_profiles:
            del cls._imu_profiles[name]
            cls.save_profiles()
            return True
        return False
