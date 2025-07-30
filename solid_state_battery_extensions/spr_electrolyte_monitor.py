"""SPR电解质监测模块"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
from pathlib import Path
import sys

# 导入现有optics模块
sys.path.append(str(Path(__file__).parent.parent / "Multiscale-Photonic-Diagnostics-for-Solid-State-Batteries-main"))
from optics.elements import MetalFilm
from optics.materials import gold_n_complex_scalar
from optics.ray import Ray

from .electrolyte_materials import ElectrolyteMaterials, ElectrolyteType


@dataclass
class SPRMeasurement:
    """SPR测量数据结构"""
    timestamp: float
    incident_angle_deg: float
    wavelength_nm: float
    reflectance: float
    phase_shift_rad: float
    field_enhancement: float


@dataclass
class BeamParameters:
    """光束参数"""
    wavelength_nm: float = 632.8  # He-Ne激光
    power_mw: float = 1.0
    beam_diameter_mm: float = 1.0
    polarization: str = "TM"  # SPR需要TM偏振


class SPRElectrolyteMonitor:
    """SPR电解质监测器"""
    
    def __init__(
        self,
        electrolyte_type: ElectrolyteType,
        electrolyte_material: str,
        metal_film_thickness_nm: float = 50.0,
        prism_n: float = 1.515,  # BK7玻璃
        materials_db: Optional[ElectrolyteMaterials] = None
    ):
        """初始化SPR电解质监测器"""
        self.electrolyte_type = electrolyte_type
        self.electrolyte_material = electrolyte_material
        self.materials_db = materials_db or ElectrolyteMaterials()
        
        # 获取电解质属性
        self.electrolyte_props = self.materials_db.get_material(
            electrolyte_type, electrolyte_material
        )
        
        # 光束参数
        self.beam_params = BeamParameters()
        
        # SPR配置
        self.prism_n = prism_n
        self.metal_film_thickness = metal_film_thickness_nm
        
        # 角度扫描范围
        self.angle_range_deg = (40.0, 75.0)
        self.angle_resolution_deg = 0.1
        
        # 测量历史
        self.measurement_history: List[SPRMeasurement] = []
        
        # 初始化SPR系统
        self._setup_spr_system()
    
    def _setup_spr_system(self):
        """设置SPR系统"""
        # 获取电解质折射率
        self.electrolyte_n = self.materials_db.get_refractive_index(
            self.electrolyte_type, 
            self.electrolyte_material,
            self.beam_params.wavelength_nm
        )
        
        # 创建金属膜元件
        self.metal_film = MetalFilm(
            thickness_nm=self.metal_film_thickness,
            n_metal=gold_n_complex_scalar,
            n_prism=self.prism_n,
            n_sample=self.electrolyte_n
        )
    
    def update_electrolyte_properties(self, new_n: float):
        """更新电解质折射率（用于实时调整）"""
        self.electrolyte_n = new_n
        self.metal_film.n_sample = new_n
    
    def set_beam_parameters(self, **kwargs):
        """设置光束参数"""
        for key, value in kwargs.items():
            if hasattr(self.beam_params, key):
                setattr(self.beam_params, key, value)
    
    def set_angle_range(self, min_angle_deg: float, max_angle_deg: float, 
                       resolution_deg: float = 0.1):
        """设置角度扫描范围"""
        self.angle_range_deg = (min_angle_deg, max_angle_deg)
        self.angle_resolution_deg = resolution_deg
    
    def measure_spr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """测量SPR反射率曲线"""
        # 生成角度序列
        angles_deg = np.arange(
            self.angle_range_deg[0],
            self.angle_range_deg[1] + self.angle_resolution_deg,
            self.angle_resolution_deg
        )
        
        reflectance = []
        phase_shifts = []
        
        for angle_deg in angles_deg:
            angle_rad = np.deg2rad(angle_deg)
            
            # 创建入射光线
            ray = Ray(
                position=np.array([0.0, 0.0]),
                direction=np.array([np.sin(angle_rad), -np.cos(angle_rad)]),
                wavelength=self.beam_params.wavelength_nm,
                polarization=self.beam_params.polarization,
                intensity=1.0
            )
            
            # 计算反射率和相位
            R = self.metal_film.get_reflectance(ray)
            phase = self.metal_film.get_phase_shift(
                self.beam_params.wavelength_nm, angle_rad
            )
            
            reflectance.append(R)
            phase_shifts.append(phase)
            
            # 记录测量数据
            measurement = SPRMeasurement(
                timestamp=time.time(),
                incident_angle_deg=angle_deg,
                wavelength_nm=self.beam_params.wavelength_nm,
                reflectance=R,
                phase_shift_rad=phase,
                field_enhancement=self.metal_film.field_enhancement(ray)
            )
            self.measurement_history.append(measurement)
        
        return np.array(angles_deg), np.array(reflectance), np.array(phase_shifts)
    
    def find_spr_angle(self) -> Tuple[float, float]:
        """找到SPR共振角度"""
        angles, reflectance, _ = self.measure_spr_curve()
        
        # 找到最小反射率对应的角度
        min_idx = np.argmin(reflectance)
        spr_angle = angles[min_idx]
        min_R = reflectance[min_idx]
        
        return float(spr_angle), float(min_R)
    
    def measure_at_angle(self, angle_deg: float) -> SPRMeasurement:
        """在指定角度测量SPR响应"""
        angle_rad = np.deg2rad(angle_deg)
        
        # 创建入射光线
        ray = Ray(
            position=np.array([0.0, 0.0]),
            direction=np.array([np.sin(angle_rad), -np.cos(angle_rad)]),
            wavelength=self.beam_params.wavelength_nm,
            polarization=self.beam_params.polarization,
            intensity=1.0
        )
        
        # 测量SPR响应
        R = self.metal_film.get_reflectance(ray)
        phase = self.metal_film.get_phase_shift(
            self.beam_params.wavelength_nm, angle_rad
        )
        enhancement = self.metal_film.field_enhancement(ray)
        
        measurement = SPRMeasurement(
            timestamp=time.time(),
            incident_angle_deg=angle_deg,
            wavelength_nm=self.beam_params.wavelength_nm,
            reflectance=R,
            phase_shift_rad=phase,
            field_enhancement=enhancement
        )
        
        self.measurement_history.append(measurement)
        return measurement
    
    def optimize_spr_angle(self) -> Dict[str, float]:
        """优化SPR工作角度以获得最佳灵敏度"""
        angles, reflectance, phases = self.measure_spr_curve()
        
        # 计算灵敏度 (dR/dθ)
        dR_dtheta = np.gradient(reflectance, angles)
        
        # 找到最大灵敏度点
        max_sensitivity_idx = np.argmax(np.abs(dR_dtheta))
        optimal_angle = angles[max_sensitivity_idx]
        max_sensitivity = abs(dR_dtheta[max_sensitivity_idx])
        
        # 找到最小反射率点（传统SPR工作点）
        min_R_idx = np.argmin(reflectance)
        spr_angle = angles[min_R_idx]
        min_reflectance = reflectance[min_R_idx]
        
        return {
            "optimal_angle_deg": float(optimal_angle),
            "max_sensitivity": float(max_sensitivity),
            "spr_angle_deg": float(spr_angle),
            "min_reflectance": float(min_reflectance),
            "phase_at_spr_rad": float(phases[min_R_idx])
        }
    
    def get_measurement_statistics(self, last_n_measurements: int = 100) -> Dict[str, float]:
        """获取最近测量的统计信息"""
        if not self.measurement_history:
            return {}
        
        recent_measurements = self.measurement_history[-last_n_measurements:]
        
        reflectances = [m.reflectance for m in recent_measurements]
        phases = [m.phase_shift_rad for m in recent_measurements]
        angles = [m.incident_angle_deg for m in recent_measurements]
        
        return {
            "mean_reflectance": float(np.mean(reflectances)),
            "std_reflectance": float(np.std(reflectances)),
            "mean_phase_rad": float(np.mean(phases)),
            "std_phase_rad": float(np.std(phases)),
            "mean_angle_deg": float(np.mean(angles)),
            "std_angle_deg": float(np.std(angles)),
            "measurement_count": len(recent_measurements)
        }
    
    def clear_measurement_history(self):
        """清除测量历史"""
        self.measurement_history.clear()
    
    def export_measurements(self, filepath: str):
        """导出测量数据到文件"""
        import json
        
        data = {
            "electrolyte_type": self.electrolyte_type.value,
            "electrolyte_material": self.electrolyte_material,
            "electrolyte_refractive_index": self.electrolyte_n,
            "metal_film_thickness_nm": self.metal_film_thickness,
            "beam_parameters": {
                "wavelength_nm": self.beam_params.wavelength_nm,
                "power_mw": self.beam_params.power_mw,
                "beam_diameter_mm": self.beam_params.beam_diameter_mm,
                "polarization": self.beam_params.polarization
            },
            "measurements": [
                {
                    "timestamp": m.timestamp,
                    "incident_angle_deg": m.incident_angle_deg,
                    "wavelength_nm": m.wavelength_nm,
                    "reflectance": m.reflectance,
                    "phase_shift_rad": m.phase_shift_rad,
                    "field_enhancement": m.field_enhancement
                }
                for m in self.measurement_history
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False) 