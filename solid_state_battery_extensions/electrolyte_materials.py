"""电解质材料属性定义模块"""

import numpy as np
from enum import Enum
from typing import Union, Dict, Any
from dataclasses import dataclass


class ElectrolyteType(Enum):
    """电解质类型枚举"""
    SULFIDE = "sulfide"  # 硫化物电解质
    POLYMER = "polymer"  # 聚合物电解质
    OXIDE = "oxide"      # 氧化物电解质
    CUSTOM = "custom"    # 自定义电解质


@dataclass
class ElectrolyteProperties:
    """电解质属性数据类"""
    name: str
    refractive_index_range: tuple[float, float]  # 折射率范围
    typical_thickness_range: tuple[float, float]  # 典型厚度范围 (nm)
    density: float  # 密度 (g/cm³)
    ionic_conductivity: float  # 离子电导率 (S/cm)
    electrochemical_window: float  # 电化学窗口 (V)
    
    
class ElectrolyteMaterials:
    """电解质材料数据库"""
    
    # 预定义材料属性
    MATERIALS_DB = {
        ElectrolyteType.SULFIDE: {
            "LPSCl": ElectrolyteProperties(
                name="Li6PS5Cl",
                refractive_index_range=(2.05, 2.10),
                typical_thickness_range=(500, 10000),  # 0.5-10 μm
                density=1.85,
                ionic_conductivity=1.3e-3,
                electrochemical_window=5.0
            ),
            "LGPS": ElectrolyteProperties(
                name="Li10GeP2S12",
                refractive_index_range=(2.06, 2.12),
                typical_thickness_range=(200, 5000),   # 0.2-5 μm
                density=2.15,
                ionic_conductivity=1.2e-2,
                electrochemical_window=5.2
            ),
            "Li2S_P2S5": ElectrolyteProperties(
                name="Li2S-P2S5",
                refractive_index_range=(2.03, 2.08),
                typical_thickness_range=(1000, 50000),  # 1-50 μm
                density=1.92,
                ionic_conductivity=6.4e-4,
                electrochemical_window=4.8
            )
        },
        
        ElectrolyteType.POLYMER: {
            "PEO": ElectrolyteProperties(
                name="Polyethylene Oxide",
                refractive_index_range=(1.42, 1.45),
                typical_thickness_range=(5000, 100000),  # 5-100 μm
                density=1.13,
                ionic_conductivity=1e-5,
                electrochemical_window=4.2
            ),
            "PVDF_HFP": ElectrolyteProperties(
                name="PVDF-HFP",
                refractive_index_range=(1.40, 1.44),
                typical_thickness_range=(10000, 200000),  # 10-200 μm
                density=1.78,
                ionic_conductivity=1e-4,
                electrochemical_window=4.5
            ),
            "PAN": ElectrolyteProperties(
                name="Polyacrylonitrile",
                refractive_index_range=(1.43, 1.46),
                typical_thickness_range=(8000, 150000),  # 8-150 μm
                density=1.18,
                ionic_conductivity=5e-5,
                electrochemical_window=4.3
            )
        },
        
        ElectrolyteType.OXIDE: {
            "LLZO": ElectrolyteProperties(
                name="Li7La3Zr2O12",
                refractive_index_range=(2.15, 2.25),
                typical_thickness_range=(100, 2000),   # 0.1-2 μm
                density=5.1,
                ionic_conductivity=3e-4,
                electrochemical_window=6.0
            ),
            "NASICON": ElectrolyteProperties(
                name="Li1.3Al0.3Ti1.7(PO4)3",
                refractive_index_range=(1.95, 2.05),
                typical_thickness_range=(500, 10000),  # 0.5-10 μm
                density=2.95,
                ionic_conductivity=7e-4,
                electrochemical_window=5.5
            )
        }
    }
    
    def __init__(self):
        self.custom_materials: Dict[str, ElectrolyteProperties] = {}
    
    def get_material(self, electrolyte_type: ElectrolyteType, material_name: str) -> ElectrolyteProperties:
        """获取指定电解质材料的属性"""
        if electrolyte_type == ElectrolyteType.CUSTOM:
            if material_name not in self.custom_materials:
                raise ValueError(f"自定义材料 '{material_name}' 未找到")
            return self.custom_materials[material_name]
        
        if electrolyte_type not in self.MATERIALS_DB:
            raise ValueError(f"不支持的电解质类型: {electrolyte_type}")
        
        type_materials = self.MATERIALS_DB[electrolyte_type]
        if material_name not in type_materials:
            raise ValueError(f"材料 '{material_name}' 在类型 '{electrolyte_type.value}' 中未找到")
        
        return type_materials[material_name]
    
    def add_custom_material(self, name: str, properties: ElectrolyteProperties):
        """添加自定义电解质材料"""
        self.custom_materials[name] = properties
    
    def get_refractive_index(self, electrolyte_type: ElectrolyteType, 
                           material_name: str, wavelength_nm: float = 632.8) -> float:
        """获取电解质在指定波长下的折射率"""
        material = self.get_material(electrolyte_type, material_name)
        
        # 简化模型：在给定范围内取中间值
        n_min, n_max = material.refractive_index_range
        n_center = (n_min + n_max) / 2
        
        # 对于He-Ne激光632.8nm的简单色散修正
        if wavelength_nm != 632.8:
            # 简单的Cauchy色散公式 n = A + B/λ²
            dispersion_factor = 1 + 0.01 * (632.8 - wavelength_nm) / 632.8
            n_center *= dispersion_factor
        
        return n_center
    
    def list_materials(self, electrolyte_type: ElectrolyteType = None) -> Dict[str, list]:
        """列出可用的材料"""
        if electrolyte_type is None:
            # 返回所有材料
            result = {}
            for e_type in ElectrolyteType:
                if e_type == ElectrolyteType.CUSTOM:
                    result[e_type.value] = list(self.custom_materials.keys())
                else:
                    result[e_type.value] = list(self.MATERIALS_DB.get(e_type, {}).keys())
            return result
        else:
            if electrolyte_type == ElectrolyteType.CUSTOM:
                return {electrolyte_type.value: list(self.custom_materials.keys())}
            else:
                return {electrolyte_type.value: list(self.MATERIALS_DB.get(electrolyte_type, {}).keys())}
    
    def get_thickness_tolerance(self, electrolyte_type: ElectrolyteType, 
                              material_name: str) -> float:
        """获取材料的厚度测量容差建议"""
        material = self.get_material(electrolyte_type, material_name)
        thickness_range = material.typical_thickness_range
        avg_thickness = sum(thickness_range) / 2
        
        # 对于薄层电解质(<1μm)，要求±50nm精度
        if avg_thickness < 1000:
            return 50.0
        # 对于厚层，精度可适当放宽
        elif avg_thickness < 10000:
            return 100.0
        else:
            return 200.0


# 创建全局材料数据库实例
electrolyte_db = ElectrolyteMaterials() 