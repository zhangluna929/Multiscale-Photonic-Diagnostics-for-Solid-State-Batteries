"""厚度分析模块"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, curve_fit
import warnings

from .electrolyte_materials import ElectrolyteMaterials, ElectrolyteType


@dataclass
class ThicknessResult:
    """厚度分析结果"""
    thickness_nm: float
    thickness_error_nm: float
    refractive_index: float
    confidence: float  # 0-1, 拟合置信度
    fitting_method: str
    residual_error: float


class ElectrolyteFilm:
    """电解质薄膜类，扩展的传输矩阵法实现"""
    
    def __init__(
        self,
        n_prism: float = 1.515,
        n_metal: complex = 0.18 + 3.06j,  # 632.8nm处金的折射率
        metal_thickness_nm: float = 50.0,
        n_electrolyte: float = 1.45,
        electrolyte_thickness_nm: float = 1000.0,
        n_substrate: float = 1.0  # 空气或其他基底
    ):
        """初始化四层结构：棱镜/金属膜/电解质/基底"""
        self.n_prism = n_prism
        self.n_metal = n_metal
        self.metal_thickness_nm = metal_thickness_nm
        self.n_electrolyte = n_electrolyte
        self.electrolyte_thickness_nm = electrolyte_thickness_nm
        self.n_substrate = n_substrate
    
    def _transfer_matrix_element(self, n1: complex, n2: complex, thickness_nm: float, 
                                k0: float, theta1: float) -> np.ndarray:
        """计算单层的传输矩阵"""
        # 计算各层中的传播角度
        sin_theta1 = np.sin(theta1)
        sin_theta2 = self.n_prism / n2 * sin_theta1
        
        cos_theta1 = np.cos(theta1)
        cos_theta2 = np.lib.scimath.sqrt(1 - sin_theta2**2)
        
        # TM偏振的菲涅尔系数
        # t_{12} = 2*n1*cos_θ1 / (n2*cos_θ1 + n1*cos_θ2)
        
        # 传播相位
        beta = n2 * cos_theta2 * k0 * thickness_nm * 1e-9
        
        # 传输矩阵 (2x2)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        
        # 对于TM偏振
        Z2 = n2 / cos_theta2  # 阻抗
        
        M = np.array([
            [cos_beta, 1j * sin_beta / Z2],
            [1j * Z2 * sin_beta, cos_beta]
        ], dtype=complex)
        
        return M
    
    def calculate_reflectance_tm(self, wavelength_nm: float, theta_deg: float) -> complex:
        """使用传输矩阵法计算TM偏振的复反射系数"""
        theta_rad = np.deg2rad(theta_deg)
        k0 = 2 * np.pi / (wavelength_nm * 1e-9)
        
        # 计算各层传播角度
        sin_theta1 = np.sin(theta_rad)
        cos_theta1 = np.cos(theta_rad)
        
        # 金属层
        sin_theta_metal = self.n_prism / self.n_metal * sin_theta1
        cos_theta_metal = np.lib.scimath.sqrt(1 - sin_theta_metal**2)
        
        # 电解质层
        sin_theta_electrolyte = self.n_prism / self.n_electrolyte * sin_theta1
        cos_theta_electrolyte = np.lib.scimath.sqrt(1 - sin_theta_electrolyte**2)
        
        # 基底层
        sin_theta_substrate = self.n_prism / self.n_substrate * sin_theta1
        cos_theta_substrate = np.lib.scimath.sqrt(1 - sin_theta_substrate**2)
        
        # 计算各层传输矩阵
        beta_metal = self.n_metal * cos_theta_metal * k0 * self.metal_thickness_nm * 1e-9
        Z_metal = self.n_metal / cos_theta_metal
        
        M_metal = np.array([
            [np.cos(beta_metal), 1j * np.sin(beta_metal) / Z_metal],
            [1j * Z_metal * np.sin(beta_metal), np.cos(beta_metal)]
        ], dtype=complex)
        
        # 电解质层传输矩阵
        beta_electrolyte = self.n_electrolyte * cos_theta_electrolyte * k0 * self.electrolyte_thickness_nm * 1e-9
        Z_electrolyte = self.n_electrolyte / cos_theta_electrolyte
        
        M_electrolyte = np.array([
            [np.cos(beta_electrolyte), 1j * np.sin(beta_electrolyte) / Z_electrolyte],
            [1j * Z_electrolyte * np.sin(beta_electrolyte), np.cos(beta_electrolyte)]
        ], dtype=complex)
        
        # 总传输矩阵
        M_total = M_metal @ M_electrolyte
        
        # 边界条件
        Z1 = self.n_prism / cos_theta1  # 棱镜阻抗
        Z_sub = self.n_substrate / cos_theta_substrate  # 基底阻抗
        
        # 从传输矩阵计算反射系数
        M11, M12, M21, M22 = M_total[0, 0], M_total[0, 1], M_total[1, 0], M_total[1, 1]
        
        numerator = (M11 * Z1 - M22 * Z_sub) + (M12 * Z1 * Z_sub - M21)
        denominator = (M11 * Z1 + M22 * Z_sub) + (M12 * Z1 * Z_sub + M21)
        
        r = numerator / denominator
        return r
    
    def get_reflectance(self, wavelength_nm: float, theta_deg: float) -> float:
        """获取反射率"""
        r = self.calculate_reflectance_tm(wavelength_nm, theta_deg)
        return float(np.abs(r)**2)
    
    def get_phase(self, wavelength_nm: float, theta_deg: float) -> float:
        """获取相位 (弧度)"""
        r = self.calculate_reflectance_tm(wavelength_nm, theta_deg)
        return float(np.angle(r))


class ThicknessAnalyzer:
    """电解质厚度分析器"""
    
    def __init__(
        self,
        materials_db: Optional[ElectrolyteMaterials] = None,
        wavelength_nm: float = 632.8,
        metal_thickness_nm: float = 50.0,
        prism_n: float = 1.515
    ):
        """初始化厚度分析器"""
        self.materials_db = materials_db or ElectrolyteMaterials()
        self.wavelength_nm = wavelength_nm
        self.metal_thickness_nm = metal_thickness_nm
        self.prism_n = prism_n
        
        # 拟合参数
        self.max_thickness_nm = 10000  # 最大厚度范围
        self.thickness_resolution_nm = 1.0  # 厚度分辨率
        
    def fit_thickness_single_angle(
        self,
        measured_reflectance: float,
        incident_angle_deg: float,
        n_electrolyte: float,
        thickness_range_nm: Tuple[float, float] = (10, 5000)
    ) -> ThicknessResult:
        """基于单个角度的反射率数据拟合厚度"""
        def objective(thickness_nm):
            """目标函数：最小化反射率差异"""
            film = ElectrolyteFilm(
                n_prism=self.prism_n,
                metal_thickness_nm=self.metal_thickness_nm,
                n_electrolyte=n_electrolyte,
                electrolyte_thickness_nm=thickness_nm
            )
            
            calculated_R = film.get_reflectance(self.wavelength_nm, incident_angle_deg)
            return abs(calculated_R - measured_reflectance)**2
        
        # 优化求解
        result = minimize_scalar(
            objective,
            bounds=thickness_range_nm,
            method='bounded'
        )
        
        best_thickness = float(result.x)
        residual_error = float(result.fun)
        
        # 估算误差 - 基于目标函数的曲率
        thickness_step = 10.0  # nm
        error_left = objective(max(best_thickness - thickness_step, thickness_range_nm[0]))
        error_right = objective(min(best_thickness + thickness_step, thickness_range_nm[1]))
        
        # 基于二阶导数估算误差
        curvature = (error_left + error_right - 2 * residual_error) / (thickness_step**2)
        thickness_error = thickness_step if curvature <= 0 else min(50.0, thickness_step / np.sqrt(curvature))
        
        # 置信度评估
        confidence = max(0.0, min(1.0, 1.0 - residual_error * 100))
        
        return ThicknessResult(
            thickness_nm=best_thickness,
            thickness_error_nm=thickness_error,
            refractive_index=n_electrolyte,
            confidence=confidence,
            fitting_method="single_angle_optimization",
            residual_error=residual_error
        )
    
    def fit_thickness_multi_angle(
        self,
        angle_data: List[float],  # 度
        reflectance_data: List[float],
        n_electrolyte: float,
        thickness_range_nm: Tuple[float, float] = (10, 5000)
    ) -> ThicknessResult:
        """基于多个角度的反射率数据拟合厚度"""
        if len(angle_data) != len(reflectance_data):
            raise ValueError("角度数据和反射率数据长度不匹配")
        
        def objective(thickness_nm):
            """目标函数：最小化所有角度的反射率差异平方和"""
            film = ElectrolyteFilm(
                n_prism=self.prism_n,
                metal_thickness_nm=self.metal_thickness_nm,
                n_electrolyte=n_electrolyte,
                electrolyte_thickness_nm=thickness_nm
            )
            
            total_error = 0.0
            for angle, measured_R in zip(angle_data, reflectance_data):
                calculated_R = film.get_reflectance(self.wavelength_nm, angle)
                total_error += (calculated_R - measured_R)**2
            
            return total_error / len(angle_data)  # 均方误差
        
        # 优化求解
        result = minimize_scalar(
            objective,
            bounds=thickness_range_nm,
            method='bounded'
        )
        
        best_thickness = float(result.x)
        residual_error = float(result.fun)
        
        # 更精确的误差估算
        h = 5.0  # nm，小步长
        f_0 = objective(best_thickness)
        f_plus = objective(min(best_thickness + h, thickness_range_nm[1]))
        f_minus = objective(max(best_thickness - h, thickness_range_nm[0]))
        
        # 二阶差分近似二阶导数
        second_derivative = (f_plus - 2*f_0 + f_minus) / (h**2)
        
        if second_derivative > 0:
            # Fisher信息矩阵对角元素的近似
            thickness_error = min(50.0, 1.0 / np.sqrt(second_derivative))
        else:
            thickness_error = 50.0  # 默认误差
        
        # 基于数据点数量和拟合质量的置信度
        confidence = max(0.0, min(1.0, 
            (1.0 - residual_error) * min(1.0, len(angle_data) / 10.0)
        ))
        
        return ThicknessResult(
            thickness_nm=best_thickness,
            thickness_error_nm=thickness_error,
            refractive_index=n_electrolyte,
            confidence=confidence,
            fitting_method="multi_angle_optimization",
            residual_error=residual_error
        )
    
    def fit_thickness_and_refractive_index(
        self,
        angle_data: List[float],
        reflectance_data: List[float],
        thickness_range_nm: Tuple[float, float] = (10, 5000),
        n_range: Tuple[float, float] = (1.4, 2.3)
    ) -> ThicknessResult:
        """同时拟合厚度和折射率"""
        from scipy.optimize import minimize
        
        def objective(params):
            """目标函数：参数为[厚度, 折射率]"""
            thickness_nm, n_electrolyte = params
            
            # 边界检查
            if not (thickness_range_nm[0] <= thickness_nm <= thickness_range_nm[1]):
                return 1e6
            if not (n_range[0] <= n_electrolyte <= n_range[1]):
                return 1e6
            
            film = ElectrolyteFilm(
                n_prism=self.prism_n,
                metal_thickness_nm=self.metal_thickness_nm,
                n_electrolyte=n_electrolyte,
                electrolyte_thickness_nm=thickness_nm
            )
            
            total_error = 0.0
            for angle, measured_R in zip(angle_data, reflectance_data):
                try:
                    calculated_R = film.get_reflectance(self.wavelength_nm, angle)
                    total_error += (calculated_R - measured_R)**2
                except:
                    return 1e6
            
            return total_error / len(angle_data)
        
        # 初始猜测
        initial_thickness = sum(thickness_range_nm) / 2
        initial_n = sum(n_range) / 2
        
        # 优化
        result = minimize(
            objective,
            x0=[initial_thickness, initial_n],
            bounds=[thickness_range_nm, n_range],
            method='L-BFGS-B'
        )
        
        if not result.success:
            warnings.warn("厚度和折射率联合拟合未收敛")
        
        best_thickness, best_n = result.x
        residual_error = float(result.fun)
        
        # 误差估算（简化）
        thickness_error = min(50.0, abs(thickness_range_nm[1] - thickness_range_nm[0]) * 0.01)
        confidence = max(0.0, min(1.0, 1.0 - residual_error * 50))
        
        return ThicknessResult(
            thickness_nm=float(best_thickness),
            thickness_error_nm=thickness_error,
            refractive_index=float(best_n),
            confidence=confidence,
            fitting_method="joint_thickness_refractive_index",
            residual_error=residual_error
        )
    
    def analyze_electrolyte_thickness(
        self,
        electrolyte_type: ElectrolyteType,
        material_name: str,
        measurement_data: Dict[str, List[float]],
        analysis_method: str = "multi_angle"
    ) -> ThicknessResult:
        """分析电解质厚度的主接口"""
        # 获取材料属性
        material_props = self.materials_db.get_material(electrolyte_type, material_name)
        n_electrolyte = self.materials_db.get_refractive_index(
            electrolyte_type, material_name, self.wavelength_nm
        )
        
        # 根据材料类型设置合理的厚度范围
        thickness_range = material_props.typical_thickness_range
        
        angles = measurement_data['angles']
        reflectance = measurement_data['reflectance']
        
        if analysis_method == "single_angle":
            if len(angles) < 1:
                raise ValueError("单角度分析需要至少一个数据点")
            return self.fit_thickness_single_angle(
                reflectance[0], angles[0], n_electrolyte, thickness_range
            )
        
        elif analysis_method == "multi_angle":
            return self.fit_thickness_multi_angle(
                angles, reflectance, n_electrolyte, thickness_range
            )
        
        elif analysis_method == "joint_fit":
            # 估计折射率范围
            n_range = material_props.refractive_index_range
            return self.fit_thickness_and_refractive_index(
                angles, reflectance, thickness_range, n_range
            )
        
        else:
            raise ValueError(f"未知的分析方法: {analysis_method}")
    
    def validate_measurement_precision(
        self,
        thickness_result: ThicknessResult,
        target_precision_nm: float = 50.0
    ) -> Dict[str, Any]:
        """验证测量精度是否满足要求"""
        meets_precision = thickness_result.thickness_error_nm <= target_precision_nm
        
        return {
            "meets_target_precision": meets_precision,
            "achieved_precision_nm": thickness_result.thickness_error_nm,
            "target_precision_nm": target_precision_nm,
            "precision_ratio": thickness_result.thickness_error_nm / target_precision_nm,
            "confidence_level": thickness_result.confidence,
            "measurement_quality": "excellent" if thickness_result.confidence > 0.9 else
                                 "good" if thickness_result.confidence > 0.7 else
                                 "acceptable" if thickness_result.confidence > 0.5 else "poor"
        } 