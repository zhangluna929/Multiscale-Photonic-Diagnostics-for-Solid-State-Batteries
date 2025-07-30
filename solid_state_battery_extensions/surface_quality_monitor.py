"""表面质量监测模块"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time
from scipy import ndimage, stats
from scipy.signal import find_peaks, savgol_filter
import warnings


@dataclass
class SurfaceQualityMetrics:
    """表面质量指标"""
    uniformity_ratio: float  # σ/宽度比值
    surface_roughness_nm: float  # 表面粗糙度 (nm)
    spot_size_variation: float  # 光斑尺寸变化
    reflection_stability: float  # 反射稳定性
    quality_score: float  # 综合质量评分 (0-1)
    measurement_timestamp: float


@dataclass
class BeamProfileData:
    """光束轮廓数据"""
    x_coordinates: np.ndarray  # x坐标 (mm)
    y_coordinates: np.ndarray  # y坐标 (mm) 
    intensity_map: np.ndarray  # 强度分布图
    peak_intensity: float
    total_power: float
    beam_width_x: float  # x方向光束宽度 (mm)
    beam_width_y: float  # y方向光束宽度 (mm)


class SurfaceQualityMonitor:
    """表面质量监测器"""
    
    def __init__(self, target_uniformity: float = 0.21):
        """初始化表面质量监测器"""
        self.target_uniformity = target_uniformity
        self.baseline_uniformity = 0.35  # 初始均匀度
        
        # 监测历史
        self.quality_history: List[SurfaceQualityMetrics] = []
        
        # 阈值设置
        self.roughness_threshold_nm = 10.0  # 粗糙度阈值
        self.stability_threshold = 0.05  # 反射稳定性阈值 (5%)
        
        # 滤波参数
        self.smoothing_window = 5
        self.noise_threshold = 0.01
    
    def analyze_beam_profile(
        self,
        intensity_data: np.ndarray,
        pixel_size_mm: float = 0.01
    ) -> BeamProfileData:
        """分析光束轮廓"""
        height, width = intensity_data.shape
        
        # 生成坐标
        x_coords = np.arange(width) * pixel_size_mm
        y_coords = np.arange(height) * pixel_size_mm
        
        # 找到峰值位置
        peak_pos = np.unravel_index(np.argmax(intensity_data), intensity_data.shape)
        peak_intensity = float(intensity_data[peak_pos])
        total_power = float(np.sum(intensity_data))
        
        # 计算光束宽度 (FWHM)
        x_profile = intensity_data[peak_pos[0], :]
        x_width = self._calculate_beam_width(x_profile, pixel_size_mm)
        
        # Y方向轮廓
        y_profile = intensity_data[:, peak_pos[1]]
        y_width = self._calculate_beam_width(y_profile, pixel_size_mm)
        
        return BeamProfileData(
            x_coordinates=x_coords,
            y_coordinates=y_coords,
            intensity_map=intensity_data,
            peak_intensity=peak_intensity,
            total_power=total_power,
            beam_width_x=x_width,
            beam_width_y=y_width
        )
    
    def _calculate_beam_width(self, profile: np.ndarray, pixel_size: float) -> float:
        """计算光束宽度 (FWHM)"""
        # 归一化轮廓
        profile_norm = profile / np.max(profile)
        
        # 找到半高点
        half_max = 0.5
        indices = np.where(profile_norm >= half_max)[0]
        
        if len(indices) == 0:
            return 0.0
        
        # FWHM宽度
        width_pixels = indices[-1] - indices[0] + 1
        return width_pixels * pixel_size
    
    def calculate_uniformity_ratio(self, intensity_data: np.ndarray) -> float:
        """计算光斑均匀度比值 (σ/宽度)"""
        # 找到光斑有效区域（强度>最大值的10%）
        threshold = 0.1 * np.max(intensity_data)
        mask = intensity_data > threshold
        
        if not np.any(mask):
            return 1.0  # 无有效光斑
        
        # 计算光斑质心
        y_indices, x_indices = np.where(mask)
        weights = intensity_data[mask]
        
        centroid_x = np.average(x_indices, weights=weights)
        centroid_y = np.average(y_indices, weights=weights)
        
        # 计算到质心的距离
        distances = np.sqrt((x_indices - centroid_x)**2 + (y_indices - centroid_y)**2)
        
        # 计算标准差
        sigma = np.sqrt(np.average(distances**2, weights=weights))
        
        # 计算光斑有效宽度
        width = np.sqrt(np.sum(mask))  # 有效像素数的平方根
        
        if width == 0:
            return 1.0
        
        return float(sigma / width)
    
    def estimate_surface_roughness(
        self,
        reflectance_map: np.ndarray,
        wavelength_nm: float = 632.8
    ) -> float:
        """基于反射率变化估算表面粗糙度"""
        # 计算反射率的空间变化
        grad_x = np.gradient(reflectance_map, axis=1)
        grad_y = np.gradient(reflectance_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 统计梯度分布
        rms_gradient = np.sqrt(np.mean(gradient_magnitude**2))
        
        # 经验公式：粗糙度与反射率梯度的关系
        roughness_nm = rms_gradient * wavelength_nm / (4 * np.pi) * 100
        
        return float(roughness_nm)
    
    def analyze_reflection_stability(
        self,
        time_series_data: List[np.ndarray],
        time_points: List[float]
    ) -> float:
        """分析反射稳定性"""
        if len(time_series_data) < 2:
            return 0.0
        
        # 计算每个时间点的平均反射率
        mean_reflectances = [np.mean(data) for data in time_series_data]
        
        # 计算相对标准差
        mean_R = np.mean(mean_reflectances)
        std_R = np.std(mean_reflectances)
        
        if mean_R == 0:
            return 1.0
        
        relative_std = std_R / mean_R
        return float(relative_std)
    
    def calculate_spot_size_variation(
        self,
        beam_profiles: List[BeamProfileData]
    ) -> float:
        """计算光斑尺寸变化"""
        if len(beam_profiles) < 2:
            return 0.0
        
        # 计算平均光斑尺寸
        spot_sizes = []
        for profile in beam_profiles:
            avg_width = (profile.beam_width_x + profile.beam_width_y) / 2
            spot_sizes.append(avg_width)
        
        # 计算变化系数 (CV = σ/μ)
        mean_size = np.mean(spot_sizes)
        std_size = np.std(spot_sizes)
        
        if mean_size == 0:
            return 1.0
        
        return float(std_size / mean_size)
    
    def comprehensive_quality_assessment(
        self,
        intensity_data: np.ndarray,
        reflectance_map: Optional[np.ndarray] = None,
        wavelength_nm: float = 632.8,
        pixel_size_mm: float = 0.01
    ) -> SurfaceQualityMetrics:
        """综合表面质量评估"""
        # 计算均匀度比值
        uniformity_ratio = self.calculate_uniformity_ratio(intensity_data)
        
        # 估算表面粗糙度
        if reflectance_map is not None:
            surface_roughness = self.estimate_surface_roughness(reflectance_map, wavelength_nm)
        else:
            # 基于强度变化估算
            surface_roughness = self.estimate_surface_roughness(intensity_data, wavelength_nm)
        
        # 光斑尺寸变化（需要历史数据）
        beam_profile = self.analyze_beam_profile(intensity_data, pixel_size_mm)
        spot_variation = 0.0  # 单次测量无法计算变化
        
        # 反射稳定性（需要时间序列数据）
        reflection_stability = 0.0  # 单次测量无法计算稳定性
        
        # 计算综合质量评分
        quality_score = self._calculate_quality_score(
            uniformity_ratio, surface_roughness, spot_variation, reflection_stability
        )
        
        metrics = SurfaceQualityMetrics(
            uniformity_ratio=uniformity_ratio,
            surface_roughness_nm=surface_roughness,
            spot_size_variation=spot_variation,
            reflection_stability=reflection_stability,
            quality_score=quality_score,
            measurement_timestamp=time.time()
        )
        
        # 添加到历史记录
        self.quality_history.append(metrics)
        
        return metrics
    
    def _calculate_quality_score(
        self,
        uniformity_ratio: float,
        surface_roughness: float,
        spot_variation: float,
        reflection_stability: float
    ) -> float:
        """计算综合质量评分"""
        # 均匀度评分 (目标0.21，基线0.35)
        uniformity_score = max(0, 1 - (uniformity_ratio - self.target_uniformity) / 
                              (self.baseline_uniformity - self.target_uniformity))
        uniformity_score = min(1, uniformity_score)
        
        # 粗糙度评分
        roughness_score = max(0, 1 - surface_roughness / self.roughness_threshold_nm)
        roughness_score = min(1, roughness_score)
        
        # 光斑稳定性评分
        spot_score = max(0, 1 - spot_variation / 0.1)  # 10%变化为阈值
        spot_score = min(1, spot_score)
        
        # 反射稳定性评分
        stability_score = max(0, 1 - reflection_stability / self.stability_threshold)
        stability_score = min(1, stability_score)
        
        # 加权平均
        weights = [0.4, 0.3, 0.15, 0.15]  # 均匀度权重最高
        scores = [uniformity_score, roughness_score, spot_score, stability_score]
        
        quality_score = sum(w * s for w, s in zip(weights, scores))
        return float(quality_score)
    
    def real_time_quality_monitoring(
        self,
        intensity_stream: List[np.ndarray],
        update_interval_s: float = 1.0
    ) -> List[SurfaceQualityMetrics]:
        """实时质量监测"""
        results = []
        
        for i, intensity_data in enumerate(intensity_stream):
            # 计算当前质量指标
            metrics = self.comprehensive_quality_assessment(intensity_data)
            results.append(metrics)
            
            # 检查是否需要报警
            if metrics.uniformity_ratio > self.baseline_uniformity * 1.1:
                warnings.warn(f"表面质量下降警告：均匀度比值 {metrics.uniformity_ratio:.3f} "
                            f"超过基线 {self.baseline_uniformity:.3f}")
            
            # 模拟时间间隔
            time.sleep(update_interval_s)
        
        return results
    
    def quality_improvement_analysis(self) -> Dict[str, Any]:
        """分析质量改善情况"""
        if len(self.quality_history) < 2:
            return {"error": "数据不足，无法进行改善分析"}
        
        # 计算趋势
        timestamps = [m.measurement_timestamp for m in self.quality_history]
        uniformity_values = [m.uniformity_ratio for m in self.quality_history]
        quality_scores = [m.quality_score for m in self.quality_history]
        
        # 线性回归分析趋势
        uniformity_slope, _, uniformity_r, _, _ = stats.linregress(
            range(len(uniformity_values)), uniformity_values
        )
        quality_slope, _, quality_r, _, _ = stats.linregress(
            range(len(quality_scores)), quality_scores
        )
        
        # 计算改善百分比
        initial_uniformity = uniformity_values[0]
        current_uniformity = uniformity_values[-1]
        improvement_percent = (initial_uniformity - current_uniformity) / initial_uniformity * 100
        
        # 目标达成评估
        target_achieved = current_uniformity <= self.target_uniformity
        progress_percent = min(100, (initial_uniformity - current_uniformity) / 
                             (initial_uniformity - self.target_uniformity) * 100)
        
        return {
            "initial_uniformity": initial_uniformity,
            "current_uniformity": current_uniformity,
            "target_uniformity": self.target_uniformity,
            "improvement_percent": improvement_percent,
            "target_achieved": target_achieved,
            "progress_percent": progress_percent,
            "uniformity_trend_slope": uniformity_slope,
            "quality_trend_slope": quality_slope,
            "correlation_uniformity": uniformity_r,
            "correlation_quality": quality_r,
            "measurement_count": len(self.quality_history)
        }
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """生成质量监测报告"""
        if not self.quality_history:
            return {"error": "无监测数据"}
        
        latest_metrics = self.quality_history[-1]
        improvement_analysis = self.quality_improvement_analysis()
        
        # 统计信息
        all_uniformity = [m.uniformity_ratio for m in self.quality_history]
        all_roughness = [m.surface_roughness_nm for m in self.quality_history]
        all_quality = [m.quality_score for m in self.quality_history]
        
        report = {
            "summary": {
                "total_measurements": len(self.quality_history),
                "monitoring_duration_hours": (
                    self.quality_history[-1].measurement_timestamp - 
                    self.quality_history[0].measurement_timestamp
                ) / 3600 if len(self.quality_history) > 1 else 0,
                "current_quality_score": latest_metrics.quality_score
            },
            "current_metrics": {
                "uniformity_ratio": latest_metrics.uniformity_ratio,
                "surface_roughness_nm": latest_metrics.surface_roughness_nm,
                "spot_size_variation": latest_metrics.spot_size_variation,
                "reflection_stability": latest_metrics.reflection_stability
            },
            "statistics": {
                "uniformity": {
                    "mean": float(np.mean(all_uniformity)),
                    "std": float(np.std(all_uniformity)),
                    "min": float(np.min(all_uniformity)),
                    "max": float(np.max(all_uniformity))
                },
                "roughness": {
                    "mean": float(np.mean(all_roughness)),
                    "std": float(np.std(all_roughness)),
                    "min": float(np.min(all_roughness)),
                    "max": float(np.max(all_roughness))
                },
                "quality": {
                    "mean": float(np.mean(all_quality)),
                    "std": float(np.std(all_quality)),
                    "min": float(np.min(all_quality)),
                    "max": float(np.max(all_quality))
                }
            },
            "improvement_analysis": improvement_analysis,
            "recommendations": self._generate_recommendations(latest_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: SurfaceQualityMetrics) -> List[str]:
        """生成改善建议"""
        recommendations = []
        
        if metrics.uniformity_ratio > self.target_uniformity:
            recommendations.append(
                f"均匀度需要改善：当前{metrics.uniformity_ratio:.3f}，目标{self.target_uniformity:.3f}"
            )
        
        if metrics.surface_roughness_nm > self.roughness_threshold_nm:
            recommendations.append(
                f"表面粗糙度过高：{metrics.surface_roughness_nm:.1f}nm，建议表面处理"
            )
        
        if metrics.spot_size_variation > 0.1:
            recommendations.append(
                f"光斑尺寸变化过大：{metrics.spot_size_variation:.3f}，检查光学系统稳定性"
            )
        
        if metrics.reflection_stability > self.stability_threshold:
            recommendations.append(
                f"反射稳定性不足：{metrics.reflection_stability:.3f}，检查环境条件"
            )
        
        if not recommendations:
            recommendations.append("表面质量良好，继续保持当前工艺条件")
        
        return recommendations
    
    def export_quality_data(self, filepath: str):
        """导出质量监测数据"""
        import json
        
        data = {
            "monitoring_config": {
                "target_uniformity": self.target_uniformity,
                "baseline_uniformity": self.baseline_uniformity,
                "roughness_threshold_nm": self.roughness_threshold_nm,
                "stability_threshold": self.stability_threshold
            },
            "quality_history": [
                {
                    "timestamp": m.measurement_timestamp,
                    "uniformity_ratio": m.uniformity_ratio,
                    "surface_roughness_nm": m.surface_roughness_nm,
                    "spot_size_variation": m.spot_size_variation,
                    "reflection_stability": m.reflection_stability,
                    "quality_score": m.quality_score
                }
                for m in self.quality_history
            ],
            "quality_report": self.generate_quality_report()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False) 