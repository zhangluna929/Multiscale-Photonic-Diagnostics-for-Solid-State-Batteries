"""锂枝晶早期预警监测模块"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import time
import json
import csv
from pathlib import Path
import threading
from queue import Queue
import logging
from enum import Enum

try:
    from .spr_electrolyte_monitor import SPRElectrolyteMonitor, SPRMeasurement, BeamParameters
except ImportError:
    from spr_electrolyte_monitor import SPRElectrolyteMonitor, SPRMeasurement, BeamParameters


class DendriteRiskLevel(Enum):
    """锂枝晶风险等级"""
    LOW = "低风险"
    MEDIUM = "中等风险"
    HIGH = "高风险"
    CRITICAL = "严重风险"


class AlertType(Enum):
    """报警类型"""
    REFLECTANCE_CHANGE = "反射率变化"
    PHASE_CHANGE = "相位变化"
    MORPHOLOGY_CHANGE = "形态变化"
    SURFACE_IRREGULARITY = "表面不规则性"
    DENDRITE_FORMATION = "枝晶形成"


@dataclass
class SpotMorphologyData:
    """光斑形态数据"""
    timestamp: float
    uniformity_index: float  # 光斑均匀度指数 (0-1)
    distribution_variance: float  # 分布方差
    intensity_profile: np.ndarray  # 强度分布
    center_shift: Tuple[float, float]  # 中心偏移 (x, y)
    shape_factor: float  # 形状因子
    irregularity_score: float  # 不规则性评分


@dataclass
class DendriteAlert:
    """锂枝晶预警信息"""
    timestamp: float
    alert_type: AlertType
    risk_level: DendriteRiskLevel
    message: str
    measured_value: float
    threshold_value: float
    confidence: float  # 置信度 (0-1)
    recommended_action: str


@dataclass
class DendriteMonitoringConfig:
    """锂枝晶监测配置"""
    # 反射率变化阈值
    reflectance_threshold_percent: float = 0.05  # ±0.05%
    
    # 相位变化阈值
    phase_threshold_deg: float = 0.1  # ±0.1°
    
    # 光斑形态阈值
    uniformity_threshold: float = 0.85  # 均匀度阈值
    irregularity_threshold: float = 0.3  # 不规则性阈值
    
    # 数据采集频率
    sampling_rate_hz: float = 10.0  # 10Hz以上
    
    # 数据存储设置
    auto_save_enabled: bool = True
    save_interval_seconds: float = 60.0  # 每分钟保存一次
    max_history_size: int = 10000  # 最大历史记录数
    
    # 报警设置
    alert_enabled: bool = True
    alert_callback: Optional[Callable[[DendriteAlert], None]] = None


class LithiumDendriteMonitor:
    """锂枝晶早期预警监测器"""
    
    def __init__(
        self,
        spr_monitor: SPRElectrolyteMonitor,
        config: Optional[DendriteMonitoringConfig] = None
    ):
        """初始化锂枝晶监测器"""
        self.spr_monitor = spr_monitor
        self.config = config or DendriteMonitoringConfig()
        
        # 监测数据存储
        self.baseline_reflectance: Optional[float] = None
        self.baseline_phase: Optional[float] = None
        self.baseline_morphology: Optional[SpotMorphologyData] = None
        
        # 历史数据
        self.reflectance_history: List[Tuple[float, float]] = []  # (timestamp, value)
        self.phase_history: List[Tuple[float, float]] = []
        self.morphology_history: List[SpotMorphologyData] = []
        self.alert_history: List[DendriteAlert] = []
        
        # 实时监控相关
        self.monitoring_active: bool = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.data_queue: Queue = Queue()
        
        # 数据保存相关
        self.last_save_time: float = time.time()
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
        # 初始化
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """设置监测系统"""
        # 设置SPR监测器到最佳角度
        optimization_result = self.spr_monitor.optimize_spr_angle()
        self.optimal_angle = optimization_result["optimal_angle_deg"]
        
        self.logger.info(f"锂枝晶监测器初始化完成，最佳角度: {self.optimal_angle:.2f}°")
    
    def establish_baseline(self, measurement_count: int = 50) -> Dict[str, float]:
        """建立基线测量数据"""
        self.logger.info(f"开始建立基线数据，测量次数: {measurement_count}")
        
        reflectances = []
        phases = []
        morphologies = []
        
        for i in range(measurement_count):
            # SPR测量
            measurement = self.spr_monitor.measure_at_angle(self.optimal_angle)
            reflectances.append(measurement.reflectance)
            phases.append(measurement.phase_shift_rad)
            
            # 光斑形态测量
            morphology = self._measure_spot_morphology()
            morphologies.append(morphology)
            
            # 控制测量频率
            time.sleep(1.0 / self.config.sampling_rate_hz)
        
        # 计算基线值
        self.baseline_reflectance = float(np.mean(reflectances))
        self.baseline_phase = float(np.mean(phases))
        
        # 基线形态数据
        uniformities = [m.uniformity_index for m in morphologies]
        irregularities = [m.irregularity_score for m in morphologies]
        
        baseline_stats = {
            "baseline_reflectance": self.baseline_reflectance,
            "reflectance_std": float(np.std(reflectances)),
            "baseline_phase_rad": self.baseline_phase,
            "phase_std_rad": float(np.std(phases)),
            "baseline_uniformity": float(np.mean(uniformities)),
            "uniformity_std": float(np.std(uniformities)),
            "baseline_irregularity": float(np.mean(irregularities)),
            "irregularity_std": float(np.std(irregularities)),
            "measurement_count": measurement_count
        }
        
        self.logger.info("基线数据建立完成")
        return baseline_stats
    
    def _measure_spot_morphology(self) -> SpotMorphologyData:
        """测量光斑形态数据"""
        # 模拟光斑强度分布测量（实际应用中需要连接CCD/CMOS相机）
        
        # 生成二维高斯分布作为光斑强度分布
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # 添加小的随机扰动模拟实际测量
        center_x = np.random.normal(0, 0.1)
        center_y = np.random.normal(0, 0.1)
        sigma_x = 1.0 + np.random.normal(0, 0.05)
        sigma_y = 1.0 + np.random.normal(0, 0.05)
        
        intensity = np.exp(-((X - center_x)**2 / (2 * sigma_x**2) + 
                            (Y - center_y)**2 / (2 * sigma_y**2)))
        
        # 添加噪声
        noise_level = 0.02
        intensity += np.random.normal(0, noise_level, intensity.shape)
        intensity = np.clip(intensity, 0, 1)
        
        # 计算形态参数
        uniformity_index = self._calculate_uniformity_index(intensity)
        distribution_variance = float(np.var(intensity))
        center_shift = (float(center_x), float(center_y))
        shape_factor = sigma_x / sigma_y if sigma_y > 0 else 1.0
        irregularity_score = self._calculate_irregularity_score(intensity)
        
        return SpotMorphologyData(
            timestamp=time.time(),
            uniformity_index=uniformity_index,
            distribution_variance=distribution_variance,
            intensity_profile=intensity.flatten(),
            center_shift=center_shift,
            shape_factor=shape_factor,
            irregularity_score=irregularity_score
        )
    
    def _calculate_uniformity_index(self, intensity: np.ndarray) -> float:
        """计算光斑均匀度指数"""
        # 使用强度分布的标准差来评估均匀度
        normalized_intensity = intensity / np.max(intensity)
        std_dev = np.std(normalized_intensity)
        uniformity = np.exp(-std_dev * 5)  # 经验公式
        return float(np.clip(uniformity, 0, 1))
    
    def _calculate_irregularity_score(self, intensity: np.ndarray) -> float:
        """计算不规则性评分"""
        # 计算强度分布的偏斜度和峰度来评估不规则性
        from scipy import stats
        
        flat_intensity = intensity.flatten()
        skewness = abs(stats.skew(flat_intensity))
        kurtosis = abs(stats.kurtosis(flat_intensity))
        
        # 组合偏斜度和峰度得到不规则性评分
        irregularity = (skewness + kurtosis) / 10.0  # 归一化
        return float(np.clip(irregularity, 0, 1))
    
    def detect_dendrite_risk(self) -> Tuple[DendriteRiskLevel, List[DendriteAlert]]:
        """检测锂枝晶风险"""
        if self.baseline_reflectance is None:
            raise ValueError("请先建立基线数据")
        
        alerts = []
        risk_scores = []
        
        # 当前测量
        current_measurement = self.spr_monitor.measure_at_angle(self.optimal_angle)
        current_morphology = self._measure_spot_morphology()
        
        # 检查反射率变化
        reflectance_change_percent = abs(
            (current_measurement.reflectance - self.baseline_reflectance) / 
            self.baseline_reflectance * 100
        )
        
        if reflectance_change_percent > self.config.reflectance_threshold_percent:
            confidence = min(reflectance_change_percent / self.config.reflectance_threshold_percent, 1.0)
            risk_scores.append(confidence * 0.4)  # 权重0.4
            
            alert = DendriteAlert(
                timestamp=time.time(),
                alert_type=AlertType.REFLECTANCE_CHANGE,
                risk_level=self._determine_risk_level(confidence),
                message=f"反射率变化超出阈值: {reflectance_change_percent:.3f}%",
                measured_value=reflectance_change_percent,
                threshold_value=self.config.reflectance_threshold_percent,
                confidence=confidence,
                recommended_action="监控电极表面状态，检查是否有微小形态变化"
            )
            alerts.append(alert)
        
        # 检查相位变化
        phase_change_deg = abs(
            np.rad2deg(current_measurement.phase_shift_rad - self.baseline_phase)
        )
        
        if phase_change_deg > self.config.phase_threshold_deg:
            confidence = min(phase_change_deg / self.config.phase_threshold_deg, 1.0)
            risk_scores.append(confidence * 0.3)  # 权重0.3
            
            alert = DendriteAlert(
                timestamp=time.time(),
                alert_type=AlertType.PHASE_CHANGE,
                risk_level=self._determine_risk_level(confidence),
                message=f"相位变化超出阈值: {phase_change_deg:.3f}°",
                measured_value=phase_change_deg,
                threshold_value=self.config.phase_threshold_deg,
                confidence=confidence,
                recommended_action="检查电极表面微裂纹或突起"
            )
            alerts.append(alert)
        
        # 检查光斑形态
        if current_morphology.uniformity_index < self.config.uniformity_threshold:
            confidence = (self.config.uniformity_threshold - current_morphology.uniformity_index) / self.config.uniformity_threshold
            risk_scores.append(confidence * 0.2)  # 权重0.2
            
            alert = DendriteAlert(
                timestamp=time.time(),
                alert_type=AlertType.MORPHOLOGY_CHANGE,
                risk_level=self._determine_risk_level(confidence),
                message=f"光斑均匀度下降: {current_morphology.uniformity_index:.3f}",
                measured_value=current_morphology.uniformity_index,
                threshold_value=self.config.uniformity_threshold,
                confidence=confidence,
                recommended_action="检查光学系统和电极表面质量"
            )
            alerts.append(alert)
        
        if current_morphology.irregularity_score > self.config.irregularity_threshold:
            confidence = min(current_morphology.irregularity_score / self.config.irregularity_threshold, 1.0)
            risk_scores.append(confidence * 0.3)  # 权重0.3
            
            alert = DendriteAlert(
                timestamp=time.time(),
                alert_type=AlertType.SURFACE_IRREGULARITY,
                risk_level=self._determine_risk_level(confidence),
                message=f"表面不规则性增加: {current_morphology.irregularity_score:.3f}",
                measured_value=current_morphology.irregularity_score,
                threshold_value=self.config.irregularity_threshold,
                confidence=confidence,
                recommended_action="检查电极表面是否出现不规则形状，可能是枝晶形成的初步信号"
            )
            alerts.append(alert)
        
        # 确定总体风险等级
        if risk_scores:
            overall_risk_score = max(risk_scores)
            overall_risk_level = self._determine_risk_level(overall_risk_score)
        else:
            overall_risk_level = DendriteRiskLevel.LOW
        
        # 如果有多个高风险指标，提升风险等级
        high_risk_count = sum(1 for alert in alerts if alert.risk_level in [DendriteRiskLevel.HIGH, DendriteRiskLevel.CRITICAL])
        if high_risk_count >= 2:
            overall_risk_level = DendriteRiskLevel.CRITICAL
            
            dendrite_alert = DendriteAlert(
                timestamp=time.time(),
                alert_type=AlertType.DENDRITE_FORMATION,
                risk_level=DendriteRiskLevel.CRITICAL,
                message=f"检测到{high_risk_count}个高风险指标，锂枝晶形成风险极高",
                measured_value=float(high_risk_count),
                threshold_value=2.0,
                confidence=0.9,
                recommended_action="立即停止充电，检查电极状态，采取预防措施"
            )
            alerts.append(dendrite_alert)
        
        # 存储历史数据
        self.reflectance_history.append((time.time(), current_measurement.reflectance))
        self.phase_history.append((time.time(), current_measurement.phase_shift_rad))
        self.morphology_history.append(current_morphology)
        self.alert_history.extend(alerts)
        
        # 触发回调
        if self.config.alert_enabled and self.config.alert_callback:
            for alert in alerts:
                self.config.alert_callback(alert)
        
        return overall_risk_level, alerts
    
    def _determine_risk_level(self, confidence: float) -> DendriteRiskLevel:
        """根据置信度确定风险等级"""
        if confidence < 0.3:
            return DendriteRiskLevel.LOW
        elif confidence < 0.6:
            return DendriteRiskLevel.MEDIUM
        elif confidence < 0.8:
            return DendriteRiskLevel.HIGH
        else:
            return DendriteRiskLevel.CRITICAL
    
    def start_realtime_monitoring(self):
        """开始实时监控"""
        if self.monitoring_active:
            self.logger.warning("监控已在运行中")
            return
        
        if self.baseline_reflectance is None:
            raise ValueError("请先建立基线数据")
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"开始实时监控，采样频率: {self.config.sampling_rate_hz} Hz")
    
    def stop_realtime_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        self.logger.info("实时监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        sample_interval = 1.0 / self.config.sampling_rate_hz
        
        while self.monitoring_active:
            start_time = time.time()
            
            try:
                # 检测锂枝晶风险
                risk_level, alerts = self.detect_dendrite_risk()
                
                # 将数据放入队列供外部获取
                monitoring_data = {
                    "timestamp": time.time(),
                    "risk_level": risk_level,
                    "alerts": alerts,
                    "reflectance": self.reflectance_history[-1][1] if self.reflectance_history else None,
                    "phase": self.phase_history[-1][1] if self.phase_history else None,
                    "morphology": self.morphology_history[-1] if self.morphology_history else None
                }
                self.data_queue.put(monitoring_data)
                
                # 自动保存数据
                if (self.config.auto_save_enabled and 
                    time.time() - self.last_save_time > self.config.save_interval_seconds):
                    self._auto_save_data()
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
            
            # 控制采样频率
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            time.sleep(sleep_time)
    
    def get_realtime_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """获取实时监控数据"""
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None
    
    def _auto_save_data(self):
        """自动保存数据"""
        try:
            self.save_monitoring_data("auto_save")
            self.last_save_time = time.time()
            self.logger.info("自动保存数据完成")
        except Exception as e:
            self.logger.error(f"自动保存失败: {e}")
    
    def save_monitoring_data(self, filename_prefix: str = "dendrite_monitoring"):
        """保存监测数据"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式数据
        json_filename = f"{filename_prefix}_{timestamp}.json"
        self._save_json_data(json_filename)
        
        # 保存CSV格式数据
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        self._save_csv_data(csv_filename)
    
    def _save_json_data(self, filename: str):
        """保存JSON格式数据"""
        data = {
            "config": {
                "reflectance_threshold_percent": self.config.reflectance_threshold_percent,
                "phase_threshold_deg": self.config.phase_threshold_deg,
                "uniformity_threshold": self.config.uniformity_threshold,
                "irregularity_threshold": self.config.irregularity_threshold,
                "sampling_rate_hz": self.config.sampling_rate_hz
            },
            "baseline_data": {
                "reflectance": self.baseline_reflectance,
                "phase_rad": self.baseline_phase,
                "optimal_angle_deg": self.optimal_angle
            },
            "reflectance_history": self.reflectance_history,
            "phase_history": self.phase_history,
            "alerts": [
                {
                    "timestamp": alert.timestamp,
                    "type": alert.alert_type.value,
                    "risk_level": alert.risk_level.value,
                    "message": alert.message,
                    "measured_value": alert.measured_value,
                    "threshold_value": alert.threshold_value,
                    "confidence": alert.confidence,
                    "recommended_action": alert.recommended_action
                }
                for alert in self.alert_history
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv_data(self, filename: str):
        """保存CSV格式数据"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                "timestamp", "type", "risk_level", "message", "measured_value", 
                "threshold_value", "confidence", "recommended_action"
            ])
            
            # 写入警报数据
            for alert in self.alert_history:
                writer.writerow([
                    alert.timestamp,
                    alert.alert_type.value,
                    alert.risk_level.value,
                    alert.message,
                    alert.measured_value,
                    alert.threshold_value,
                    alert.confidence,
                    alert.recommended_action
                ])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取监测统计信息"""
        if not self.alert_history:
            return {"message": "无监测数据"}
        
        # 统计警报类型
        alert_type_counts = {}
        risk_level_counts = {}
        
        for alert in self.alert_history:
            alert_type = alert.alert_type.value
            risk_level = alert.risk_level.value
            
            alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1
            risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1
        
        # 最近的风险趋势
        recent_alerts = self.alert_history[-10:] if len(self.alert_history) >= 10 else self.alert_history
        recent_risk_levels = [alert.risk_level.value for alert in recent_alerts]
        
        return {
            "total_alerts": len(self.alert_history),
            "alert_type_distribution": alert_type_counts,
            "risk_level_distribution": risk_level_counts,
            "recent_risk_trend": recent_risk_levels,
            "monitoring_duration_hours": (time.time() - self.alert_history[0].timestamp) / 3600 if self.alert_history else 0,
            "data_points_collected": len(self.reflectance_history),
            "baseline_reflectance": self.baseline_reflectance,
            "baseline_phase_deg": np.rad2deg(self.baseline_phase) if self.baseline_phase else None
        }
    
    def clear_history(self):
        """清除历史数据"""
        self.reflectance_history.clear()
        self.phase_history.clear()
        self.morphology_history.clear()
        self.alert_history.clear()
        
        # 清空数据队列
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                break
        
        self.logger.info("历史数据已清除") 