"""SEI膜电极表面反应监测模块"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import time
from pathlib import Path
import sys
from enum import Enum
import warnings

# 导入现有optics模块
sys.path.append(str(Path(__file__).parent.parent / "Multiscale-Photonic-Diagnostics-for-Solid-State-Batteries-main"))
from optics.elements import MetalFilm
from optics.materials import gold_n_complex_scalar
from optics.ray import Ray

from .electrolyte_materials import ElectrolyteMaterials, ElectrolyteType


class ElectrodeType(Enum):
    """电极类型"""
    ANODE = "anode"          # 负极
    CATHODE = "cathode"      # 正极


class SEIFormationStage(Enum):
    """SEI膜形成阶段"""
    INITIAL = "initial"           # 初始状态
    FORMATION = "formation"       # 形成期
    GROWTH = "growth"            # 生长期
    STABILIZATION = "stabilization"  # 稳定期
    DEGRADATION = "degradation"   # 降解期


class ChargeDischargeState(Enum):
    """充放电状态"""
    IDLE = "idle"                # 静置
    CHARGING = "charging"        # 充电
    DISCHARGING = "discharging"  # 放电
    PAUSE = "pause"             # 暂停


@dataclass
class SEIMeasurement:
    """SEI膜测量数据结构"""
    timestamp: float
    # SPR测量参数
    incident_angle_deg: float
    wavelength_nm: float
    reflectance: float
    phase_shift_rad: float
    
    # SEI膜特性
    sei_thickness_nm: float
    sei_refractive_index: float
    thickness_error_nm: float
    
    # 电池状态
    charge_discharge_state: ChargeDischargeState
    voltage_v: float
    current_a: float
    capacity_ah: float
    
    # 变化检测
    reflectance_change: float
    thickness_change_nm: float
    formation_stage: SEIFormationStage
    
    # 质量指标
    measurement_confidence: float  # 0-1
    signal_noise_ratio: float


@dataclass
class SEIMonitorConfig:
    """SEI监测配置"""
    # 阈值设置
    reflectance_change_threshold: float = 0.02  # ±0.02
    thickness_change_alert_nm: float = 10.0     # 10nm
    thickness_measurement_error_nm: float = 5.0  # ±5nm
    
    # SEI膜参数范围
    sei_n_min: float = 1.45  # SEI膜最小折射率
    sei_n_max: float = 1.55  # SEI膜最大折射率
    sei_n_nominal: float = 1.50  # 标称折射率
    
    # 监测参数
    sampling_rate_hz: float = 1.0  # 采样频率
    measurement_averaging: int = 5  # 测量平均次数
    
    # SPR设置
    wavelength_nm: float = 632.8
    angle_range_deg: Tuple[float, float] = (40.0, 75.0)
    angle_resolution_deg: float = 0.1


@dataclass
class ElectrodeProperties:
    """电极材料属性"""
    electrode_type: ElectrodeType
    material_name: str
    base_refractive_index: float
    surface_roughness_nm: float
    initial_sei_thickness_nm: float = 0.0


class SEIElectrodeMonitor:
    """SEI膜电极表面反应监测器"""
    
    def __init__(
        self,
        electrode_props: ElectrodeProperties,
        config: Optional[SEIMonitorConfig] = None,
        metal_film_thickness_nm: float = 50.0,
        prism_n: float = 1.515
    ):
        """初始化SEI膜电极监测器"""
        self.electrode_props = electrode_props
        self.config = config or SEIMonitorConfig()
        self.metal_film_thickness = metal_film_thickness_nm
        self.prism_n = prism_n
        
        # 监测状态
        self.is_monitoring = False
        self.current_cd_state = ChargeDischargeState.IDLE
        
        # 测量历史
        self.measurement_history: List[SEIMeasurement] = []
        self.baseline_measurement: Optional[SEIMeasurement] = None
        
        # 回调函数
        self.alert_callbacks: List[Callable[[str, SEIMeasurement], None]] = []
        
        # 初始化系统
        self._setup_monitoring_system()
    
    def _setup_monitoring_system(self):
        """设置监测系统"""
        # 创建SPR金属膜系统
        self.metal_film = MetalFilm(
            thickness_nm=self.metal_film_thickness,
            n_metal=gold_n_complex_scalar,
            n_prism=self.prism_n,
            n_sample=self.config.sei_n_nominal  # 初始使用标称SEI折射率
        )
        
        # 建立基线测量
        self._establish_baseline()
    
    def _establish_baseline(self):
        """建立基线测量"""
        print("正在建立SEI监测基线...")
        
        # 进行多次测量求平均
        measurements = []
        for i in range(self.config.measurement_averaging):
            measurement = self._perform_single_measurement()
            measurements.append(measurement)
            time.sleep(0.1)  # 短暂间隔
        
        # 计算平均值作为基线
        avg_reflectance = np.mean([m.reflectance for m in measurements])
        avg_phase = np.mean([m.phase_shift_rad for m in measurements])
        
        self.baseline_measurement = SEIMeasurement(
            timestamp=time.time(),
            incident_angle_deg=self._find_optimal_angle(),
            wavelength_nm=self.config.wavelength_nm,
            reflectance=avg_reflectance,
            phase_shift_rad=avg_phase,
            sei_thickness_nm=self.electrode_props.initial_sei_thickness_nm,
            sei_refractive_index=self.config.sei_n_nominal,
            thickness_error_nm=0.0,
            charge_discharge_state=ChargeDischargeState.IDLE,
            voltage_v=0.0,
            current_a=0.0,
            capacity_ah=0.0,
            reflectance_change=0.0,
            thickness_change_nm=0.0,
            formation_stage=SEIFormationStage.INITIAL,
            measurement_confidence=1.0,
            signal_noise_ratio=50.0
        )
        
        print(f"基线建立完成 - 反射率: {avg_reflectance:.4f}, 相位: {avg_phase:.4f} rad")
    
    def _find_optimal_angle(self) -> float:
        """找到最佳监测角度"""
        angles = np.arange(
            self.config.angle_range_deg[0],
            self.config.angle_range_deg[1],
            self.config.angle_resolution_deg
        )
        
        reflectances = []
        for angle in angles:
            ray = self._create_ray(angle)
            R = self.metal_film.get_reflectance(ray)
            reflectances.append(R)
        
        # 找到最大灵敏度点（反射率梯度最大处）
        gradient = np.gradient(reflectances)
        optimal_idx = np.argmax(np.abs(gradient))
        
        return float(angles[optimal_idx])
    
    def _create_ray(self, angle_deg: float) -> Ray:
        """创建指定角度的光线"""
        angle_rad = np.deg2rad(angle_deg)
        return Ray(
            position=np.array([0.0, 0.0]),
            direction=np.array([np.sin(angle_rad), -np.cos(angle_rad)]),
            wavelength=self.config.wavelength_nm,
            polarization="TM",
            intensity=1.0
        )
    
    def _perform_single_measurement(self) -> SEIMeasurement:
        """执行单次SPR测量"""
        if self.baseline_measurement:
            angle = self.baseline_measurement.incident_angle_deg
        else:
            angle = self._find_optimal_angle()
        
        ray = self._create_ray(angle)
        
        # SPR测量
        reflectance = self.metal_film.get_reflectance(ray)
        phase_shift = self.metal_film.get_phase_shift(
            self.config.wavelength_nm, np.deg2rad(angle)
        )
        
        # 计算SEI膜参数
        sei_thickness, sei_n, thickness_error = self._calculate_sei_parameters(
            reflectance, phase_shift
        )
        
        # 计算变化量
        reflectance_change = 0.0
        thickness_change = 0.0
        if self.baseline_measurement:
            reflectance_change = reflectance - self.baseline_measurement.reflectance
            thickness_change = sei_thickness - self.baseline_measurement.sei_thickness_nm
        
        # 判断SEI形成阶段
        formation_stage = self._determine_formation_stage(
            sei_thickness, thickness_change, reflectance_change
        )
        
        # 计算测量置信度
        confidence = self._calculate_measurement_confidence(
            reflectance, phase_shift, thickness_error
        )
        
        measurement = SEIMeasurement(
            timestamp=time.time(),
            incident_angle_deg=angle,
            wavelength_nm=self.config.wavelength_nm,
            reflectance=reflectance,
            phase_shift_rad=phase_shift,
            sei_thickness_nm=sei_thickness,
            sei_refractive_index=sei_n,
            thickness_error_nm=thickness_error,
            charge_discharge_state=self.current_cd_state,
            voltage_v=0.0,  # 需要外部提供
            current_a=0.0,  # 需要外部提供
            capacity_ah=0.0,  # 需要外部提供
            reflectance_change=reflectance_change,
            thickness_change_nm=thickness_change,
            formation_stage=formation_stage,
            measurement_confidence=confidence,
            signal_noise_ratio=40.0 + np.random.normal(0, 5)  # 模拟信噪比
        )
        
        return measurement
    
    def _calculate_sei_parameters(self, reflectance: float, phase_shift: float) -> Tuple[float, float, float]:
        """基于反射率和相位计算SEI膜参数"""
        # 简化的SEI膜厚度计算模型
        
        if not self.baseline_measurement:
            return 0.0, self.config.sei_n_nominal, 0.0
        
        # 基于反射率变化估算厚度
        R_change = abs(reflectance - self.baseline_measurement.reflectance)
        
        # 经验关系：反射率变化与厚度的关系
        thickness_sensitivity = 0.001  # nm^-1
        estimated_thickness = self.baseline_measurement.sei_thickness_nm + \
                            R_change / thickness_sensitivity
        
        # 基于相位变化估算折射率
        phase_change = phase_shift - self.baseline_measurement.phase_shift_rad
        n_change = phase_change * self.config.wavelength_nm / (4 * np.pi * estimated_thickness + 1e-10)
        estimated_n = self.config.sei_n_nominal + n_change
        
        # 约束折射率范围
        estimated_n = np.clip(estimated_n, self.config.sei_n_min, self.config.sei_n_max)
        
        # 估算厚度误差
        thickness_error = min(self.config.thickness_measurement_error_nm, 
                            estimated_thickness * 0.05)  # 5%相对误差
        
        return float(estimated_thickness), float(estimated_n), float(thickness_error)
    
    def _determine_formation_stage(self, thickness_nm: float, 
                                 thickness_change: float, 
                                 reflectance_change: float) -> SEIFormationStage:
        """判断SEI膜形成阶段"""
        if thickness_nm < 1.0:
            return SEIFormationStage.INITIAL
        elif thickness_change > 2.0 and abs(reflectance_change) > 0.01:
            return SEIFormationStage.FORMATION
        elif thickness_change > 0.5:
            return SEIFormationStage.GROWTH
        elif abs(thickness_change) < 0.2:
            return SEIFormationStage.STABILIZATION
        elif thickness_change < -1.0:
            return SEIFormationStage.DEGRADATION
        else:
            return SEIFormationStage.GROWTH
    
    def _calculate_measurement_confidence(self, reflectance: float, 
                                        phase_shift: float, 
                                        thickness_error: float) -> float:
        """计算测量置信度"""
        # 基于多个因素计算置信度
        confidence = 1.0
        
        # 反射率信号强度
        if reflectance < 0.01 or reflectance > 0.99:
            confidence *= 0.7  # 极值信号置信度降低
        
        # 厚度误差影响
        error_factor = 1.0 - (thickness_error / self.config.thickness_measurement_error_nm)
        confidence *= max(0.1, error_factor)
        
        # 相位信号稳定性
        if abs(phase_shift) > np.pi:
            confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))
    
    def add_alert_callback(self, callback: Callable[[str, SEIMeasurement], None]):
        """添加报警回调函数"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, measurement: SEIMeasurement):
        """触发报警"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, measurement)
            except Exception as e:
                warnings.warn(f"报警回调执行失败: {e}")
    
    def _check_alerts(self, measurement: SEIMeasurement):
        """检查报警条件"""
        # 反射率变化报警
        if abs(measurement.reflectance_change) > self.config.reflectance_change_threshold:
            self._trigger_alert("REFLECTANCE_CHANGE", measurement)
        
        # 厚度变化报警
        if abs(measurement.thickness_change_nm) > self.config.thickness_change_alert_nm:
            self._trigger_alert("THICKNESS_CHANGE", measurement)
        
        # 测量精度报警
        if measurement.thickness_error_nm > self.config.thickness_measurement_error_nm:
            self._trigger_alert("MEASUREMENT_ERROR", measurement)
        
        # SEI膜异常报警
        if (measurement.formation_stage == SEIFormationStage.DEGRADATION and 
            measurement.thickness_change_nm < -5.0):
            self._trigger_alert("SEI_DEGRADATION", measurement)
        
        # 置信度低报警
        if measurement.measurement_confidence < 0.3:
            self._trigger_alert("LOW_CONFIDENCE", measurement) 

    def start_monitoring(self, charge_discharge_state: ChargeDischargeState = ChargeDischargeState.IDLE):
        """开始SEI监测"""
        self.is_monitoring = True
        self.current_cd_state = charge_discharge_state
        print(f"开始SEI膜监测 - 状态: {charge_discharge_state.value}")
    
    def stop_monitoring(self):
        """停止SEI监测"""
        self.is_monitoring = False
        print("SEI膜监测已停止")
    
    def update_battery_state(self, voltage_v: float, current_a: float, 
                           capacity_ah: float, cd_state: ChargeDischargeState):
        """更新电池状态"""
        self.current_cd_state = cd_state
        
        # 如果正在监测，更新最新测量的电池参数
        if self.measurement_history and self.is_monitoring:
            latest = self.measurement_history[-1]
            # 创建更新的测量记录
            updated_measurement = SEIMeasurement(
                **{k: v for k, v in latest.__dict__.items() 
                   if k not in ['voltage_v', 'current_a', 'capacity_ah', 'charge_discharge_state']},
                voltage_v=voltage_v,
                current_a=current_a,
                capacity_ah=capacity_ah,
                charge_discharge_state=cd_state
            )
            self.measurement_history[-1] = updated_measurement
    
    def measure_sei_state(self) -> SEIMeasurement:
        """执行单次SEI状态测量"""
        if not self.is_monitoring:
            self.start_monitoring()
        
        measurement = self._perform_single_measurement()
        self.measurement_history.append(measurement)
        
        # 检查报警条件
        self._check_alerts(measurement)
        
        return measurement
    
    def continuous_monitoring(self, duration_seconds: float, 
                            callback: Optional[Callable[[SEIMeasurement], None]] = None):
        """连续监测SEI状态"""
        if not self.is_monitoring:
            self.start_monitoring()
        
        start_time = time.time()
        interval = 1.0 / self.config.sampling_rate_hz
        
        print(f"开始连续监测 {duration_seconds} 秒...")
        
        while (time.time() - start_time) < duration_seconds and self.is_monitoring:
            measurement = self.measure_sei_state()
            
            if callback:
                callback(measurement)
            
            # 打印关键信息
            self._print_measurement_summary(measurement)
            
            time.sleep(interval)
        
        print("连续监测完成")
    
    def _print_measurement_summary(self, measurement: SEIMeasurement):
        """打印测量摘要"""
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"SEI厚度: {measurement.sei_thickness_nm:.1f}nm "
              f"(±{measurement.thickness_error_nm:.1f}nm) | "
              f"反射率变化: {measurement.reflectance_change:+.4f} | "
              f"形成阶段: {measurement.formation_stage.value} | "
              f"置信度: {measurement.measurement_confidence:.2f}")
    
    def get_sei_trend_analysis(self, window_minutes: float = 5.0) -> Dict[str, Any]:
        """分析SEI膜变化趋势"""
        if len(self.measurement_history) < 2:
            return {"error": "数据不足，无法进行趋势分析"}
        
        # 筛选时间窗口内的数据
        current_time = time.time()
        window_seconds = window_minutes * 60
        recent_data = [
            m for m in self.measurement_history
            if (current_time - m.timestamp) <= window_seconds
        ]
        
        if len(recent_data) < 2:
            return {"error": "时间窗口内数据不足"}
        
        # 提取关键参数
        timestamps = [m.timestamp for m in recent_data]
        thicknesses = [m.sei_thickness_nm for m in recent_data]
        reflectances = [m.reflectance for m in recent_data]
        
        # 计算趋势
        time_points = [(t - timestamps[0]) / 60 for t in timestamps]  # 转换为分钟
        
        # 厚度变化趋势
        thickness_trend = np.polyfit(time_points, thicknesses, 1)[0]  # nm/min
        
        # 反射率变化趋势
        reflectance_trend = np.polyfit(time_points, reflectances, 1)[0]
        
        # 变化统计
        thickness_change_total = thicknesses[-1] - thicknesses[0]
        reflectance_change_total = reflectances[-1] - reflectances[0]
        
        # 稳定性评估
        thickness_std = np.std(thicknesses)
        thickness_stability = "稳定" if thickness_std < 1.0 else "不稳定"
        
        return {
            "analysis_window_min": window_minutes,
            "data_points": len(recent_data),
            "thickness_trend_nm_per_min": float(thickness_trend),
            "reflectance_trend_per_min": float(reflectance_trend),
            "total_thickness_change_nm": float(thickness_change_total),
            "total_reflectance_change": float(reflectance_change_total),
            "thickness_stability": thickness_stability,
            "current_formation_stage": recent_data[-1].formation_stage.value,
            "average_confidence": float(np.mean([m.measurement_confidence for m in recent_data]))
        }
    
    def detect_reaction_events(self, sensitivity: float = 0.5) -> List[Dict[str, Any]]:
        """检测电极表面反应事件"""
        if len(self.measurement_history) < 10:
            return []
        
        events = []
        
        # 设置检测阈值
        thickness_threshold = 2.0 * sensitivity  # nm
        reflectance_threshold = 0.01 * sensitivity
        
        for i in range(1, len(self.measurement_history)):
            current = self.measurement_history[i]
            previous = self.measurement_history[i-1]
            
            # 厚度突变检测
            thickness_jump = abs(current.sei_thickness_nm - previous.sei_thickness_nm)
            if thickness_jump > thickness_threshold:
                events.append({
                    "timestamp": current.timestamp,
                    "type": "thickness_jump",
                    "magnitude": thickness_jump,
                    "direction": "increase" if current.sei_thickness_nm > previous.sei_thickness_nm else "decrease",
                    "charge_state": current.charge_discharge_state.value
                })
            
            # 反射率突变检测
            reflectance_jump = abs(current.reflectance - previous.reflectance)
            if reflectance_jump > reflectance_threshold:
                events.append({
                    "timestamp": current.timestamp,
                    "type": "reflectance_jump",
                    "magnitude": reflectance_jump,
                    "direction": "increase" if current.reflectance > previous.reflectance else "decrease",
                    "charge_state": current.charge_discharge_state.value
                })
            
            # SEI形成阶段变化检测
            if current.formation_stage != previous.formation_stage:
                events.append({
                    "timestamp": current.timestamp,
                    "type": "formation_stage_change",
                    "from_stage": previous.formation_stage.value,
                    "to_stage": current.formation_stage.value,
                    "charge_state": current.charge_discharge_state.value
                })
        
        return events
    
    def get_measurement_statistics(self, last_n_measurements: int = 100) -> Dict[str, Any]:
        """获取测量统计信息"""
        if not self.measurement_history:
            return {"error": "无测量数据"}
        
        recent_data = self.measurement_history[-last_n_measurements:]
        
        # 基础统计
        thicknesses = [m.sei_thickness_nm for m in recent_data]
        reflectances = [m.reflectance for m in recent_data]
        errors = [m.thickness_error_nm for m in recent_data]
        confidences = [m.measurement_confidence for m in recent_data]
        
        # 按充放电状态分组统计
        state_stats = {}
        for state in ChargeDischargeState:
            state_data = [m for m in recent_data if m.charge_discharge_state == state]
            if state_data:
                state_thicknesses = [m.sei_thickness_nm for m in state_data]
                state_stats[state.value] = {
                    "count": len(state_data),
                    "mean_thickness_nm": float(np.mean(state_thicknesses)),
                    "std_thickness_nm": float(np.std(state_thicknesses))
                }
        
        return {
            "total_measurements": len(recent_data),
            "time_span_hours": (recent_data[-1].timestamp - recent_data[0].timestamp) / 3600,
            "thickness_stats": {
                "mean_nm": float(np.mean(thicknesses)),
                "std_nm": float(np.std(thicknesses)),
                "min_nm": float(np.min(thicknesses)),
                "max_nm": float(np.max(thicknesses)),
                "range_nm": float(np.max(thicknesses) - np.min(thicknesses))
            },
            "reflectance_stats": {
                "mean": float(np.mean(reflectances)),
                "std": float(np.std(reflectances)),
                "min": float(np.min(reflectances)),
                "max": float(np.max(reflectances))
            },
            "measurement_quality": {
                "mean_error_nm": float(np.mean(errors)),
                "mean_confidence": float(np.mean(confidences)),
                "high_confidence_ratio": float(np.mean([c > 0.7 for c in confidences]))
            },
            "state_statistics": state_stats
        }
    
    def export_sei_data(self, filepath: str, include_raw_data: bool = True):
        """导出SEI监测数据"""
        import json
        
        export_data = {
            "monitor_info": {
                "electrode_type": self.electrode_props.electrode_type.value,
                "electrode_material": self.electrode_props.material_name,
                "monitor_config": {
                    "reflectance_threshold": self.config.reflectance_change_threshold,
                    "thickness_alert_nm": self.config.thickness_change_alert_nm,
                    "thickness_error_nm": self.config.thickness_measurement_error_nm,
                    "sei_n_range": [self.config.sei_n_min, self.config.sei_n_max],
                    "sampling_rate_hz": self.config.sampling_rate_hz
                }
            },
            "baseline_measurement": None,
            "statistics": self.get_measurement_statistics(),
            "trend_analysis": self.get_sei_trend_analysis(),
            "reaction_events": self.detect_reaction_events()
        }
        
        if self.baseline_measurement:
            export_data["baseline_measurement"] = {
                "timestamp": self.baseline_measurement.timestamp,
                "reflectance": self.baseline_measurement.reflectance,
                "phase_shift_rad": self.baseline_measurement.phase_shift_rad,
                "sei_thickness_nm": self.baseline_measurement.sei_thickness_nm
            }
        
        if include_raw_data:
            export_data["measurements"] = [
                {
                    "timestamp": m.timestamp,
                    "incident_angle_deg": m.incident_angle_deg,
                    "reflectance": m.reflectance,
                    "phase_shift_rad": m.phase_shift_rad,
                    "sei_thickness_nm": m.sei_thickness_nm,
                    "sei_refractive_index": m.sei_refractive_index,
                    "thickness_error_nm": m.thickness_error_nm,
                    "charge_discharge_state": m.charge_discharge_state.value,
                    "voltage_v": m.voltage_v,
                    "current_a": m.current_a,
                    "reflectance_change": m.reflectance_change,
                    "thickness_change_nm": m.thickness_change_nm,
                    "formation_stage": m.formation_stage.value,
                    "measurement_confidence": m.measurement_confidence
                }
                for m in self.measurement_history
            ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"SEI监测数据已导出到: {filepath}")
    
    def clear_measurement_history(self):
        """清除测量历史"""
        self.measurement_history.clear()
        print("测量历史已清除")
    
    def recalibrate_baseline(self):
        """重新校准基线"""
        print("重新校准SEI监测基线...")
        self._establish_baseline()


# 默认报警回调函数
def default_alert_handler(alert_type: str, measurement: SEIMeasurement):
    """默认报警处理函数"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(measurement.timestamp))
    
    alert_messages = {
        "REFLECTANCE_CHANGE": f"反射率变化异常: {measurement.reflectance_change:+.4f}",
        "THICKNESS_CHANGE": f"SEI膜厚度变化异常: {measurement.thickness_change_nm:+.1f}nm",
        "MEASUREMENT_ERROR": f"测量误差过大: ±{measurement.thickness_error_nm:.1f}nm",
        "SEI_DEGRADATION": f"SEI膜降解检测: 厚度减少{abs(measurement.thickness_change_nm):.1f}nm",
        "LOW_CONFIDENCE": f"测量置信度过低: {measurement.measurement_confidence:.2f}"
    }
    
    message = alert_messages.get(alert_type, f"未知报警类型: {alert_type}")
    
    print(f" [{timestamp}] SEI监测报警 - {message}")
    print(f"   当前状态: {measurement.charge_discharge_state.value}")
    print(f"   SEI厚度: {measurement.sei_thickness_nm:.1f}nm")
    print(f"   形成阶段: {measurement.formation_stage.value}") 