"""报警系统模块"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging
from pathlib import Path
import threading
from queue import Queue
import warnings

from .data_processor import ProcessedMeasurement


class AlertLevel(Enum):
    """报警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """报警类型"""
    THICKNESS_DEVIATION = "thickness_deviation"
    SURFACE_QUALITY = "surface_quality"
    SYSTEM_ERROR = "system_error"
    MEASUREMENT_ANOMALY = "measurement_anomaly"
    EQUIPMENT_FAILURE = "equipment_failure"


@dataclass
class AlertRule:
    """报警规则"""
    rule_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    condition: str  # 条件表达式字符串
    threshold_value: float
    comparison_operator: str  # ">", "<", ">=", "<=", "==", "!="
    description: str
    enabled: bool = True
    cooldown_seconds: float = 300.0  # 冷却时间，防止重复报警


@dataclass
class Alert:
    """报警记录"""
    alert_id: str
    timestamp: float
    alert_type: AlertType
    alert_level: AlertLevel
    rule_id: str
    message: str
    measurement_id: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = None
    acknowledged: bool = False
    resolved: bool = False


class AlertHandler:
    """报警处理器基类"""
    
    def handle_alert(self, alert: Alert) -> bool:
        """处理报警，返回是否成功"""
        raise NotImplementedError


class ConsoleAlertHandler(AlertHandler):
    """控制台报警处理器"""
    
    def handle_alert(self, alert: Alert) -> bool:
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))
        print(f"[{timestamp_str}] {alert.alert_level.value.upper()}: {alert.message}")
        if alert.details:
            print(f"    详情: {alert.details}")
        return True


class FileAlertHandler(AlertHandler):
    """文件报警处理器"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger("alert_system")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def handle_alert(self, alert: Alert) -> bool:
        try:
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }[alert.alert_level]
            
            message = f"{alert.message}"
            if alert.details:
                message += f" | 详情: {json.dumps(alert.details, ensure_ascii=False)}"
            
            self.logger.log(log_level, message)
            return True
        except Exception as e:
            warnings.warn(f"文件报警处理失败: {str(e)}")
            return False


class EmailAlertHandler(AlertHandler):
    """邮件报警处理器"""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        recipient_emails: List[str],
        sender_email: Optional[str] = None
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = sender_email or username
        self.recipient_emails = recipient_emails
    
    def handle_alert(self, alert: Alert) -> bool:
        try:
            # 创建邮件
            msg = MimeMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)
            msg['Subject'] = f"[电解质监测] {alert.alert_level.value.upper()}: {alert.alert_type.value}"
            
            # 邮件正文
            body = f"""电解质层监测系统报警通知"""
            
            if alert.value is not None and alert.threshold is not None:
                body += f"- 测量值: {alert.value:.3f}\n- 阈值: {alert.threshold:.3f}\n"
            
            if alert.details:
                body += f"- 详细信息: {json.dumps(alert.details, indent=2, ensure_ascii=False)}\n"
            
            body += "\n请及时检查系统状态并采取相应措施。"
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            warnings.warn(f"邮件报警发送失败: {str(e)}")
            return False


class AlertSystem:
    """智能报警系统"""
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化报警系统"""
        # 报警规则和记录
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[AlertHandler] = []
        
        # 状态跟踪
        self.last_alert_times: Dict[str, float] = {}  # 用于冷却时间
        self.baseline_values: Dict[str, float] = {}  # 基准值
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 默认报警处理器
        self.add_handler(ConsoleAlertHandler())
        
        # 初始化默认规则
        self._setup_default_rules()
        
        # 加载配置
        if config_file:
            self.load_config(config_file)
    
    def _setup_default_rules(self):
        """设置默认报警规则"""
        # 厚度偏差报警（±50nm）
        self.add_rule(AlertRule(
            rule_id="thickness_deviation_50nm",
            alert_type=AlertType.THICKNESS_DEVIATION,
            alert_level=AlertLevel.WARNING,
            condition="abs(thickness_change)",
            threshold_value=50.0,
            comparison_operator=">=",
            description="电解质厚度变化超过±50nm",
            cooldown_seconds=60.0
        ))
        
        # 厚度偏差严重报警（±100nm）
        self.add_rule(AlertRule(
            rule_id="thickness_deviation_100nm",
            alert_type=AlertType.THICKNESS_DEVIATION,
            alert_level=AlertLevel.ERROR,
            condition="abs(thickness_change)",
            threshold_value=100.0,
            comparison_operator=">=",
            description="电解质厚度变化超过±100nm，可能存在严重问题"
        ))
        
        # 表面质量恶化报警
        self.add_rule(AlertRule(
            rule_id="surface_quality_degradation",
            alert_type=AlertType.SURFACE_QUALITY,
            alert_level=AlertLevel.WARNING,
            condition="uniformity_ratio",
            threshold_value=0.40,
            comparison_operator=">",
            description="表面均匀度恶化，超过基线水平"
        ))
        
        # 表面质量严重恶化报警
        self.add_rule(AlertRule(
            rule_id="surface_quality_critical",
            alert_type=AlertType.SURFACE_QUALITY,
            alert_level=AlertLevel.CRITICAL,
            condition="uniformity_ratio",
            threshold_value=0.50,
            comparison_operator=">",
            description="表面质量严重恶化，需要立即检查"
        ))
        
        # 测量异常报警
        self.add_rule(AlertRule(
            rule_id="measurement_confidence_low",
            alert_type=AlertType.MEASUREMENT_ANOMALY,
            alert_level=AlertLevel.WARNING,
            condition="thickness_confidence",
            threshold_value=0.5,
            comparison_operator="<",
            description="测量置信度过低，数据可能不可靠"
        ))
        
        # 反射率异常报警
        self.add_rule(AlertRule(
            rule_id="reflectance_anomaly",
            alert_type=AlertType.MEASUREMENT_ANOMALY,
            alert_level=AlertLevel.WARNING,
            condition="reflectance",
            threshold_value=0.95,
            comparison_operator=">",
            description="反射率异常高，可能存在测量问题"
        ))
    
    def add_rule(self, rule: AlertRule):
        """添加报警规则"""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str):
        """移除报警规则"""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
    
    def enable_rule(self, rule_id: str, enabled: bool = True):
        """启用/禁用报警规则"""
        with self._lock:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = enabled
    
    def add_handler(self, handler: AlertHandler):
        """添加报警处理器"""
        self.alert_handlers.append(handler)
    
    def set_baseline_value(self, parameter: str, value: float):
        """设置基准值"""
        with self._lock:
            self.baseline_values[parameter] = value
    
    def check_measurement(self, measurement: ProcessedMeasurement, 
                         previous_measurement: Optional[ProcessedMeasurement] = None) -> List[Alert]:
        """检查测量数据并生成报警"""
        alerts = []
        current_time = time.time()
        
        # 提取测量值
        measurement_values = {
            'thickness_nm': measurement.thickness_nm,
            'thickness_confidence': measurement.thickness_confidence,
            'uniformity_ratio': measurement.uniformity_ratio,
            'quality_score': measurement.quality_score,
            'reflectance': measurement.reflectance,
            'surface_roughness_nm': measurement.surface_roughness_nm
        }
        
        # 计算变化量（如果有前一次测量）
        if previous_measurement:
            measurement_values['thickness_change'] = abs(
                measurement.thickness_nm - previous_measurement.thickness_nm
            )
            measurement_values['quality_change'] = abs(
                measurement.quality_score - previous_measurement.quality_score
            )
        
        # 检查每个规则
        with self._lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # 检查冷却时间
                last_alert_time = self.last_alert_times.get(rule.rule_id, 0)
                if current_time - last_alert_time < rule.cooldown_seconds:
                    continue
                
                # 评估规则条件
                if self._evaluate_rule(rule, measurement_values):
                    alert = self._create_alert(rule, measurement, measurement_values)
                    alerts.append(alert)
                    self.last_alert_times[rule.rule_id] = current_time
        
        # 处理报警
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def _evaluate_rule(self, rule: AlertRule, values: Dict[str, float]) -> bool:
        """评估报警规则"""
        try:
            # 获取条件中的参数值
            if rule.condition not in values:
                return False
            
            measured_value = values[rule.condition]
            threshold = rule.threshold_value
            
            # 执行比较
            if rule.comparison_operator == ">":
                return measured_value > threshold
            elif rule.comparison_operator == "<":
                return measured_value < threshold
            elif rule.comparison_operator == ">=":
                return measured_value >= threshold
            elif rule.comparison_operator == "<=":
                return measured_value <= threshold
            elif rule.comparison_operator == "==":
                return abs(measured_value - threshold) < 1e-6
            elif rule.comparison_operator == "!=":
                return abs(measured_value - threshold) >= 1e-6
            else:
                warnings.warn(f"未知的比较操作符: {rule.comparison_operator}")
                return False
                
        except Exception as e:
            warnings.warn(f"规则评估错误 {rule.rule_id}: {str(e)}")
            return False
    
    def _create_alert(self, rule: AlertRule, measurement: ProcessedMeasurement, 
                     values: Dict[str, float]) -> Alert:
        """创建报警记录"""
        alert_id = f"alert_{int(time.time() * 1000)}_{rule.rule_id}"
        
        measured_value = values.get(rule.condition)
        
        # 生成报警消息
        message = f"{rule.description}"
        if measured_value is not None:
            message += f" (测量值: {measured_value:.3f}, 阈值: {rule.threshold_value:.3f})"
        
        # 详细信息
        details = {
            "material_name": measurement.material_name,
            "electrolyte_type": measurement.electrolyte_type,
            "measurement_timestamp": measurement.timestamp,
            "rule_condition": rule.condition,
            "measured_value": measured_value,
            "all_values": values
        }
        
        return Alert(
            alert_id=alert_id,
            timestamp=time.time(),
            alert_type=rule.alert_type,
            alert_level=rule.alert_level,
            rule_id=rule.rule_id,
            message=message,
            measurement_id=measurement.measurement_id,
            value=measured_value,
            threshold=rule.threshold_value,
            details=details
        )
    
    def _process_alert(self, alert: Alert):
        """处理报警"""
        # 添加到历史记录
        with self._lock:
            self.alert_history.append(alert)
        
        # 调用所有报警处理器
        for handler in self.alert_handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                warnings.warn(f"报警处理器执行失败: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认报警"""
        with self._lock:
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决报警"""
        with self._lock:
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.acknowledged = True
                    return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃的报警（未解决的）"""
        with self._lock:
            return [alert for alert in self.alert_history if not alert.resolved]
    
    def get_recent_alerts(self, hours: float = 24.0) -> List[Alert]:
        """获取最近的报警"""
        cutoff_time = time.time() - hours * 3600
        with self._lock:
            return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_alert_statistics(self, hours: float = 24.0) -> Dict[str, Any]:
        """获取报警统计信息"""
        recent_alerts = self.get_recent_alerts(hours)
        
        if not recent_alerts:
            return {"total_alerts": 0, "time_range_hours": hours}
        
        # 按类型统计
        type_counts = {}
        level_counts = {}
        
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            alert_level = alert.alert_level.value
            
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
            level_counts[alert_level] = level_counts.get(alert_level, 0) + 1
        
        # 解决率
        resolved_count = sum(1 for alert in recent_alerts if alert.resolved)
        resolution_rate = resolved_count / len(recent_alerts) if recent_alerts else 0
        
        return {
            "total_alerts": len(recent_alerts),
            "time_range_hours": hours,
            "alerts_by_type": type_counts,
            "alerts_by_level": level_counts,
            "resolved_count": resolved_count,
            "unresolved_count": len(recent_alerts) - resolved_count,
            "resolution_rate": resolution_rate,
            "most_frequent_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }
    
    def save_config(self, config_file: str):
        """保存配置到文件"""
        config = {
            "alert_rules": {
                rule_id: asdict(rule) for rule_id, rule in self.alert_rules.items()
            },
            "baseline_values": self.baseline_values.copy()
        }
        
        # 处理枚举类型
        for rule_data in config["alert_rules"].values():
            rule_data["alert_type"] = rule_data["alert_type"].value
            rule_data["alert_level"] = rule_data["alert_level"].value
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def load_config(self, config_file: str):
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 加载报警规则
            if "alert_rules" in config:
                for rule_id, rule_data in config["alert_rules"].items():
                    rule_data["alert_type"] = AlertType(rule_data["alert_type"])
                    rule_data["alert_level"] = AlertLevel(rule_data["alert_level"])
                    rule = AlertRule(**rule_data)
                    self.alert_rules[rule_id] = rule
            
            # 加载基准值
            if "baseline_values" in config:
                self.baseline_values.update(config["baseline_values"])
                
        except Exception as e:
            warnings.warn(f"配置加载失败: {str(e)}")
    
    def create_emergency_backup(self, measurement: ProcessedMeasurement, 
                               alert: Alert, backup_dir: str = "emergency_backups"):
        """创建紧急备份"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(alert.timestamp))
        backup_file = backup_path / f"emergency_backup_{timestamp_str}_{alert.alert_id}.json"
        
        backup_data = {
            "alert": asdict(alert),
            "measurement": asdict(measurement),
            "system_state": {
                "active_rules_count": len([r for r in self.alert_rules.values() if r.enabled]),
                "total_alerts_today": len(self.get_recent_alerts(24.0)),
                "baseline_values": self.baseline_values.copy()
            }
        }
        
        # 处理枚举类型
        backup_data["alert"]["alert_type"] = backup_data["alert"]["alert_type"].value
        backup_data["alert"]["alert_level"] = backup_data["alert"]["alert_level"].value
        
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"紧急备份创建失败: {str(e)}")
    
    def test_alert_system(self) -> Dict[str, Any]:
        """测试报警系统"""
        test_results = {
            "rules_count": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "handlers_count": len(self.alert_handlers),
            "baseline_values_count": len(self.baseline_values),
            "recent_alerts_count": len(self.get_recent_alerts(1.0)),
            "test_timestamp": time.time()
        }
        
        # 测试一个虚拟报警
        test_measurement = ProcessedMeasurement(
            measurement_id="test_001",
            timestamp=time.time(),
            electrolyte_type="test",
            material_name="test_material",
            spr_angle_deg=45.0,
            reflectance=0.99,  # 触发反射率异常
            phase_shift_rad=1.0,
            thickness_nm=1000.0,
            thickness_error_nm=10.0,
            thickness_confidence=0.3,  # 触发置信度过低警告
            uniformity_ratio=0.35,
            surface_roughness_nm=5.0,
            quality_score=0.8
        )
        
        test_alerts = self.check_measurement(test_measurement)
        test_results["test_alerts_generated"] = len(test_alerts)
        test_results["test_alert_types"] = [a.alert_type.value for a in test_alerts]
        
        return test_results 