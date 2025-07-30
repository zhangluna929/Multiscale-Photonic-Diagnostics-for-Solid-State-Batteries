"""数据收集器模块"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
import time
import json
import csv
import threading
from queue import Queue, Empty
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import requests
from datetime import datetime
import sqlite3
from contextlib import contextmanager

try:
    from .lithium_dendrite_monitor import DendriteAlert, DendriteRiskLevel, AlertType
except ImportError:
    from lithium_dendrite_monitor import DendriteAlert, DendriteRiskLevel, AlertType


@dataclass
class MonitoringDataPoint:
    """监测数据点"""
    timestamp: float
    device_id: str
    measurement_type: str  # "electrolyte", "sei", "dendrite"
    values: Dict[str, float]
    alerts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class DataCollectionConfig:
    """数据收集配置"""
    sampling_rate_hz: float = 15.0  # 15Hz采样率
    buffer_size: int = 1000  # 数据缓冲区大小
    auto_save_interval_seconds: float = 30.0  # 自动保存间隔
    export_formats: List[str] = None  # 导出格式 ["json", "csv", "sqlite"]
    
    # REST API配置
    api_enabled: bool = False
    api_endpoint: str = ""
    api_headers: Dict[str, str] = None
    api_timeout_seconds: float = 5.0
    
    # 异常检测配置
    anomaly_detection_enabled: bool = True
    anomaly_threshold_std: float = 3.0  # 异常阈值（标准差倍数）
    
    # 历史数据管理
    max_memory_points: int = 10000  # 内存中最大数据点数
    database_file: str = "battery_monitoring.db"
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["json", "csv", "sqlite"]
        if self.api_headers is None:
            self.api_headers = {"Content-Type": "application/json"}


class DataCollector:
    """数据收集器"""
    
    def __init__(self, config: Optional[DataCollectionConfig] = None):
        """初始化数据收集器"""
        self.config = config or DataCollectionConfig()
        
        # 数据缓冲区
        self.data_buffer: Queue = Queue(maxsize=self.config.buffer_size)
        self.processed_data: List[MonitoringDataPoint] = []
        
        # 线程控制
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        # 异常检测
        self.baseline_statistics: Dict[str, Dict[str, float]] = {}
        self.anomaly_history: List[Dict[str, Any]] = []
        
        # 性能监控
        self.collection_stats = {
            "total_points": 0,
            "anomalies_detected": 0,
            "api_uploads": 0,
            "api_failures": 0,
            "last_collection_time": 0
        }
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据库
        if "sqlite" in self.config.export_formats:
            self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS monitoring_data (""")
                
                conn.execute("""CREATE TABLE IF NOT EXISTS anomalies (""")
                
                conn.execute("""CREATE INDEX IF NOT EXISTS idx_timestamp ON monitoring_data(timestamp);""")
                
                conn.execute("""CREATE INDEX IF NOT EXISTS idx_device_type ON monitoring_data(device_id, measurement_type);""")
            
            self.logger.info("数据库初始化完成")
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接上下文管理器"""
        conn = sqlite3.connect(self.config.database_file)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def add_data_point(
        self, 
        device_id: str,
        measurement_type: str,
        values: Dict[str, float],
        alerts: List[DendriteAlert] = None,
        metadata: Dict[str, Any] = None
    ):
        """添加数据点到收集器"""
        alerts = alerts or []
        metadata = metadata or {}
        
        # 转换预警对象为字典
        alert_dicts = []
        for alert in alerts:
            if isinstance(alert, DendriteAlert):
                alert_dicts.append({
                    "timestamp": alert.timestamp,
                    "type": alert.alert_type.value,
                    "risk_level": alert.risk_level.value,
                    "message": alert.message,
                    "measured_value": alert.measured_value,
                    "threshold_value": alert.threshold_value,
                    "confidence": alert.confidence,
                    "recommended_action": alert.recommended_action
                })
            else:
                alert_dicts.append(alert)
        
        data_point = MonitoringDataPoint(
            timestamp=time.time(),
            device_id=device_id,
            measurement_type=measurement_type,
            values=values,
            alerts=alert_dicts,
            metadata=metadata
        )
        
        try:
            self.data_buffer.put_nowait(data_point)
            self.collection_stats["total_points"] += 1
            self.collection_stats["last_collection_time"] = time.time()
        except:
            self.logger.warning("数据缓冲区已满，丢弃数据点")
    
    def start_collection(self):
        """开始数据收集"""
        if self.collection_active:
            self.logger.warning("数据收集已在运行中")
            return
        
        self.collection_active = True
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info(f"数据收集已启动，采样率: {self.config.sampling_rate_hz} Hz")
    
    def stop_collection(self):
        """停止数据收集"""
        self.collection_active = False
        
        # 等待线程结束
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # 处理剩余数据
        self._flush_buffer()
        
        self.logger.info("数据收集已停止")
    
    def _processing_loop(self):
        """数据处理循环"""
        last_save_time = time.time()
        process_interval = 1.0 / self.config.sampling_rate_hz
        
        while self.collection_active:
            start_time = time.time()
            
            try:
                # 从缓冲区获取数据
                data_point = self.data_buffer.get(timeout=process_interval)
                
                # 处理数据点
                self._process_data_point(data_point)
                
                # 自动保存
                if (time.time() - last_save_time) >= self.config.auto_save_interval_seconds:
                    self._auto_save()
                    last_save_time = time.time()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"数据处理错误: {e}")
            
            # 控制处理频率
            elapsed = time.time() - start_time
            sleep_time = max(0, process_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_data_point(self, data_point: MonitoringDataPoint):
        """处理单个数据点"""
        # 添加到内存缓存
        self.processed_data.append(data_point)
        
        # 管理内存使用
        if len(self.processed_data) > self.config.max_memory_points:
            # 移除最老的数据点
            self.processed_data = self.processed_data[-self.config.max_memory_points:]
        
        # 异常检测
        if self.config.anomaly_detection_enabled:
            anomalies = self._detect_anomalies(data_point)
            if anomalies:
                self._handle_anomalies(anomalies)
        
        # API上传
        if self.config.api_enabled:
            self._upload_to_api(data_point)
        
        # 数据库存储
        if "sqlite" in self.config.export_formats:
            self._save_to_database(data_point)
    
    def _detect_anomalies(self, data_point: MonitoringDataPoint) -> List[Dict[str, Any]]:
        """检测数据异常"""
        anomalies = []
        measurement_type = data_point.measurement_type
        
        # 更新基线统计
        if measurement_type not in self.baseline_statistics:
            self.baseline_statistics[measurement_type] = {}
        
        for param_name, value in data_point.values.items():
            key = f"{measurement_type}_{param_name}"
            
            if key not in self.baseline_statistics[measurement_type]:
                # 初始化统计
                self.baseline_statistics[measurement_type][key] = {
                    "values": [value],
                    "mean": value,
                    "std": 0.0,
                    "count": 1
                }
            else:
                stats = self.baseline_statistics[measurement_type][key]
                stats["values"].append(value)
                
                # 保持最近1000个值
                if len(stats["values"]) > 1000:
                    stats["values"] = stats["values"][-1000:]
                
                # 更新统计
                stats["mean"] = np.mean(stats["values"])
                stats["std"] = np.std(stats["values"])
                stats["count"] = len(stats["values"])
                
                # 异常检测
                if stats["count"] > 10 and stats["std"] > 0:
                    deviation = abs(value - stats["mean"]) / stats["std"]
                    
                    if deviation > self.config.anomaly_threshold_std:
                        # 确定异常严重程度
                        if deviation > 5.0:
                            severity = "critical"
                        elif deviation > 4.0:
                            severity = "high"
                        else:
                            severity = "medium"
                        
                        anomaly = {
                            "timestamp": data_point.timestamp,
                            "measurement_type": measurement_type,
                            "parameter_name": param_name,
                            "measured_value": value,
                            "expected_value": stats["mean"],
                            "deviation_std": deviation,
                            "severity": severity
                        }
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _handle_anomalies(self, anomalies: List[Dict[str, Any]]):
        """处理检测到的异常"""
        for anomaly in anomalies:
            self.anomaly_history.append(anomaly)
            self.collection_stats["anomalies_detected"] += 1
            
            # 记录异常到数据库
            if "sqlite" in self.config.export_formats:
                try:
                    with self._get_db_connection() as conn:
                        conn.execute("""INSERT INTO anomalies""", (
                            anomaly["timestamp"],
                            anomaly["measurement_type"],
                            anomaly["parameter_name"],
                            anomaly["measured_value"],
                            anomaly["expected_value"],
                            anomaly["deviation_std"],
                            anomaly["severity"]
                        ))
                except Exception as e:
                    self.logger.error(f"保存异常记录失败: {e}")
            
            # 记录日志
            self.logger.warning(
                f"检测到异常: {anomaly['measurement_type']}.{anomaly['parameter_name']} "
                f"= {anomaly['measured_value']:.3f} "
                f"(期望: {anomaly['expected_value']:.3f}, "
                f"偏差: {anomaly['deviation_std']:.1f}σ, "
                f"严重程度: {anomaly['severity']})"
            )
    
    def _upload_to_api(self, data_point: MonitoringDataPoint):
        """上传数据到API"""
        try:
            data = asdict(data_point)
            response = requests.post(
                self.config.api_endpoint,
                json=data,
                headers=self.config.api_headers,
                timeout=self.config.api_timeout_seconds
            )
            
            if response.status_code == 200:
                self.collection_stats["api_uploads"] += 1
            else:
                self.collection_stats["api_failures"] += 1
                self.logger.warning(f"API上传失败: {response.status_code}")
                
        except requests.RequestException as e:
            self.collection_stats["api_failures"] += 1
            self.logger.error(f"API上传错误: {e}")
    
    def _save_to_database(self, data_point: MonitoringDataPoint):
        """保存数据到数据库"""
        try:
            with self._get_db_connection() as conn:
                conn.execute("""INSERT INTO monitoring_data""", (
                    data_point.timestamp,
                    data_point.device_id,
                    data_point.measurement_type,
                    json.dumps(data_point.values),
                    json.dumps(data_point.alerts),
                    json.dumps(data_point.metadata)
                ))
        except Exception as e:
            self.logger.error(f"数据库保存失败: {e}")
    
    def _auto_save(self):
        """自动保存数据到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if "json" in self.config.export_formats:
                self.export_json(f"monitoring_data_{timestamp}.json")
            
            if "csv" in self.config.export_formats:
                self.export_csv(f"monitoring_data_{timestamp}.csv")
            
            self.logger.info("自动保存完成")
        except Exception as e:
            self.logger.error(f"自动保存失败: {e}")
    
    def _flush_buffer(self):
        """清空缓冲区中的剩余数据"""
        while not self.data_buffer.empty():
            try:
                data_point = self.data_buffer.get_nowait()
                self._process_data_point(data_point)
            except Empty:
                break
    
    def export_json(self, filename: str):
        """导出JSON格式数据"""
        data = {
            "export_timestamp": time.time(),
            "config": asdict(self.config),
            "statistics": self.collection_stats,
            "data_points": [asdict(dp) for dp in self.processed_data],
            "anomalies": self.anomaly_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"数据已导出到: {filename}")
    
    def export_csv(self, filename: str):
        """导出CSV格式数据"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                "timestamp", "device_id", "measurement_type", "parameter", 
                "value", "alert_count", "anomaly_count"
            ])
            
            # 写入数据
            for dp in self.processed_data:
                for param, value in dp.values.items():
                    writer.writerow([
                        dp.timestamp,
                        dp.device_id,
                        dp.measurement_type,
                        param,
                        value,
                        len(dp.alerts),
                        len([a for a in self.anomaly_history 
                             if a["timestamp"] == dp.timestamp])
                    ])
        
        self.logger.info(f"CSV数据已导出到: {filename}")
    
    def get_recent_data(self, limit: int = 100) -> List[MonitoringDataPoint]:
        """获取最近的数据点"""
        return self.processed_data[-limit:] if self.processed_data else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取收集统计信息"""
        stats = self.collection_stats.copy()
        stats.update({
            "buffer_size": self.data_buffer.qsize(),
            "processed_points": len(self.processed_data),
            "anomaly_count": len(self.anomaly_history),
            "collection_active": self.collection_active,
            "baseline_parameters": len(self.baseline_statistics),
            "uptime_hours": (time.time() - stats["last_collection_time"]) / 3600 
                           if stats["last_collection_time"] > 0 else 0
        })
        return stats
    
    def query_database(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        measurement_type: Optional[str] = None,
        device_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """查询数据库中的历史数据"""
        if "sqlite" not in self.config.export_formats:
            raise ValueError("SQLite未启用")
        
        query = "SELECT * FROM monitoring_data WHERE 1=1"
        params = []
        
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if measurement_type is not None:
            query += " AND measurement_type = ?"
            params.append(measurement_type)
        
        if device_id is not None:
            query += " AND device_id = ?"
            params.append(device_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def clear_data(self, keep_recent: int = 100):
        """清除数据，保留最近的数据点"""
        if keep_recent > 0:
            self.processed_data = self.processed_data[-keep_recent:]
        else:
            self.processed_data.clear()
        
        self.anomaly_history.clear()
        
        # 清空缓冲区
        while not self.data_buffer.empty():
            try:
                self.data_buffer.get_nowait()
            except Empty:
                break
        
        self.logger.info(f"数据已清除，保留最近 {keep_recent} 个数据点") 