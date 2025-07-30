"""数据处理模块"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import sqlite3
import time
from pathlib import Path
import threading
from queue import Queue
import pandas as pd
from datetime import datetime, timedelta
import warnings

from .electrolyte_materials import ElectrolyteType
from .spr_electrolyte_monitor import SPRMeasurement
from .thickness_analyzer import ThicknessResult
from .surface_quality_monitor import SurfaceQualityMetrics


@dataclass
class ProcessedMeasurement:
    """处理后的综合测量数据"""
    measurement_id: str
    timestamp: float
    electrolyte_type: str
    material_name: str
    
    # SPR数据
    spr_angle_deg: float
    reflectance: float
    phase_shift_rad: float
    
    # 厚度分析
    thickness_nm: float
    thickness_error_nm: float
    thickness_confidence: float
    
    # 表面质量
    uniformity_ratio: float
    surface_roughness_nm: float
    quality_score: float
    
    # 环境条件
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    pressure_mbar: Optional[float] = None


@dataclass
class DataStreamConfig:
    """数据流配置"""
    sampling_rate_hz: float = 1.0
    buffer_size: int = 1000
    auto_save_interval_s: float = 300.0  # 5分钟自动保存
    data_retention_days: int = 30
    compression_enabled: bool = True


class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(
        self,
        db_path: str = "electrolyte_monitoring.db",
        config: Optional[DataStreamConfig] = None
    ):
        """初始化实时数据处理器"""
        self.db_path = Path(db_path)
        self.config = config or DataStreamConfig()
        
        # 数据缓冲区
        self.data_buffer = Queue(maxsize=self.config.buffer_size)
        self.processed_data: List[ProcessedMeasurement] = []
        
        # 线程控制
        self.processing_thread = None
        self.is_running = False
        self._lock = threading.Lock()
        
        # 统计信息
        self.total_measurements = 0
        self.last_save_time = time.time()
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建主要测量数据表
            cursor.execute("""CREATE TABLE IF NOT EXISTS measurements (""")
            
            # 创建原始SPR数据表
            cursor.execute("""CREATE TABLE IF NOT EXISTS spr_raw_data (""")
            
            # 创建系统日志表
            cursor.execute("""CREATE TABLE IF NOT EXISTS system_logs (""")
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON measurements(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_material ON measurements(material_name)")
            
            conn.commit()
    
    def start_processing(self):
        """启动实时数据处理"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.log_event("INFO", "实时数据处理已启动")
    
    def stop_processing(self):
        """停止实时数据处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        # 保存剩余数据
        self._save_buffer_to_db()
        self.log_event("INFO", "实时数据处理已停止")
    
    def _processing_loop(self):
        """数据处理循环"""
        while self.is_running:
            try:
                # 检查是否需要自动保存
                current_time = time.time()
                if current_time - self.last_save_time > self.config.auto_save_interval_s:
                    self._save_buffer_to_db()
                    self.last_save_time = current_time
                
                # 等待新数据
                time.sleep(0.1)
                
            except Exception as e:
                self.log_event("ERROR", f"数据处理循环错误: {str(e)}")
                time.sleep(1.0)
    
    def add_measurement(
        self,
        spr_data: SPRMeasurement,
        thickness_result: ThicknessResult,
        quality_metrics: SurfaceQualityMetrics,
        electrolyte_type: ElectrolyteType,
        material_name: str,
        environmental_data: Optional[Dict[str, float]] = None
    ) -> str:
        """添加新的测量数据"""
        measurement_id = f"{int(time.time() * 1000)}_{self.total_measurements}"
        
        # 环境数据处理
        env_data = environmental_data or {}
        
        # 创建综合测量数据
        processed_measurement = ProcessedMeasurement(
            measurement_id=measurement_id,
            timestamp=spr_data.timestamp,
            electrolyte_type=electrolyte_type.value,
            material_name=material_name,
            spr_angle_deg=spr_data.incident_angle_deg,
            reflectance=spr_data.reflectance,
            phase_shift_rad=spr_data.phase_shift_rad,
            thickness_nm=thickness_result.thickness_nm,
            thickness_error_nm=thickness_result.thickness_error_nm,
            thickness_confidence=thickness_result.confidence,
            uniformity_ratio=quality_metrics.uniformity_ratio,
            surface_roughness_nm=quality_metrics.surface_roughness_nm,
            quality_score=quality_metrics.quality_score,
            temperature_c=env_data.get('temperature_c'),
            humidity_percent=env_data.get('humidity_percent'),
            pressure_mbar=env_data.get('pressure_mbar')
        )
        
        # 添加到缓冲区
        with self._lock:
            try:
                self.data_buffer.put_nowait(processed_measurement)
                self.processed_data.append(processed_measurement)
                self.total_measurements += 1
            except:
                # 缓冲区满，移除最旧的数据
                try:
                    self.data_buffer.get_nowait()
                    self.data_buffer.put_nowait(processed_measurement)
                except:
                    pass
        
        return measurement_id
    
    def _save_buffer_to_db(self):
        """将缓冲区数据保存到数据库"""
        with self._lock:
            if self.data_buffer.empty():
                return
            
            measurements_to_save = []
            while not self.data_buffer.empty():
                try:
                    measurement = self.data_buffer.get_nowait()
                    measurements_to_save.append(measurement)
                except:
                    break
        
        if not measurements_to_save:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for measurement in measurements_to_save:
                    data = asdict(measurement)
                    placeholders = ', '.join(['?' for _ in data])
                    columns = ', '.join(data.keys())
                    query = f"INSERT OR REPLACE INTO measurements ({columns}) VALUES ({placeholders})"
                    cursor.execute(query, list(data.values()))
                
                conn.commit()
                
            self.log_event("INFO", f"保存了 {len(measurements_to_save)} 条测量数据")
            
        except Exception as e:
            self.log_event("ERROR", f"数据保存错误: {str(e)}")
    
    def get_recent_data(
        self,
        hours: float = 1.0,
        limit: Optional[int] = None
    ) -> List[ProcessedMeasurement]:
        """获取最近的测量数据"""
        cutoff_time = time.time() - hours * 3600
        
        with self._lock:
            recent_data = [
                m for m in self.processed_data 
                if m.timestamp >= cutoff_time
            ]
        
        # 按时间倒序排列
        recent_data.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            recent_data = recent_data[:limit]
        
        return recent_data
    
    def get_data_by_material(
        self,
        material_name: str,
        hours: Optional[float] = None
    ) -> List[ProcessedMeasurement]:
        """根据材料名称获取数据"""
        with self._lock:
            material_data = [
                m for m in self.processed_data 
                if m.material_name == material_name
            ]
        
        if hours:
            cutoff_time = time.time() - hours * 3600
            material_data = [
                m for m in material_data 
                if m.timestamp >= cutoff_time
            ]
        
        return material_data
    
    def calculate_statistics(
        self,
        data: Optional[List[ProcessedMeasurement]] = None,
        hours: float = 24.0
    ) -> Dict[str, Any]:
        """计算统计信息"""
        if data is None:
            data = self.get_recent_data(hours)
        
        if not data:
            return {"error": "无数据可统计"}
        
        # 提取数值数据
        thicknesses = [m.thickness_nm for m in data]
        reflectances = [m.reflectance for m in data]
        uniformities = [m.uniformity_ratio for m in data]
        quality_scores = [m.quality_score for m in data]
        
        def calc_stats(values):
            """计算基本统计量"""
            if not values:
                return {}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }
        
        return {
            "data_count": len(data),
            "time_span_hours": (data[0].timestamp - data[-1].timestamp) / 3600 if len(data) > 1 else 0,
            "thickness_stats": calc_stats(thicknesses),
            "reflectance_stats": calc_stats(reflectances),
            "uniformity_stats": calc_stats(uniformities),
            "quality_stats": calc_stats(quality_scores),
            "material_distribution": self._get_material_distribution(data),
            "measurement_rate_per_hour": len(data) / max(hours, 1.0)
        }
    
    def _get_material_distribution(self, data: List[ProcessedMeasurement]) -> Dict[str, int]:
        """获取材料分布统计"""
        distribution = {}
        for measurement in data:
            material = measurement.material_name
            distribution[material] = distribution.get(material, 0) + 1
        return distribution
    
    def export_data(
        self,
        filepath: str,
        format: str = "json",
        hours: Optional[float] = None,
        material_filter: Optional[str] = None
    ):
        """导出数据到文件"""
        # 获取数据
        if hours:
            data = self.get_recent_data(hours)
        else:
            with self._lock:
                data = self.processed_data.copy()
        
        # 材料过滤
        if material_filter:
            data = [m for m in data if m.material_name == material_filter]
        
        if not data:
            raise ValueError("没有符合条件的数据可导出")
        
        filepath = Path(filepath)
        
        if format.lower() == "json":
            self._export_json(data, filepath)
        elif format.lower() == "csv":
            self._export_csv(data, filepath)
        elif format.lower() == "excel":
            self._export_excel(data, filepath)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_json(self, data: List[ProcessedMeasurement], filepath: Path):
        """导出为JSON格式"""
        export_data = {
            "export_info": {
                "timestamp": time.time(),
                "data_count": len(data),
                "format_version": "1.0"
            },
            "measurements": [asdict(m) for m in data],
            "statistics": self.calculate_statistics(data)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, data: List[ProcessedMeasurement], filepath: Path):
        """导出为CSV格式"""
        df = pd.DataFrame([asdict(m) for m in data])
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _export_excel(self, data: List[ProcessedMeasurement], filepath: Path):
        """导出为Excel格式"""
        df = pd.DataFrame([asdict(m) for m in data])
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Measurements', index=False)
            
            # 添加统计信息表
            stats = self.calculate_statistics(data)
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    def log_event(self, level: str, message: str, details: str = ""):
        """记录系统事件"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO system_logs (timestamp, level, message, details) VALUES (?, ?, ?, ?)",
                    (time.time(), level, message, details)
                )
                conn.commit()
        except Exception as e:
            warnings.warn(f"无法记录日志: {str(e)}")
    
    def get_system_logs(self, hours: float = 24.0, level: Optional[str] = None) -> List[Dict]:
        """获取系统日志"""
        cutoff_time = time.time() - hours * 3600
        
        query = "SELECT timestamp, level, message, details FROM system_logs WHERE timestamp >= ?"
        params = [cutoff_time]
        
        if level:
            query += " AND level = ?"
            params.append(level)
        
        query += " ORDER BY timestamp DESC"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        "timestamp": row[0],
                        "level": row[1],
                        "message": row[2],
                        "details": row[3],
                        "datetime": datetime.fromtimestamp(row[0]).isoformat()
                    }
                    for row in rows
                ]
        except Exception as e:
            warnings.warn(f"无法获取系统日志: {str(e)}")
            return []
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        cutoff_time = time.time() - days * 24 * 3600
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除旧的测量数据
                cursor.execute("DELETE FROM measurements WHERE timestamp < ?", (cutoff_time,))
                cursor.execute("DELETE FROM spr_raw_data WHERE timestamp < ?", (cutoff_time,))
                cursor.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                # 优化数据库
                cursor.execute("VACUUM")
                
                self.log_event("INFO", f"清理了 {deleted_count} 条旧数据")
                
        except Exception as e:
            self.log_event("ERROR", f"数据清理错误: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            buffer_usage = self.data_buffer.qsize() / self.config.buffer_size
            memory_usage = len(self.processed_data)
        
        recent_data = self.get_recent_data(1.0)  # 最近1小时
        processing_rate = len(recent_data) if recent_data else 0
        
        return {
            "total_measurements": self.total_measurements,
            "buffer_usage_percent": buffer_usage * 100,
            "memory_measurements": memory_usage,
            "processing_rate_per_hour": processing_rate,
            "is_running": self.is_running,
            "database_size_mb": self.db_path.stat().st_size / (1024*1024) if self.db_path.exists() else 0
        } 