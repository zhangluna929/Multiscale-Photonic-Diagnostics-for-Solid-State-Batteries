"""固态锂电池电解质层厚度与表面质量监测模块"""

from .electrolyte_materials import ElectrolyteMaterials, ElectrolyteType
from .spr_electrolyte_monitor import SPRElectrolyteMonitor
from .thickness_analyzer import ThicknessAnalyzer
from .surface_quality_monitor import SurfaceQualityMonitor
from .data_processor import DataProcessor
from .alert_system import AlertSystem

# 新增锂枝晶监测模块
from .lithium_dendrite_monitor import (
    LithiumDendriteMonitor, 
    DendriteMonitoringConfig, 
    DendriteRiskLevel, 
    AlertType,
    DendriteAlert,
    SpotMorphologyData
)

# 新增数据收集器
from .data_collector import (
    DataCollector,
    DataCollectionConfig,
    MonitoringDataPoint
)

__all__ = [
    # 原有模块
    "ElectrolyteMaterials",
    "ElectrolyteType", 
    "SPRElectrolyteMonitor",
    "ThicknessAnalyzer",
    "SurfaceQualityMonitor",
    "DataProcessor",
    "AlertSystem",
    
    # 锂枝晶监测模块
    "LithiumDendriteMonitor",
    "DendriteMonitoringConfig",
    "DendriteRiskLevel",
    "AlertType",
    "DendriteAlert",
    "SpotMorphologyData",
    
    # 数据收集模块
    "DataCollector",
    "DataCollectionConfig",
    "MonitoringDataPoint",
]

__version__ = "1.1.0"
__author__ = "固态电池监测团队"

# 模块功能描述
__features__ = {
    "electrolyte_monitoring": "电解质层厚度监测（±50nm精度）",
    "surface_quality": "表面质量监控（光斑均匀度提升40%）",
    "dendrite_detection": "锂枝晶早期预警（反射率±0.05%、相位±0.1°）",
    "realtime_data": "实时数据收集（10Hz+采样率）",
    "data_export": "多格式数据导出（JSON/CSV/SQLite）",
    "anomaly_detection": "智能异常检测和预警",
    "api_integration": "REST API数据上传集成"
} 