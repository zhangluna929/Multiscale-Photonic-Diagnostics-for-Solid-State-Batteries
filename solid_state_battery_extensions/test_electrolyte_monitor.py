"""电解质监测系统综合测试模块"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
import unittest

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from solid_state_battery_extensions.electrolyte_materials import (
    ElectrolyteMaterials, ElectrolyteType, electrolyte_db
)
from solid_state_battery_extensions.spr_electrolyte_monitor import (
    SPRElectrolyteMonitor, BeamParameters
)
from solid_state_battery_extensions.thickness_analyzer import (
    ThicknessAnalyzer, ElectrolyteFilm
)
from solid_state_battery_extensions.surface_quality_monitor import (
    SurfaceQualityMonitor
)
from solid_state_battery_extensions.data_processor import (
    RealTimeDataProcessor, DataStreamConfig
)
from solid_state_battery_extensions.alert_system import (
    AlertSystem, AlertLevel, AlertType, FileAlertHandler
)


class TestElectrolyteMaterials(unittest.TestCase):
    """测试电解质材料数据库"""
    
    def setUp(self):
        self.materials_db = ElectrolyteMaterials()
    
    def test_sulfide_materials(self):
        """测试硫化物电解质材料"""
        # 测试LPSCl
        material = self.materials_db.get_material(ElectrolyteType.SULFIDE, "LPSCl")
        self.assertEqual(material.name, "Li6PS5Cl")
        self.assertTrue(2.05 <= material.refractive_index_range[0] <= 2.10)
        
        # 测试折射率获取
        n_632 = self.materials_db.get_refractive_index(
            ElectrolyteType.SULFIDE, "LPSCl", 632.8
        )
        self.assertTrue(2.05 <= n_632 <= 2.10)
    
    def test_polymer_materials(self):
        """测试聚合物电解质材料"""
        material = self.materials_db.get_material(ElectrolyteType.POLYMER, "PEO")
        self.assertEqual(material.name, "Polyethylene Oxide")
        self.assertTrue(1.42 <= material.refractive_index_range[0] <= 1.45)
    
    def test_thickness_tolerance(self):
        """测试厚度容差计算"""
        # 薄层电解质应该有更严格的容差
        tolerance_lgps = self.materials_db.get_thickness_tolerance(
            ElectrolyteType.SULFIDE, "LGPS"
        )
        self.assertEqual(tolerance_lgps, 50.0)  # <1μm，应该是50nm
        
        # 厚层电解质容差可以更宽松
        tolerance_peo = self.materials_db.get_thickness_tolerance(
            ElectrolyteType.POLYMER, "PEO"
        )
        self.assertTrue(tolerance_peo >= 100.0)  # >10μm，容差应该更宽松


class TestSPRElectrolyteMonitor(unittest.TestCase):
    """测试SPR电解质监测器"""
    
    def setUp(self):
        self.monitor = SPRElectrolyteMonitor(
            electrolyte_type=ElectrolyteType.SULFIDE,
            electrolyte_material="LPSCl",
            metal_film_thickness_nm=50.0
        )
    
    def test_spr_curve_measurement(self):
        """测试SPR曲线测量"""
        angles, reflectance, phases = self.monitor.measure_spr_curve()
        
        # 检查数据完整性
        self.assertEqual(len(angles), len(reflectance))
        self.assertEqual(len(angles), len(phases))
        self.assertTrue(len(angles) > 100)  # 应该有足够的数据点
        
        # 检查反射率范围
        self.assertTrue(0.0 <= np.min(reflectance) <= 1.0)
        self.assertTrue(0.0 <= np.max(reflectance) <= 1.0)
        
        # 应该能找到SPR共振点（反射率极小值）
        min_reflectance = np.min(reflectance)
        self.assertTrue(min_reflectance < 0.5)  # SPR应该显著降低反射率
    
    def test_spr_angle_finding(self):
        """测试SPR角度查找"""
        spr_angle, min_reflectance = self.monitor.find_spr_angle()
        
        # 检查角度范围合理
        self.assertTrue(40.0 <= spr_angle <= 75.0)
        self.assertTrue(0.0 <= min_reflectance <= 0.5)
    
    def test_angle_optimization(self):
        """测试角度优化"""
        optimization_result = self.monitor.optimize_spr_angle()
        
        # 检查返回的键
        required_keys = [
            "optimal_angle_deg", "max_sensitivity", 
            "spr_angle_deg", "min_reflectance"
        ]
        for key in required_keys:
            self.assertIn(key, optimization_result)
        
        # 检查值的合理性
        self.assertTrue(40.0 <= optimization_result["optimal_angle_deg"] <= 75.0)
        self.assertTrue(optimization_result["max_sensitivity"] > 0)


class TestThicknessAnalyzer(unittest.TestCase):
    """测试厚度分析器"""
    
    def setUp(self):
        self.analyzer = ThicknessAnalyzer(
            wavelength_nm=632.8,
            metal_thickness_nm=50.0
        )
    
    def test_electrolyte_film(self):
        """测试电解质薄膜传输矩阵计算"""
        film = ElectrolyteFilm(
            n_electrolyte=2.075,  # LPSCl的折射率
            electrolyte_thickness_nm=1000.0
        )
        
        # 测试反射率计算
        reflectance = film.get_reflectance(632.8, 60.0)
        self.assertTrue(0.0 <= reflectance <= 1.0)
        
        # 测试相位计算
        phase = film.get_phase(632.8, 60.0)
        self.assertTrue(-np.pi <= phase <= np.pi)
    
    def test_thickness_fitting_accuracy(self):
        """测试厚度拟合精度"""
        # 创建模拟数据
        true_thickness = 800.0  # nm
        true_n = 2.075
        
        film = ElectrolyteFilm(
            n_electrolyte=true_n,
            electrolyte_thickness_nm=true_thickness
        )
        
        # 生成模拟测量数据
        angles = np.linspace(50, 70, 50)
        reflectance_data = [film.get_reflectance(632.8, angle) for angle in angles]
        
        # 添加噪声
        noise_level = 0.005  # 0.5% noise
        reflectance_data = [
            r + np.random.normal(0, noise_level) for r in reflectance_data
        ]
        
        # 拟合厚度
        result = self.analyzer.fit_thickness_multi_angle(
            list(angles), reflectance_data, true_n, (100, 2000)
        )
        
        # 检查精度
        thickness_error = abs(result.thickness_nm - true_thickness)
        self.assertTrue(thickness_error <= 50.0, f"厚度误差 {thickness_error:.1f}nm 超过目标精度50nm")
        
        # 检查置信度
        self.assertTrue(result.confidence > 0.7, f"置信度 {result.confidence:.3f} 过低")
    
    def test_joint_fitting(self):
        """测试厚度和折射率联合拟合"""
        # 模拟数据
        angles = np.linspace(45, 75, 30)
        # 使用已知参数生成"真实"数据
        film = ElectrolyteFilm(n_electrolyte=1.43, electrolyte_thickness_nm=5000.0)
        reflectance_data = [film.get_reflectance(632.8, angle) for angle in angles]
        
        # 联合拟合
        result = self.analyzer.fit_thickness_and_refractive_index(
            list(angles), reflectance_data, (1000, 10000), (1.4, 1.5)
        )
        
        # 检查结果合理性
        self.assertTrue(1000 <= result.thickness_nm <= 10000)
        self.assertTrue(1.4 <= result.refractive_index <= 1.5)


class TestSurfaceQualityMonitor(unittest.TestCase):
    """测试表面质量监测器"""
    
    def setUp(self):
        self.monitor = SurfaceQualityMonitor(target_uniformity=0.21)
    
    def test_uniformity_calculation(self):
        """测试均匀度计算"""
        # 创建模拟光斑数据
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # 高斯分布
        sigma = 1.0
        gaussian_spot = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        uniformity = self.monitor.calculate_uniformity_ratio(gaussian_spot)
        self.assertTrue(uniformity > 0)
        
        # 均匀光斑应该有更好的均匀度
        uniform_spot = np.ones((50, 50))
        uniform_spot[20:30, 20:30] = 1.0  # 中心区域
        uniformity_uniform = self.monitor.calculate_uniformity_ratio(uniform_spot)
        
        # 均匀光斑的均匀度比值应该更小（更好）
        self.assertTrue(uniformity_uniform < uniformity)
    
    def test_surface_roughness_estimation(self):
        """测试表面粗糙度估算"""
        # 创建平滑表面
        smooth_surface = np.ones((50, 50)) * 0.8
        roughness_smooth = self.monitor.estimate_surface_roughness(smooth_surface)
        
        # 创建粗糙表面
        rough_surface = smooth_surface + 0.1 * np.random.random((50, 50))
        roughness_rough = self.monitor.estimate_surface_roughness(rough_surface)
        
        # 粗糙表面应该有更高的粗糙度值
        self.assertTrue(roughness_rough > roughness_smooth)
    
    def test_quality_improvement_tracking(self):
        """测试质量改善跟踪"""
        # 模拟质量改善过程
        initial_uniformity = 0.35
        target_uniformity = 0.21
        
        # 生成一系列改善的数据
        for i in range(10):
            # 逐渐改善的均匀度
            current_uniformity = initial_uniformity - (initial_uniformity - target_uniformity) * i / 9
            
            # 创建对应的强度数据
            spot_data = self._create_spot_with_uniformity(current_uniformity)
            self.monitor.comprehensive_quality_assessment(spot_data)
        
        # 分析改善情况
        improvement_analysis = self.monitor.quality_improvement_analysis()
        
        self.assertTrue(improvement_analysis["improvement_percent"] > 0)
        self.assertTrue(improvement_analysis["progress_percent"] > 80)  # 应该接近目标
    
    def _create_spot_with_uniformity(self, target_uniformity: float) -> np.ndarray:
        """创建具有指定均匀度的光斑数据"""
        # 简化的光斑生成，实际应用中会更复杂
        size = 50
        center = size // 2
        
        # 基础高斯分布
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # 调整sigma以达到目标均匀度
        sigma = target_uniformity * 3  # 经验关系
        spot = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        return spot


class TestIntegratedSystem(unittest.TestCase):
    """测试集成系统"""
    
    def setUp(self):
        # 初始化所有组件
        self.spr_monitor = SPRElectrolyteMonitor(
            electrolyte_type=ElectrolyteType.SULFIDE,
            electrolyte_material="LPSCl"
        )
        
        self.thickness_analyzer = ThicknessAnalyzer()
        self.quality_monitor = SurfaceQualityMonitor()
        
        # 数据处理器
        self.data_processor = RealTimeDataProcessor(
            db_path="test_monitoring.db",
            config=DataStreamConfig(sampling_rate_hz=0.5, buffer_size=50)
        )
        
        # 报警系统
        self.alert_system = AlertSystem()
        test_log_file = "test_alerts.log"
        self.alert_system.add_handler(FileAlertHandler(test_log_file))
    
    def test_complete_measurement_workflow(self):
        """测试完整的测量工作流程"""
        # 1. SPR测量
        spr_measurement = self.spr_monitor.measure_at_angle(65.0)
        
        # 2. 厚度分析
        angles = [60, 62, 64, 66, 68]
        reflectance_data = []
        for angle in angles:
            measurement = self.spr_monitor.measure_at_angle(angle)
            reflectance_data.append(measurement.reflectance)
        
        thickness_result = self.thickness_analyzer.analyze_electrolyte_thickness(
            electrolyte_type=ElectrolyteType.SULFIDE,
            material_name="LPSCl",
            measurement_data={"angles": angles, "reflectance": reflectance_data},
            analysis_method="multi_angle"
        )
        
        # 3. 表面质量评估
        intensity_data = self._create_test_intensity_data()
        quality_metrics = self.quality_monitor.comprehensive_quality_assessment(intensity_data)
        
        # 4. 数据处理和存储
        measurement_id = self.data_processor.add_measurement(
            spr_measurement, thickness_result, quality_metrics,
            ElectrolyteType.SULFIDE, "LPSCl"
        )
        
        # 5. 报警检查
        recent_data = self.data_processor.get_recent_data(1.0)
        if recent_data:
            alerts = self.alert_system.check_measurement(recent_data[0])
        
        # 验证结果
        self.assertIsNotNone(measurement_id)
        self.assertTrue(thickness_result.thickness_nm > 0)
        self.assertTrue(0 <= quality_metrics.quality_score <= 1)
        
        print(f" 完整工作流程测试通过")
        print(f"  - SPR角度: {spr_measurement.incident_angle_deg:.1f}°")
        print(f"  - 反射率: {spr_measurement.reflectance:.4f}")
        print(f"  - 厚度: {thickness_result.thickness_nm:.1f} ± {thickness_result.thickness_error_nm:.1f} nm")
        print(f"  - 质量评分: {quality_metrics.quality_score:.3f}")
        print(f"  - 均匀度: {quality_metrics.uniformity_ratio:.3f}")
    
    def test_precision_validation(self):
        """测试精度验证"""
        # 测试厚度精度目标 ±50nm
        test_thicknesses = [500, 800, 1200, 1500]  # nm
        
        for true_thickness in test_thicknesses:
            # 创建模拟数据
            film = ElectrolyteFilm(
                n_electrolyte=2.075,
                electrolyte_thickness_nm=true_thickness
            )
            
            angles = np.linspace(50, 70, 20)
            reflectance_data = [film.get_reflectance(632.8, angle) for angle in angles]
            
            # 添加现实的噪声
            noise_std = 0.002  # 0.2% 标准差
            reflectance_data = [
                r + np.random.normal(0, noise_std) for r in reflectance_data
            ]
            
            # 分析厚度
            result = self.thickness_analyzer.fit_thickness_multi_angle(
                list(angles), reflectance_data, 2.075, (100, 2000)
            )
            
            # 验证精度
            error = abs(result.thickness_nm - true_thickness)
            precision_target = 50.0  # nm
            
            self.assertTrue(
                error <= precision_target,
                f"厚度 {true_thickness}nm: 误差 {error:.1f}nm 超过目标精度 {precision_target}nm"
            )
            
            print(f" 厚度 {true_thickness}nm: 测量值 {result.thickness_nm:.1f}nm, 误差 {error:.1f}nm")
    
    def test_surface_quality_improvement(self):
        """测试表面质量改善目标"""
        # 目标：从0.35提升到0.21（40%改善）
        initial_uniformity = 0.35
        target_uniformity = 0.21
        improvement_target = 40.0  # %
        
        # 模拟改善过程
        uniformity_values = []
        for i in range(10):
            # 线性改善
            current = initial_uniformity - (initial_uniformity - target_uniformity) * i / 9
            uniformity_values.append(current)
            
            # 创建对应的测试数据
            spot_data = self._create_spot_with_uniformity(current)
            self.quality_monitor.comprehensive_quality_assessment(spot_data)
        
        # 分析改善
        improvement_analysis = self.quality_monitor.quality_improvement_analysis()
        
        actual_improvement = improvement_analysis["improvement_percent"]
        self.assertTrue(
            actual_improvement >= improvement_target * 0.9,  # 允许10%误差
            f"表面质量改善 {actual_improvement:.1f}% 未达到目标 {improvement_target}%"
        )
        
        print(f" 表面质量改善: {actual_improvement:.1f}% (目标: {improvement_target}%)")
    
    def test_alert_system_responsiveness(self):
        """测试报警系统响应性"""
        # 创建会触发报警的测量数据
        
        # 1. 厚度偏差报警测试
        from solid_state_battery_extensions.data_processor import ProcessedMeasurement
        
        normal_measurement = ProcessedMeasurement(
            measurement_id="test_normal",
            timestamp=time.time(),
            electrolyte_type="sulfide",
            material_name="LPSCl",
            spr_angle_deg=65.0,
            reflectance=0.15,
            phase_shift_rad=1.5,
            thickness_nm=1000.0,
            thickness_error_nm=25.0,
            thickness_confidence=0.9,
            uniformity_ratio=0.25,
            surface_roughness_nm=5.0,
            quality_score=0.85
        )
        
        # 厚度偏差过大的测量
        deviation_measurement = ProcessedMeasurement(
            measurement_id="test_deviation",
            timestamp=time.time() + 1,
            electrolyte_type="sulfide", 
            material_name="LPSCl",
            spr_angle_deg=65.0,
            reflectance=0.15,
            phase_shift_rad=1.5,
            thickness_nm=1060.0,  # 60nm偏差，应该触发报警
            thickness_error_nm=25.0,
            thickness_confidence=0.9,
            uniformity_ratio=0.25,
            surface_roughness_nm=5.0,
            quality_score=0.85
        )
        
        # 检查报警
        alerts = self.alert_system.check_measurement(deviation_measurement, normal_measurement)
        
        # 应该产生厚度偏差报警
        thickness_alerts = [a for a in alerts if a.alert_type == AlertType.THICKNESS_DEVIATION]
        self.assertTrue(len(thickness_alerts) > 0, "未检测到厚度偏差报警")
        
        print(f" 报警系统响应测试通过，产生 {len(alerts)} 个报警")
    
    def _create_test_intensity_data(self) -> np.ndarray:
        """创建测试用强度数据"""
        size = 64
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        
        # 创建接近目标均匀度的光斑
        sigma = 1.2
        intensity = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        # 添加一些噪声
        intensity += 0.05 * np.random.random((size, size))
        intensity = np.clip(intensity, 0, 1)
        
        return intensity
    
    def _create_spot_with_uniformity(self, target_uniformity: float) -> np.ndarray:
        """创建具有指定均匀度的光斑"""
        size = 50
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # 根据目标均匀度调整参数
        sigma = 0.5 + target_uniformity * 2
        spot = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        # 添加适量噪声
        noise_level = target_uniformity * 0.1
        spot += noise_level * np.random.random((size, size))
        
        return np.clip(spot, 0, 1)
    
    def tearDown(self):
        """清理测试资源"""
        try:
            self.data_processor.stop_processing()
        except:
            pass
        
        # 清理测试文件
        test_files = ["test_monitoring.db", "test_alerts.log"]
        for file in test_files:
            path = Path(file)
            if path.exists():
                try:
                    path.unlink()
                except:
                    pass


def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n" + "="*60)
    print("电解质监测系统性能基准测试")
    print("="*60)
    
    # SPR测量速度测试
    monitor = SPRElectrolyteMonitor(
        electrolyte_type=ElectrolyteType.SULFIDE,
        electrolyte_material="LPSCl"
    )
    
    start_time = time.time()
    angles, reflectance, phases = monitor.measure_spr_curve()
    spr_time = time.time() - start_time
    
    print(f"SPR曲线测量时间: {spr_time:.3f} 秒 ({len(angles)} 个数据点)")
    
    # 厚度分析速度测试
    analyzer = ThicknessAnalyzer()
    angles_subset = angles[::10]  # 每10个点取一个
    reflectance_subset = reflectance[::10]
    
    start_time = time.time()
    result = analyzer.fit_thickness_multi_angle(
        list(angles_subset), list(reflectance_subset), 2.075, (100, 2000)
    )
    thickness_time = time.time() - start_time
    
    print(f"厚度分析时间: {thickness_time:.3f} 秒")
    print(f"测量精度: ±{result.thickness_error_nm:.1f} nm")
    
    # 表面质量分析速度测试
    quality_monitor = SurfaceQualityMonitor()
    test_data = np.random.random((64, 64))
    
    start_time = time.time()
    quality_metrics = quality_monitor.comprehensive_quality_assessment(test_data)
    quality_time = time.time() - start_time
    
    print(f"表面质量分析时间: {quality_time:.3f} 秒")
    
    # 总处理时间
    total_time = spr_time + thickness_time + quality_time
    print(f"单次完整测量总时间: {total_time:.3f} 秒")
    
    if total_time < 5.0:
        print(" 性能满足实时监测要求 (<5秒)")
    else:
        print(" 性能可能需要优化以满足实时要求")


def main():
    """主测试函数"""
    print("固态电池电解质监测系统 - 综合测试")
    print("="*60)
    
    # 运行单元测试
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestElectrolyteMaterials,
        TestSPRElectrolyteMonitor, 
        TestThicknessAnalyzer,
        TestSurfaceQualityMonitor,
        TestIntegratedSystem
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 性能基准测试
    if result.wasSuccessful():
        run_performance_benchmark()
        
        print("\n" + "="*60)
        print(" 所有测试通过！系统功能正常")
        print("系统已准备好进行电解质层厚度与表面质量监测")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(" 存在测试失败，请检查系统配置")
        print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 