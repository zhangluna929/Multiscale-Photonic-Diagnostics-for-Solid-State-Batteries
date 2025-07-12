# Multiscale Photonic Diagnostics Suite for Solid-State Batteries  
# 多尺度光学诊断&束形优化平台（SPR × Beam-Shaping）

> No magic, just photons & Python.  
> 代码不整花活，踏踏实实追光。

What it does (nothing more, nothing less):  
它到底干啥：

* Multi-mirror ray tracing (2-D geom optics)  
  多镜组二维几何光线追迹
* Kretschmann-style SPR modelling (transfer-matrix, TM only)  
  Kretschmann 结构 SPR 传输矩阵建模（仅 TM）
* Multi-layer stacks & temperature drift  
  多层薄膜与温漂效应分析
* Gaussian → flat-top beam shaping optimisation  
  高斯到平顶光斑整形优化
* CLI & GUI (Streamlit) front-end  
  命令行 + Streamlit 可视化界面

## Why this repo? / 初衷

> 从做纳米金膜 SPR 传感到固态锂电池界面表征，设备换了好几版，Bug 却一直没少。于是干脆写一套“材料 → 光学 → 电池”一条龙脚本，省得学弟学妹踩坑。

**科研脉络 / Research lineage**
1. _Nanoplasmonics (2015–2018)_  
   - e-beam 蒸镀 45 nm Au + 2 nm Cr  
   - Kretschmann SPR 监测多肽自组装，灵敏度 1.5×10-6 RIU  
2. _Porous‐Electrode Engineering (2019–2021)_  
   - Bruggeman 模型 -> NMC811 电极孔隙率-SOC 曲线  
   - 相位-SPR 检测 5 nm PDA-PEI 涂层，Δφ≈12 mrad  
3. _Solid-State Batteries (2022–now)_  
   - LLZO / LiPON / LATP 多层栈，温漂 + 应力耦合  
   - 圆柱壳封装 + 激光扫描，OTA 质检接入 Kafka  

如果这条进化链跟你的研究路线撞车——恭喜，直接拿去改。

## Feature matrix / 功能矩阵
| 功能 | 关键文件 | 备注 |
|------|-----------|------|
| 2-D 光束整形 | `geometry.py` `tracer.py` | 多镜组，高斯→平顶
| SPR 反射率 / 相位 | `elements.py` | 3-层 T-matrix，|E|²
| 多层膜温-应力漂移 | `stack.py` | `thermo_mech_drift()`
| 多孔电极 SOC | `porous.py` | Bruggeman / MG EMA
| 3-D 封装曲率 | `curved.py` | 圆柱壳+Ray3D
| Optuna 自动调参 | `optimizer.py` | 镜组 + 金膜厚度
| Kafka 产线推流 | `utils.export_results_kafka` | MES-ready

> Still feels like overkill? Remember: “任何足够复杂的电池实验，终将重现一台 Web 服务器。”

## Quick taste / 两分钟上手
```bash
pip install -r requirements.txt  # 准备咖啡
streamlit run streamlit_app.py   # 拿出爆米花
```
滑动镜子数量 / 金膜厚度，右侧立即刷新：⬇️  
<img src="docs/img/demo.gif" width="600" />

## Tech highlights / 技术亮点
* **Phase-SPR**：`get_phase_shift()` nm-level 锂枝晶早期预警  
* **Thermo-optic + Photo-elastic**：LLZO dn/dT≈1×10-4 K-1，dn/dσ≈2×10-5 MPa-1  
* **Cylindrical Tracing**：10 mm 直径圆柱壳，透过率损失 <3 %  
* **Beam-shaping Optuna**：20-trial demo → 均匀度提升 40 %，SPR dip 深 50 %  
* **Kafka Live QC**：>10 Hz 推流，生产线实时抓异常

## Roadmap 1.0
- [ ] NREL 自动同步材料库
- [ ] GPU 光线追迹 (CuPy / Torch)
- [ ] NSGA-II 多目标（slope, fwhm, Δφ）
- [ ] ML 预测 dn/dT, dn/dσ

> Pull requests welcomed — photons are cheap, lab time is not.

## Repo layout  |  目录结构
```
optics/          core package | 核心代码
  ├─ ray.py      – Ray object | 光线类
  ├─ elements.py – Mirror / Prism / MetalFilm
  ├─ tracer.py   – multi-element ray tracer | 多元件追迹
  ├─ stack.py    – transfer matrix & temp drift | 传输矩阵/温漂
  ├─ metrics.py  – peak_slope, fwhm
  ├─ geometry.py – generators | 几何生成器
  ├─ optimizer.py – Optuna pipeline | 优化管线
examples/        notebooks & scripts | 示例
docs/            API & theory | 文档
streamlit_app.py GUI demo
```

## Citation  |  引用
If you use this code, cite:  
如使用本工具，请引用：

```
@software{spr_toolkit_2025,
  author = {Grandpa-Coder},
  title  = {Open SPR & Beam-Shaping Toolkit},
  year   = {2025},
  url    = {https://github.com/your/repo}
}
``` 