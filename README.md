# Optical SPR & Beam-Shaping
# 光学 SPR & 光束整形

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

## Quick start  |  快速开始
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py  # interactive demo / 交互演示
```

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
