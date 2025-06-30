# Optical Ray Tracing for Materials & Battery Interfaces  
**多镜组光学追迹与光斑分布分析（面向材料和电池界面研究）**

An **interactive Jupyter Notebook** demo to explore how multi-mirror optical systems shape beam profiles and divergence—tailored for nanomaterials & battery interface applications.  
这是一个交互式 Jupyter Notebook 演示，展示多镜组如何影响光斑分布与发散角，特别面向纳米材料和电池界面研究。

---

## 📘 Project Overview / 项目概述

- **Objective / 目标**  
  - Simulate and visualize ray tracing through configurable mirror arrays  
  - Analyze spot intensity distribution and beam divergence on a sample “screen”  
  - Provide material‐property interfaces (refractive index, absorption) for future nano/battery studies  
  模拟并可视化光线通过可配置镜组后的路径；分析屏幕上光斑分布和发散角；预留材料属性输入（折射率、吸收率）以便后续纳米/电池研究。

- **Key Features / 核心功能**  
  1. **Multi-mirror configuration** / 多镜组配置  
  2. **Interactive sliders** / 可交互滑块：镜子数量、发散角、材料折射率、吸收率  
  3. **Spot heatmap & divergence** / 光斑热力图与发散角计算  
  4. **Bilingual code & docs** / 中英双语注释与说明  
  5. **Extendable material interface** / 可扩展的材料属性接口  

---

## 🛠 Installation & Requirements / 安装与依赖

```bash
# Clone this repo
git clone https://github.com/你的用户名/Optical-Ray-Tracing-for-Materials.git
cd Optical-Ray-Tracing-for-Materials

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

# Install dependencies
pip install numpy matplotlib ipywidgets
