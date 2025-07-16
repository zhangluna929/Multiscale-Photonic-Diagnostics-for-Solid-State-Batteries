# Contributing Guide / 贡献指南

感谢你对 Optical SPR & Beam-Shaping  的兴趣！以下说明帮助你快速参与开发。

## 1. 环境 / Environment
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```
> requirements-dev.txt 可自行添加 black、pytest、mypy 等。  
> Python ≥3.9 建议。

## 2. 分支策略 / Branching
* **main** 受保护，仅合并通过 Review 的 PR。  
* **feature/***  开发新功能。  
* **fix/***      Bug 修复。  
* **docs/***     纯文档更新。

## 3. 代码规范 / Style
* Black + isort 格式化；提交前 `pre-commit run --all-files`。  
* 类型标注：mypy `strict` 模式通过。  
* Docstring 使用 Google 风格，中英可混排。

## 4. 提交规范 / Commit
```
<type>: <short desc>

<body>  # 可选，解释动机、解决方案
```
* type = feat / fix / docs / refactor / test / chore

## 5. 单元测试 / Tests
* PyTest 放在 `tests/`，新增功能请覆盖 80%+ 代码。  
* CI (GitHub Actions) 会自动跑 `pytest -q`。  
* 如需大型数值文件请使用 pytest fixtures 或生成伪数据。

## 6. PR 模板 / Pull Request
* 描述变更点、动机、相关 issue。  
* Checklist：
  - [ ] 代码格式通过 pre-commit  
  - [ ] 单元测试通过 & 覆盖  
  - [ ] 文档更新  

## 7. Issue 报告
* 描述环境、复现步骤、期望结果、实际结果。  
* 提供最小可复现脚本 / 数据。

## 8. Roadmap
欢迎认领：
1. **材料数据库自动同步**：抓取 NREL/Illuminant GitHub 并缓存。  
2. **GPU 加速 3-D 追迹**：PyTorch 或 Cupy。  
3. **多目标 NSGA-II 优化**：整合 Optuna v3 多目标。  
4. **深度模型**：预测 dn/dT, dn/dσ。

## 9. 问题 & 联系方式
* GitHub Discussions / Issues 皆可。  

Happy tracing! 光随心动 🥂 
