# CourierBoxAI — 空气质量等级分类项目

本项目使用 Python 完成空气质量数据的清洗、特征工程、探索性分析与可视化，并基于决策树、SVM 与随机森林构建分类模型，输出评估指标与混淆矩阵图，并持久化模型文件以便复用。

- 完整报告：`report.md`（同时提供 `report.html`）
- 主要脚本：`main.py`（一键执行完整流程）
- 运行环境：Python `3.12`

---

## 快速开始

- 安装 uv
```bash
pip install uv
```

- 创建虚拟环境
```bash
uv venv
```

- 激活虚拟环境（Windows）
```bash
.\.venv\Scripts\activate
```

- 同步依赖（使用 `uv.lock` 与 `pyproject.toml`）
```bash
uv sync
```

- 运行主程序（自动进行数据加载、预处理、特征工程、可视化、模型训练与评估）
```bash
uv run main.py
```

- 如需新增依赖（示例）
```bash
uv add scikit-learn imbalanced-learn
```
```bash
pip install pandas matplotlib seaborn xlrd scikit-learn imblearn
```

- 运行主程序（自动进行数据加载、预处理、特征工程、可视化、模型训练与评估）
```bash
python main.py
```

---

## 数据说明

- 数据来源：Kaggle 空气质量监测公开数据集（Air Quality Dataset）
- 存储位置：`data/.xls`
- 关键字段：
  - 标签：`质量等级`
  - 指标：`AQI`, `PM2.5`, `PM10`, `CO`, `SO2`, `NO2`, `O3`
  - 构造特征：`PM_ratio = PM2.5 / (PM10 + 1e-6)`, `AQI_PM_interaction = AQI * PM2.5`

---

## 目录结构

- `data/`：原始数据（Excel）
- `image/`：所有图像输出（统一路径）
  - `quality_dist.png`（标签分布）
  - `corr_heatmap.png`（相关性热力图）
  - `boxplot.png`、`heatmap.png`、`pairplot.png`（历史探索图）
  - `决策树_confusion_matrix.png`、`svm_confusion_matrix.png`、`randomforest_confusion_matrix.png`（最新混淆矩阵）
  - `cm_dt.png`、`cm_svm.png`（历史混淆矩阵）
- `temp/`：模型持久化文件（`.pkl`）
  - `decision_tree.pkl`、`svm.pkl`、`randomforest.pkl`、`决策树.pkl`
- `main.py`：主流程脚本
- `metrics.md`：评估指标（历史版本，最新指标以控制台输出为准）
- `report.md` / `report.html`：完整项目报告

---

## 运行产出

- 图像输出（自动保存到 `image/`）：
  - 标签分布、相关性热力图、三种模型的混淆矩阵
- 模型文件（保存到 `temp/`）：
  - `decision_tree.pkl`、`svm.pkl`、`randomforest.pkl`
- 指标文件：
  - `metrics.md`（早期版本记录；最新运行的评估指标已在控制台打印）

---

## 复用已训练模型（示例）

- 加载决策树模型并进行预测（示例）
```python
from joblib import load
import numpy as np

model = load(r"pkl\decision_tree.pkl")
# 示例：与训练一致的特征顺序（请根据实际特征列顺序构造输入）
# ['AQI','PM2.5','PM10','CO','SO2','NO2','O3','PM_ratio','AQI_PM_interaction']
X_sample = np.array([[90, 45, 85, 0.65, 4, 36, 134, 45/85, 90*45]])
print(model.predict(X_sample))
```

---

## 注意事项

- 评估数值可能随数据处理与参数搜索略有变化；请以“最近一次控制台输出”为准。
- 若出现类不平衡（如“优”样本极少），代码内已使用 SMOTE 过采样；仍建议补充样本与进行更细致的调参。
- 生成图像与模型的保存路径已统一到 `image/` 与 `temp/`，如需变更请在 `main.py` 中调整。

---

## 参考

- Kaggle: Air Quality Dataset[https://www.kaggle.com/datasets/hanwizardhanwizard/polling]
- scikit-learn / imbalanced-learn 文档
- 本项目将上传GitHub仓库：[https://github.com/yourusername/CourierBoxAI](https://github.com/yourusername/CourierBoxAI)



```
