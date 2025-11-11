"""
@Time    : 2025/11/11 10:36
@Author  : Zhang Hao yv
@File    : main.py
@IDE     : PyCharm
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.multiclass import OneVsRestClassifier
import joblib
import numpy as np
from pandas.api.types import is_numeric_dtype
from imblearn.over_sampling import SMOTE  # 添加SMOTE处理不平衡

from move import classify_and_move_files

# 忽略sklearn警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 设置数据显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Kaggle 数据来源信息（报告需要）
DATA_SOURCE = "本项目数据集来源：Kaggle 空气质量监测公开数据集（Air Quality Dataset）。"


def load_and_explore_data(file_path='data/.xls'):
    """加载并初步探索数据"""
    print(DATA_SOURCE)
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到，请确认文件路径 '{file_path}' 是否正确。")
        return None
    try:
        df = pd.read_excel(file_path, engine='xlrd')

        print("\n===== 数据基本信息 =====")
        print(f"数据集形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        print("\n===== 数据样本（前 5 行）=====")
        print(df.head())

        print("\n===== 字段类型 =====")
        print(df.dtypes)

        print("\n===== 描述性统计 =====")
        print(df.describe())

        return df
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None


def preprocess_data(df):
    """数据预处理：去重、填充缺失值、剔除异常值、缩尾处理"""
    print("\n===== 数据预处理 =====")

    # 缺失值检查
    print("\n缺失值情况：")
    print(df.isnull().sum())

    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    # 数值列进行 IQR 异常值处理
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    initial_shape = df.shape
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[df[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]

    print(f"异常值处理前数据集: {initial_shape}")
    print(f"异常值处理后数据集: {df.shape}")
    return df


def feature_engineering(df):
    """构造新特征：PM 比值、月份"""
    print("\n===== 特征工程 =====")

    # 月份已存在，无需提取

    # PM2.5 与 PM10 比值
    if "PM2.5" in df.columns and "PM10" in df.columns:
        df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-6)

    # 添加AQI与PM2.5的交互特征
    if "AQI" in df.columns and "PM2.5" in df.columns:
        df["AQI_PM_interaction"] = df["AQI"] * df["PM2.5"]

    print("已构造特征：PM_ratio, AQI_PM_interaction")
    return df


def visualize_data(df):
    """绘制更完整的数据探索图表"""
    print("\n===== 数据可视化 =====")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 标签分布直方图
    if "质量等级" in df.columns:
        plt.figure(figsize=(6, 4))
        df["质量等级"].value_counts().plot(kind="bar")
        plt.title("空气质量等级分布")
        plt.tight_layout()
        plt.savefig("quality_dist.png")
        plt.show()

    # 相关性热力图（无日期列，跳过时间序列）
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("污染物相关性热力图")
    plt.tight_layout()
    plt.savefig("corr_heatmap.png")
    plt.show()


def select_features_and_label(df):
    """选择特征列与标签"""
    label = "质量等级"
    if label not in df.columns:
        raise ValueError("未找到标签列：质量等级")

    candidates = ["AQI", "PM2.5", "PM10", "CO", "SO2", "NO2", "O3", "PM_ratio", "月份", "AQI_PM_interaction"]
    # 只选择数值型特征
    features = [c for c in candidates if c in df.columns and is_numeric_dtype(df[c])]

    if not features:
        raise ValueError("没有找到有效的数值特征列")

    df_model = df[features + [label]].dropna()

    print(f"\n建模特征：{features}")
    print("标签：质量等级")
    print(f"有效样本数：{len(df_model)}")

    return df_model, features, label


def plot_confusion_matrix(y_true, y_pred, name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()


def compute_roc_auc(y_true, model, X, name):
    """计算多类ROC-AUC"""
    if not hasattr(model, 'predict_proba'):
        print(f"{name} ROC-AUC 计算失败（无概率输出）")
        return
    try:
        y_prob = model.predict_proba(X)
        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_true)
        auc = roc_auc_score(y_true_enc, y_prob, multi_class='ovr')
        print(f"{name} ROC-AUC (OVR): {auc:.4f}")
    except ValueError as e:
        print(f"{name} ROC-AUC 计算失败: {e}")


def build_and_evaluate_models(df):
    """模型训练 + 参数调优 + k-fold 交叉验证 + 混淆矩阵 + ROC-AUC + 模型保存"""
    df_model, features, label = select_features_and_label(df)
    X = df_model[features].values  # 转换为numpy数组以避免特征名警告
    y = df_model[label].values

    # 检查类别分布并调整cv folds
    unique, counts = np.unique(y, return_counts=True)
    print(f"类别分布: {dict(zip(unique, counts))}")
    min_class_size = min(counts)
    n_splits = min(5, max(2, min_class_size // 2)) if min_class_size >= 2 else 2
    print(f"最小类别样本数: {min_class_size}, 使用 {n_splits}-fold CV")

    # 使用SMOTE处理不平衡，调整k_neighbors以避免错误
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
    print(f"SMOTE k_neighbors: {k_neighbors}")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"SMOTE后样本数: {len(y_res)}, 类别分布: {dict(zip(*np.unique(y_res, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    models = {}

    # ===== 决策树参数调优 =====
    dt_params = {"max_depth": [3, 5, 8, 12], "criterion": ["gini", "entropy"]}
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=n_splits, scoring='accuracy')
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    models['决策树'] = best_dt

    # ===== SVM 参数调优 =====
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, random_state=42))
    ])
    svm_params = {"svm__C": [0.1, 1, 5], "svm__kernel": ["rbf", "linear"]}
    svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=n_splits, scoring='accuracy')
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    models['SVM'] = best_svm

    # ===== RandomForest 参数调优（新增）=====
    rf_params = {"n_estimators": [50, 100], "max_depth": [5, 10], "min_samples_split": [2, 5]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=n_splits, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    models['RandomForest'] = best_rf

    # ===== 交叉验证 =====
    print("\n===== {}-fold 交叉验证 =====".format(n_splits))
    for name, model in models.items():
        scores = cross_val_score(model, X_res, y_res, cv=n_splits, scoring='accuracy')
        print(f"{name} 平均准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # ===== 测试集评估 =====
    def metrics(name, model):
        y_pred = model.predict(X_test)
        print(f"\n===== {name} 评估 =====")
        print(classification_report(y_test, y_pred, zero_division=0))
        plot_confusion_matrix(y_test, y_pred, name)
        compute_roc_auc(y_test, model, X_test, name)

    for name, model in models.items():
        metrics(name, model)

    # ===== 保存模型 =====
    for name, model in models.items():
        joblib.dump(model, f"{name.lower().replace(' ', '_')}.pkl")
    print("模型已保存：decision_tree.pkl, svm.pkl, random_forest.pkl")

    # ===== 新数据预测示例 =====
    print("\n===== 新样本预测示例 =====")
    sample = X_test[:5]
    print("输入特征：")
    print(sample)
    for name, model in models.items():
        print(f"{name} 预测：", model.predict(sample))


def main():
    print("空气质量统计项目启动...")
    df = load_and_explore_data()
    if df is not None:
        df = preprocess_data(df)
        df = feature_engineering(df)
        visualize_data(df)
        build_and_evaluate_models(df)

    classify_and_move_files()


if __name__ == "__main__":
    main()