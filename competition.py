# ============================================================
# 供水管网漏损智能监测系统 - 完整最终版
# 功能：
#   1. 数据加载与标签重构
#   2. 滑动窗口特征工程
#   3. 多模型对比（XGBoost / LightGBM / RandomForest）
#   4. 跨域泛化实验（源域→零样本→微调）
#   5. 主模型保存（供 Web 应用使用）
# 数据：Hourly Anomaly Scores and Leak Labels from a Multi-Source
#       Urban Water Distribution Network Dataset
# DOI: 10.5281/zenodo.15096167
# ============================================================

# ---------- 环境配置 ----------
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# 第 1 步：加载数据
# ============================================================
print("=" * 60)
print("第 1 步：加载数据")
print("=" * 60)

df = pd.read_csv("input_model_potenc_predXfault7_A.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"数据形状: {df.shape}")
print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")


# ============================================================
# 第 2 步：原始标签分布
# ============================================================
print("\n" + "=" * 60)
print("第 2 步：原始标签分布（±7 天窗口）")
print("=" * 60)

print(df["fault_d7"].value_counts())
print(f"原始漏损窗口占比: {df['fault_d7'].mean():.2%}")


# ============================================================
# 第 3 步：缺失值分析 + 丢弃高缺失列
# ============================================================
print("\n" + "=" * 60)
print("第 3 步：缺失值分析")
print("=" * 60)

missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"缺失数": missing, "缺失率(%)": missing_pct})
print(missing_df[missing_df["缺失数"] > 0])

drop_cols = missing_df[missing_df["缺失率(%)"] > 40].index.tolist()
drop_cols = [c for c in drop_cols if c not in ["timestamp", "fault_d7"]]
print(f"\n丢弃列（缺失>40%）: {drop_cols}")
df = df.drop(columns=drop_cols)


# ============================================================
# 第 4 步：标签重构
# ============================================================
print("\n" + "=" * 60)
print("第 4 步：标签重构 - 事件前后 24 小时为 1")
print("=" * 60)

df["fault_change"] = df["fault_d7"].diff()
df["event_id"] = (df["fault_change"] == 1).cumsum()

event_starts = df[df["fault_change"] == 1]["timestamp"].tolist()
print(f"共 {len(event_starts)} 个漏损事件")

df["fault_short"] = 0
for start in event_starts:
    mask = (df["timestamp"] >= start - pd.Timedelta(hours=24)) & \
           (df["timestamp"] <= start + pd.Timedelta(hours=24))
    df.loc[mask, "fault_short"] = 1

print(f"新标签漏损占比: {df['fault_short'].mean():.2%}")
print(f"新标签漏损时段数: {int(df['fault_short'].sum())}")


# ============================================================
# 第 5 步：特征工程
# ============================================================
print("\n" + "=" * 60)
print("第 5 步：特征工程")
print("=" * 60)

exclude_cols = ["timestamp", "fault_d7", "fault_short", "fault_change", "event_id"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# 5.1 缺失指示列
high_missing_cols = [c for c in feature_cols if df[c].isnull().sum() / len(df) > 0.05]
print(f"缺失率>5%的列（将加指示变量）: {high_missing_cols}")

for col in high_missing_cols:
    df[f"{col}_is_missing"] = df[col].isnull().astype(int)

# 5.2 中位数填充
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

# 5.3 滑动窗口特征
core_score_cols = [
    "gw_lvl_site2_anomaly_score",
    "gw_temp_site2_anomaly_score",
    "temp_site1_anomaly_score",
    "sfc_temp_site1_anomaly_score",
]

for col in core_score_cols:
    if col in df.columns:
        df[f"{col}_mean_24h"] = df[col].rolling(24, min_periods=1).mean()
        df[f"{col}_std_24h"] = df[col].rolling(24, min_periods=1).std().fillna(0)
        df[f"{col}_diff_24h"] = df[col] - df[f"{col}_mean_24h"]
        df[f"{col}_mean_72h"] = df[col].rolling(72, min_periods=1).mean()
        df[f"{col}_diff_72h"] = df[col] - df[f"{col}_mean_72h"]

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"最终特征列数: {len(feature_cols)}")
print(f"其中滑动窗口特征: {len([c for c in feature_cols if '_24h' in c or '_72h' in c])} 个")
print(f"其中缺失指示列: {len([c for c in feature_cols if c.endswith('_is_missing')])} 个")


# ============================================================
# 第 6 步：按事件切分
# ============================================================
print("\n" + "=" * 60)
print("第 6 步：按漏损事件切分数据")
print("=" * 60)

events = df[df["fault_short"] == 1]["event_id"].unique()
print(f"共 {len(events)} 个有事件的 event_id")

np.random.seed(42)
test_events = np.random.choice(events, size=max(1, int(len(events) * 0.2)), replace=False)
test_mask = df["event_id"].isin(test_events)
train_mask = ~test_mask

X = df[feature_cols]
y = df["fault_short"]

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"训练集: {len(X_train)} 条, 漏损占比: {y_train.mean():.2%}")
print(f"测试集: {len(X_test)} 条, 漏损占比: {y_test.mean():.2%}")


# ============================================================
# 第 7 步：多模型对比
# ============================================================
print("\n" + "=" * 60)
print("第 7 步：多模型对比（事件切分）")
print("=" * 60)

scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())

models = {
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, random_state=42, eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"model": model, "auc": auc, "y_prob": y_prob, "y_pred": y_pred}
    print(f"\n【{name}】")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["正常(0)", "漏损窗口(1)"]))

best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\n>>> 最优模型: {best_name} (AUC = {results[best_name]['auc']:.4f})")


# ============================================================
# 第 8 步：特征重要性
# ============================================================
print("\n" + "=" * 60)
print("第 8 步：特征重要性分析")
print("=" * 60)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n【特征重要性 Top 15】")
print(importance.head(15).to_string(index=False))

groups = {
    "能耗异常": [c for c in feature_cols if ("_kW" in c or "power_hour" in c or "kvar" in c) and "is_missing" not in c and "24h" not in c and "72h" not in c],
    "水源温度异常": [c for c in feature_cols if "ws_temp" in c and "is_missing" not in c and "24h" not in c and "72h" not in c],
    "水源活力异常": [c for c in feature_cols if "ws_vigor" in c and "is_missing" not in c and "24h" not in c and "72h" not in c],
    "水源水位异常": [c for c in feature_cols if "ws_level" in c and "is_missing" not in c and "24h" not in c and "72h" not in c],
    "环境温度异常": ["temp_site1_anomaly_score"],
    "地表温度异常": ["sfc_temp_site1_anomaly_score"],
    "地下水异常": ["gw_lvl_site2_anomaly_score", "gw_temp_site2_anomaly_score"],
    "滑动窗口特征": [c for c in feature_cols if "_24h" in c or "_72h" in c],
    "缺失指示": [c for c in feature_cols if c.endswith("_is_missing")],
}

group_importance = {}
for group, cols in groups.items():
    cols_in = [c for c in cols if c in feature_cols]
    if cols_in:
        group_importance[group] = importance[importance["feature"].isin(cols_in)]["importance"].sum()

print("\n【按异常来源分组的重要性】")
print(pd.Series(group_importance).sort_values(ascending=False).to_string())


# ============================================================
# 第 9 步：保存主模型
# ============================================================
print("\n" + "=" * 60)
print("第 9 步：保存主模型")
print("=" * 60)

joblib.dump({
    "model": best_model,
    "feature_cols": feature_cols,
    "best_name": best_name,
    "auc": results[best_name]["auc"],
}, "main_model.pkl")

print("主模型已保存: main_model.pkl")
print(f"  模型类型: {best_name}")
print(f"  特征数: {len(feature_cols)}")
print(f"  AUC: {results[best_name]['auc']:.4f}")


# ============================================================
# 第 10 步：跨域泛化实验（修正版）
# ============================================================
print("\n" + "=" * 60)
print("第 10 步：跨域泛化实验")
print("=" * 60)

slot_mapping = {
    "score_mean_24h": {
        "source": "gw_temp_site2_anomaly_score_mean_24h",
        "target": "temp_site1_anomaly_score_mean_24h"
    },
    "score_std_24h": {
        "source": "gw_temp_site2_anomaly_score_std_24h",
        "target": "temp_site1_anomaly_score_std_24h"
    },
    "score_diff_24h": {
        "source": "gw_temp_site2_anomaly_score_diff_24h",
        "target": "temp_site1_anomaly_score_diff_24h"
    },
    "score_mean_72h": {
        "source": "gw_temp_site2_anomaly_score_mean_72h",
        "target": "temp_site1_anomaly_score_mean_72h"
    },
    "score_diff_72h": {
        "source": "gw_temp_site2_anomaly_score_diff_72h",
        "target": "temp_site1_anomaly_score_diff_72h"
    },
}

valid_slots = {}
for slot, cols in slot_mapping.items():
    if cols["source"] in df.columns and cols["target"] in df.columns:
        valid_slots[slot] = cols

print(f"有效槽位数: {len(valid_slots)} / {len(slot_mapping)}")

source_slot_cols = [valid_slots[s]["source"] for s in valid_slots]
target_slot_cols = [valid_slots[s]["target"] for s in valid_slots]

# 源域训练
X_source = df[source_slot_cols]
y_source = df["fault_short"]

source_events = df[df["fault_short"] == 1]["event_id"].unique()
np.random.seed(42)
source_test_events = np.random.choice(
    source_events, size=max(1, int(len(source_events) * 0.2)), replace=False
)
source_test_mask = df["event_id"].isin(source_test_events)
source_train_mask = ~source_test_mask

X_src_train = X_source[source_train_mask]
y_src_train = y_source[source_train_mask]
X_src_test = X_source[source_test_mask]
y_src_test = y_source[source_test_mask]

src_model = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1
)
src_model.fit(X_src_train, y_src_train)

src_auc = roc_auc_score(y_src_test, src_model.predict_proba(X_src_test)[:, 1])
print(f"\n源域测试 AUC: {src_auc:.4f}")

# 零样本迁移
target_events = df[df["fault_short"] == 1]["event_id"].unique()
np.random.seed(42)
target_test_events = np.random.choice(
    target_events, size=max(1, int(len(target_events) * 0.2)), replace=False
)
target_test_mask = df["event_id"].isin(target_test_events)
target_train_mask = ~target_test_mask

X_tgt_test = df[target_slot_cols][target_test_mask].copy()
y_tgt_test = df["fault_short"][target_test_mask]
X_tgt_test.columns = source_slot_cols

y_tgt_prob_zero = src_model.predict_proba(X_tgt_test)[:, 1]
zero_shot_auc = roc_auc_score(y_tgt_test, y_tgt_prob_zero)
print(f"零样本迁移 AUC: {zero_shot_auc:.4f}")

# 少量微调
target_train_events = df[
    target_train_mask & (df["fault_short"] == 1)
]["event_id"].unique()
n_finetune_events = min(3, len(target_train_events))

if n_finetune_events > 0:
    np.random.seed(42)
    finetune_events = np.random.choice(
        target_train_events, size=n_finetune_events, replace=False
    )
    finetune_mask = target_train_mask & df["event_id"].isin(finetune_events)

    X_finetune = df[target_slot_cols][finetune_mask].copy()
    y_finetune = df["fault_short"][finetune_mask]
    X_finetune.columns = source_slot_cols

    finetuned_model = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    finetuned_model.fit(X_finetune, y_finetune)

    y_tgt_prob_ft = finetuned_model.predict_proba(X_tgt_test)[:, 1]
    finetuned_auc = roc_auc_score(y_tgt_test, y_tgt_prob_ft)
    print(f"微调后 AUC: {finetuned_auc:.4f}")
else:
    finetuned_auc = None
    finetuned_model = None

# 保存跨域模型
joblib.dump({
    "source_model": src_model,
    "finetuned_model": finetuned_model,
    "source_slot_cols": source_slot_cols,
    "target_slot_cols": target_slot_cols,
    "slot_mapping": valid_slots,
}, "domain_shift_models.pkl")

print("\n跨域模型已保存: domain_shift_models.pkl")
print(f" 源域 AUC: {src_auc:.4f}")
print(f" 零样本 AUC: {zero_shot_auc:.4f}")
print(f" 微调后 AUC: {finetuned_auc if finetuned_auc else 0:.4f}")

# ============================================================
# 第 11 步：所有画图（先画，最后统一 show）
# ============================================================
print("\n" + "=" * 60)
print("第 11 步：生成所有图表")
print("=" * 60)

# 图1：特征重要性 Top 15
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.barh(importance["feature"][:15], importance["importance"][:15], color="steelblue")
ax1.set_xlabel("Importance")
ax1.set_title(f"Feature Importance Top 15 ({best_name})")
ax1.invert_yaxis()
plt.tight_layout()
plt.savefig("fig1_feature_importance.png", dpi=150)

# 图2：分组重要性
fig2, ax2 = plt.subplots(figsize=(10, 6))
group_series = pd.Series(group_importance).sort_values(ascending=True)
ax2.barh(group_series.index, group_series.values, color="coral")
ax2.set_xlabel("Total Importance")
ax2.set_title("Importance by Anomaly Source")
plt.tight_layout()
plt.savefig("fig2_group_importance.png", dpi=150)

# 图3：多模型 AUC 对比
fig3, ax3 = plt.subplots(figsize=(8, 6))
model_names = list(results.keys())
aucs = [results[m]["auc"] for m in model_names]
ax3.bar(model_names, aucs, color=["steelblue", "coral", "seagreen"])
ax3.set_ylabel("AUC-ROC")
ax3.set_title("Model Comparison (Event-based Split)")
ax3.axhline(y=0.5, color="black", linestyle="--", label="Random")
for i, v in enumerate(aucs):
    ax3.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
ax3.legend()
plt.tight_layout()
plt.savefig("fig3_model_comparison.png", dpi=150)

# 图4：事件研究
event_study_cols = ["gw_temp_site2_anomaly_score", "gw_lvl_site2_anomaly_score"]
window_hours = 168
fig4, axes4 = plt.subplots(len(event_study_cols), 1, figsize=(12, 8), sharex=True)

for ax, col in zip(axes4, event_study_cols):
    if col not in df.columns:
        continue
    curves = []
    for start in event_starts:
        mask = (df["timestamp"] >= start - pd.Timedelta(hours=window_hours)) & \
               (df["timestamp"] <= start + pd.Timedelta(hours=window_hours))
        window_df = df[mask].sort_values("timestamp")
        if len(window_df) >= window_hours * 2 * 0.8:
            curves.append(window_df[col].values[:window_hours * 2 + 1])
    if curves:
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        mean_curve = np.nanmean(padded, axis=0)
        x_axis = np.arange(len(mean_curve)) - window_hours
        ax.plot(x_axis, mean_curve, color="darkred", linewidth=1.5)
        ax.axvline(x=0, color="black", linestyle="--", label="事件起点")
        ax.set_ylabel(col, fontsize=9)
        ax.set_title(f"{col} 事件前后 7 天变化（均值）")
        ax.legend()

plt.xlabel("相对事件起点的小时数（负=事件前，正=事件后）")
plt.tight_layout()
plt.savefig("fig4_event_study.png", dpi=150)

# 图6：跨域泛化
fig6, ax6 = plt.subplots(figsize=(9, 6))
stages = ["源域测试\n(站点2)", "零样本迁移\n(站点2→站点1)", "少量微调\n(站点1)"]
auc_values = [src_auc, zero_shot_auc, finetuned_auc if finetuned_auc else 0]
colors = ["steelblue", "coral", "seagreen"]
bars = ax6.bar(stages, auc_values, color=colors)
ax6.axhline(y=0.5, color="black", linestyle="--", label="Random")
ax6.set_ylabel("AUC-ROC")
ax6.set_title("跨域泛化实验：域偏移与迁移效果")
ax6.set_ylim(0, 1)
for bar, val in zip(bars, auc_values):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f"{val:.3f}", ha="center", fontsize=11)
ax6.legend()
plt.tight_layout()
plt.savefig("fig6_domain_shift.png", dpi=150)

print("已生成 5 张图: fig1, fig2, fig3, fig4, fig6")

# ============================================================
# 第 12 步：一次性弹出所有图
# ============================================================
print("\n所有图已保存，正在弹出窗口...")
plt.show()