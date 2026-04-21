import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import shutil
import joblib
from sklearn.model_selection import train_test_split

# ================= 🔧 基础配置 =================
CSV_FILE = r"D:\Paper_1\paper_all\shuju_outout\xunlian1.csv"
SAVE_ROOT = r"D:\Paper_1\paper_all\saved_models_pkl_noSMrz"  # 结果保存仓库

# 每次运行时清空旧的模型文件，防止混淆
if os.path.exists(SAVE_ROOT): shutil.rmtree(SAVE_ROOT)
os.makedirs(SAVE_ROOT)

# 映射字典
AI_ZONES = {1: "Semi-Arid(0.2-0.5)", 2: "Dry_Sub-Humid(0.5-0.65)", 3: "Humid(＞0.65)"}
FOREST_TYPES = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "MF"}

FEATURE_COLS = [
    'Precipitation', #'SMrz',
    'T2m', 'VPD', 'Radiation',
    'LAI', 'TreeHeight',
    'Sand', 'Clay', 'CFVO', 'BulkDensity'
]


# ================= 🤖 核心训练函数 =================
def train_and_save(df_sub, group_name, sub_name):
    print(f"   🚀 Training: [{group_name} - {sub_name}] (n={len(df_sub)})")

    X = df_sub[FEATURE_COLS]
    y = df_sub['Threshold']

    # 1. 训练 XGBoost
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=6,
        n_jobs=-1, random_state=42, eval_metric="rmse", early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 2. 计算 SHAP
    # (这是最慢的一步，但因为保存了结果，以后改图再也不用跑这步了)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # 3. 打包数据
    data_pack = {
        "model": model,
        "shap_values": shap_values,
        "X": X,
        "y": y,
        "group_name": group_name,
        "sub_name": sub_name
    }

    # 4. 保存为 .pkl 文件
    save_dir = os.path.join(SAVE_ROOT, group_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # 处理一下文件名里的特殊符号
    safe_sub_name = sub_name.replace("＞", "gt").replace("(", "").replace(")", "").replace(" ", "_")
    save_path = os.path.join(save_dir, f"{safe_sub_name}.pkl")

    joblib.dump(data_pack, save_path, compress=3)
    print(f"      ✅ Saved to: {save_path}")


# ================= 🚀 主程序 (全量 + 分组) =================
if __name__ == "__main__":
    print("Step 1: Reading Data...")
    if not os.path.exists(CSV_FILE):
        print(f"❌ File not found: {CSV_FILE}")
        exit()

    df = pd.read_csv(CSV_FILE)
    print(f"   -> Total Rows: {len(df)}")

    # --- PART A: Global Analysis (全量) ---
    print("\n🌍 === 1. Global Analysis (All Data) ===")
    train_and_save(df, "Global", "All_Data")

    # --- PART B: Nested Analysis (分组循环) ---
    print("\n🌲 === 2. Nested Analysis (Zones & Types) ===")
    for z_id, z_name in AI_ZONES.items():
        df_ai = df[df['AI_Zone'] == z_id]

        if len(df_ai) == 0: continue

        print(f"\n📍 Zone: {z_name}")
        for ft_id, ft_name in FOREST_TYPES.items():
            df_target = df_ai[df_ai['ForestType'] == ft_id]

            # 样本量少于100就跳过
            if len(df_target) < 100:
                print(f"   ⚠️ Skip {ft_name} (Sample size {len(df_target)} < 100)")
                continue

            train_and_save(df_target, z_name, ft_name)

    print(f"\n🎉 All models trained and saved in:\n   {SAVE_ROOT}")
    print("👉 Now run 'Step2_Load_and_Plot.py' to generate images.")