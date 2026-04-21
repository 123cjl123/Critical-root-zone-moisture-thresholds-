import rioxarray
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ================= 🔧 1. 全路径配置 =================

# 输出 CSV 文件的保存路径
OUTPUT_CSV = r"D:\Paper_1\paper_all\shuju_outout\xunlian1.csv"

# --- 目标变量 (Y) ---
FILE_THRESHOLD = r"D:\Paper_1\paper_all\3.yuzhi_results2\Global_Thresholds_20th_IDW_Filled.tif"

# --- 分组变量 ---
FILE_AI = r"D:\Paper_1\paper_all\AI\AI_025.tif"
FILE_FOREST_TYPE = r"D:\Paper_1\paper_all\森林数据及代码\SENLIN_0.25.tif"

# --- 解释变量 (X) - 全部写全路径 ---
FEATURES = {
    # === Group 1: 水分供给与状态 (Water Supply & State) ===
    "Precipitation": r"D:\Paper_1\paper_all\shuju_outout\Precipitation.tif",
    "SMrz":          r"D:\Paper_1\paper_all\shuju_outout\SMrz.tif",
    "SMs":           r"D:\Paper_1\paper_all\shuju_outout\SMs.tif",

    # === Group 2: 大气能量与需求 (Energy & Demand) ===
    "T2m":           r"D:\Paper_1\paper_all\shuju_outout\T2m.tif",
    "VPD":           r"D:\Paper_1\paper_all\shuju_outout\VPD.tif",
    "Radiation":     r"D:\Paper_1\paper_all\shuju_outout\Radiation.tif",

    # === Group 3: 植被结构与生理 (Vegetation) ===
    "LAI":           r"D:\Paper_1\paper_all\shuju_outout\LAI.tif",
    "TreeHeight":    r"D:\Paper_1\paper_all\shuju_outout\ETH_CanopyHeight_2020_Full_025deg.tif",

    # === Group 4: 土壤质地 (Soil Texture) ===
    "Sand":          r"D:\Paper_1\paper_all\shuju_outout\SoilGrids_Sand_0-100cm_Weighted_025deg.tif",
    "Clay":          r"D:\Paper_1\paper_all\shuju_outout\SoilGrids_Clay_0-100cm_Weighted_025deg.tif",
    "CFVO":          r"D:\Paper_1\paper_all\shuju_outout\SoilGrids_CFVO_Gravel_0-100cm_Weighted_025deg.tif",
    "BulkDensity":   r"D:\Paper_1\paper_all\shuju_outout\SoilGrids_BDOD_BulkDensity_0-100cm_Weighted_025deg.tif",
}


# ================= 🚀 2. 数据处理脚本 =================

def prepare_data():
    print("🚀 开始构建数据集 (已移除根系深度)...")

    # 1. 读取基准 (Threshold) - 用作空间参考
    if not os.path.exists(FILE_THRESHOLD):
        print(f"❌ 找不到 Y 变量文件: {FILE_THRESHOLD}")
        return

    # 读取并压缩多余维度 (squeeze)
    da_y = rioxarray.open_rasterio(FILE_THRESHOLD).squeeze()

    # 初始化字典
    data_dict = {}
    data_dict["Threshold"] = da_y.values.flatten()

    # 2. 读取 AI 并强制对齐到 Y
    print("   -> 读取 AI...")
    if os.path.exists(FILE_AI):
        da_ai = rioxarray.open_rasterio(FILE_AI).squeeze().rio.reproject_match(da_y)
        data_dict["AI"] = da_ai.values.flatten()
    else:
        print(f"❌ AI 文件不存在: {FILE_AI}");
        return

    # 3. 读取 ForestType 并强制对齐
    print("   -> 读取 森林类型...")
    if os.path.exists(FILE_FOREST_TYPE):
        da_ft = rioxarray.open_rasterio(FILE_FOREST_TYPE).squeeze().rio.reproject_match(da_y)
        data_dict["ForestType"] = da_ft.values.flatten()
    else:
        print(f"❌ 森林类型文件不存在: {FILE_FOREST_TYPE}");
        return

    # 4. 读取所有特征 (Features)
    print("   -> 读取解释变量 (Features)...")
    for name, full_path in tqdm(FEATURES.items(), desc="Loading Features"):
        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"   ⚠️ 警告: 文件不存在，将跳过变量: {name}")
            print(f"      路径: {full_path}")
            continue

        try:
            # 读取 -> 挤压维度 -> 强制重投影对齐(reproject_match) -> 展平
            da_x = rioxarray.open_rasterio(full_path).squeeze().rio.reproject_match(da_y)
            data_dict[name] = da_x.values.flatten()
        except Exception as e:
            print(f"❌ 读取 {name} 失败: {e}")
            return

    # 5. 转 DataFrame
    print("   -> 正在构建 DataFrame (这可能需要几秒钟)...")
    df = pd.DataFrame(data_dict)
    print(f"   -> 原始像元总数: {len(df)}")

    # 6. 数据清洗
    # a. 删除空值 (只要任何一列有空值就删)
    df_clean = df.dropna()
    print(f"   -> 删除空值后的行数: {len(df_clean)}")

    # b. 🌟 严格只保留森林类型 1-5
    df_clean = df_clean[df_clean['ForestType'].isin([1, 2, 3, 4, 5])]
    print(f"   -> 仅保留森林(1-5)后的行数: {len(df_clean)}")

    # c. 划分 AI 区域 (Zone 1: 半干旱, Zone 2: 半湿润, Zone 3: 湿润)
    print("   -> 划分气候区...")
    df_clean['AI_Zone'] = 0
    df_clean.loc[(df_clean['AI'] >= 0.2) & (df_clean['AI'] < 0.5), 'AI_Zone'] = 1
    df_clean.loc[(df_clean['AI'] >= 0.5) & (df_clean['AI'] < 0.65), 'AI_Zone'] = 2
    df_clean.loc[df_clean['AI'] >= 0.65, 'AI_Zone'] = 3

    # d. 只保留有效 AI 区 (Zone > 0)
    df_final = df_clean[df_clean['AI_Zone'] > 0]

    # 7. 保存
    print(f"   -> 最终进入分析的样本数: {len(df_final)}")

    # 自动创建输出目录(如果不存在)
    out_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 数据准备完成: {OUTPUT_CSV}")

    # 打印变量列表以确认
    print("-" * 30)
    print("   最终包含的变量:")
    for col in df_final.columns:
        print(f"   - {col}")
    print("-" * 30)


if __name__ == "__main__":
    prepare_data()