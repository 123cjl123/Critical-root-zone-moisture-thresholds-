import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 🔧 参数配置 =================
# 输入数据 (这里用 GLEAM 的阈值做展示，也可以换成 ERA5)
FILE_THRESH = r"D:\Paper_1\Spatial_Analysis_Data\Global_Thresholds_20th_Relaxed.tif"
FILE_AI = r"D:\Paper_1\AI\AI_Result0.25_TIFs\AI_Official_Resampled_025_Filtered.tif"
FILE_FOREST = r"D:\Paper_1\ESA\senlin\senlin0.25.tif"

OUTPUT_DIR = r"D:\Paper_1\3AI+5forest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 您的森林分类
FOREST_LABELS = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "MF"}

# 绘图字体
plt.rcParams['font.sans-serif'] = ['Arial']


def run_climate_forest_boxplot():
    print("Step 1: Loading & Aligning Data...")

    da_thresh = rioxarray.open_rasterio(FILE_THRESH).isel(band=0).squeeze()
    da_ai = rioxarray.open_rasterio(FILE_AI).isel(band=0).squeeze()
    da_forest = rioxarray.open_rasterio(FILE_FOREST).isel(band=0).squeeze()

    # 对齐
    da_ai = da_ai.rio.reproject_match(da_thresh)
    da_forest = da_forest.rio.reproject_match(da_thresh, resampling=0)  # 最近邻

    print("Step 2: Building DataFrame...")
    df = pd.DataFrame({
        'Threshold': da_thresh.values.flatten(),
        'AI': da_ai.values.flatten(),
        'ForestID': da_forest.values.flatten()
    })

    # 过滤：只保留 5 种森林 + 有效阈值
    df = df.dropna()
    df = df[df['ForestID'].isin([1, 2, 3, 4, 5])]
    df['Forest'] = df['ForestID'].map(FOREST_LABELS)

    # 🌟 关键：严格按照您的 3 档划分 🌟
    conditions = [
        (df['AI'] >= 0.2) & (df['AI'] < 0.5),  # Semi-Arid
        (df['AI'] >= 0.5) & (df['AI'] < 0.65),  # Dry Sub-Humid
        (df['AI'] >= 0.65)  # Humid
    ]
    choices = ['Semi-Arid\n(0.2-0.5)', 'Dry Sub-Humid\n(0.5-0.65)', 'Humid\n(>0.65)']

    df['Climate Zone'] = np.select(conditions, choices, default='Other')
    # 剔除不在这三档里的像素
    df = df[df['Climate Zone'] != 'Other']

    # 定义排序：从干到湿
    order_list = ['Semi-Arid\n(0.2-0.5)', 'Dry Sub-Humid\n(0.5-0.65)', 'Humid\n(>0.65)']

    print("Step 3: Plotting...")
    plt.figure(figsize=(10, 6), dpi=300)

    # 🎨 绘图逻辑：
    # X轴 = 您的3个气候区
    # Y轴 = 阈值
    # 颜色 (Hue) = 您的5种森林
    sns.boxplot(
        data=df,
        x='Climate Zone',
        y='Threshold',
        hue='Forest',  # 按森林类型分色
        order=order_list,  # 保证顺序是从左到右变湿
        palette='viridis',  # 颜色主题
        showfliers=False,  # 不显示异常值点(会让图更干净)
        width=0.7,
        linewidth=1.2
    )

    plt.title("Critical Soil Moisture Thresholds across Climate Zones and Forest Types", fontsize=14)
    plt.ylabel("Critical Threshold ($m^3/m^3$)", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Forest Type', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "Fig_Discussion_Climate_Gradient.png")
    plt.savefig(out_path)
    print(f"✅ 图表已保存: {out_path}")


if __name__ == "__main__":
    run_climate_forest_boxplot()