import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
import seaborn as sns
import os
import warnings
from datetime import datetime
import matplotlib as mpl

warnings.filterwarnings('ignore')

# ================= 🔧 1. 参数与路径配置 =================
FILE_THRESH = r""
FILE_FOREST = r""
OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOREST_LABELS = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "MF"}
MIN_PIXEL_COUNT = 50

# ================= 🎨 2. 配色方案 =================
COLOR_THEME = {
    'Boreal': '#A6CEE3',
    'Temperate': '#B2DF8A',
    'Tropical': '#FB9A99'
}


# ================= 🛠️ 3. 数据处理与绘图 =================
def run_biome_boxplot_final_fix_v2():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Loading Data...")

    # 读取与对齐
    da_thresh = rioxarray.open_rasterio(FILE_THRESH).isel(band=0).squeeze()
    da_forest = rioxarray.open_rasterio(FILE_FOREST).isel(band=0).squeeze()
    da_forest = da_forest.rio.reproject_match(da_thresh, resampling=0)
    da_lat = xr.broadcast(da_thresh.y, da_thresh.x)[0]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Building DataFrame...")
    df = pd.DataFrame({
        'Threshold': da_thresh.values.flatten(),
        'ForestID': da_forest.values.flatten(),
        'Latitude': da_lat.values.flatten()
    })

    df = df.dropna()
    df = df[df['ForestID'].isin([1, 2, 3, 4, 5])]
    df['Forest'] = df['ForestID'].map(FOREST_LABELS)

    # 划分纬度带
    abs_lat = df['Latitude'].abs()
    conditions = [
        (abs_lat < 23.5),  # Tropical
        (abs_lat >= 23.5) & (abs_lat < 50),  # Temperate
        (abs_lat >= 50)  # Boreal
    ]
    choices = ['Tropical', 'Temperate', 'Boreal']
    df['Zone'] = np.select(conditions, choices, default='Other')
    df = df[df['Zone'] != 'Other']

    df['Label'] = df['Zone'] + "\n" + df['Forest']

    # 排序与过滤
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧹 Filtering small groups...")
    zones_order = ['Boreal', 'Temperate', 'Tropical']
    forests_sub_order = ['ENF', 'DNF', 'MF', 'DBF', 'EBF']
    final_order = []
    palette_dict = {}

    for z in zones_order:
        for f in forests_sub_order:
            label = f"{z}\n{f}"
            count = len(df[df['Label'] == label])
            if count > MIN_PIXEL_COUNT:
                final_order.append(label)
                palette_dict[label] = COLOR_THEME[z]

    # ================= 🌟 关键修复：字体设置 🌟 =================
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎨 Plotting with fixed fonts...")

    # 1. 重置 Matplotlib 配置，清除可能导致冲突的旧设置
    mpl.rcParams.update(mpl.rcParamsDefault)

    # 2. 设置通用字体优先使用 Arial (用于英文和数字)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

    # 3. 关键：告诉 Matplotlib 使用标准的 LaTeX 数学字体引擎来渲染公式
    # 'cm' 代表 Computer Modern，是学术论文标准的数学字体
    plt.rcParams['mathtext.fontset'] = 'cm'
    # 如果您希望数学符号看起来更像无衬线字体，可以取消下面这行的注释：
    # plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # 4. 应用 Seaborn 风格 (不再强制在 sns 里设置 font="Arial")
    sns.set_theme(style="ticks", font_scale=1.1, rc={"axes.unicode_minus": False})

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    sns.boxplot(
        data=df,
        x='Label',
        y='Threshold',
        order=final_order,
        palette=palette_dict,
        showfliers=False,
        width=0.65,
        linewidth=1.2,
        boxprops=dict(alpha=0.8, edgecolor='#333333'),
        medianprops=dict(color='#222222', linewidth=1.5),
        whiskerprops=dict(color='#333333'),
        capprops=dict(color='#333333'),
        ax=ax
    )

    # Y轴标签：确保使用 r"..." 原始字符串
    ax.set_ylabel(r"Critical Root-zone Soil Moisture ($\theta_{crit}$, $m^3/m^3$)", fontsize=14)

    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.15)
    sns.despine(trim=True, offset=5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "Fig_Biome_Gradient_Fixed_Theta.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 图片已保存(修复了θ): {out_path}")


if __name__ == "__main__":
    run_biome_boxplot_final_fix_v2()
