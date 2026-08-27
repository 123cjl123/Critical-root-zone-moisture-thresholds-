import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
import seaborn as sns
import os
import warnings
import time
from datetime import datetime
import matplotlib as mpl  # 引入 mpl 以便进行底层字体设置

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 🔧 1. 参数与路径配置 =================
FILE_THRESH = r""
FILE_AI = r""
FILE_FOREST = r""

OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 森林ID映射
FOREST_MAPPING = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "MF"}
FOREST_ORDER = ["ENF", "DNF", "MF", "DBF", "EBF"]

# 🎨 配色方案
light_palette = {
    "ENF": "#88CCEE",  # 浅蓝
    "DNF": "#44AA99",  # 浅青
    "MF": "#DDCC77",  # 浅沙
    "DBF": "#CC6677",  # 浅玫瑰
    "EBF": "#AA4499"  # 浅紫
}

# 🏷️ AI 梯度标签
AI_LABELS = [
    'Semi-Arid\n(0.2 ≤ AI < 0.5)',
    'Dry Sub-Humid\n(0.5 ≤ AI < 0.65)',
    'Humid\n(AI ≥ 0.65)'
]


# ================= 🛠️ 2. 工具函数 =================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_and_process_data():
    log("🚀 开始加载 TIF 数据...")

    da_thresh = rioxarray.open_rasterio(FILE_THRESH).isel(band=0).squeeze()
    da_ai = rioxarray.open_rasterio(FILE_AI).isel(band=0).squeeze()
    da_forest = rioxarray.open_rasterio(FILE_FOREST).isel(band=0).squeeze()

    log("🔄 正在对齐栅格 (Reproject Match)...")
    da_ai = da_ai.rio.reproject_match(da_thresh)
    da_forest = da_forest.rio.reproject_match(da_thresh, resampling=0)

    log("📊 正在构建 DataFrame...")
    df = pd.DataFrame({
        'Threshold': da_thresh.values.flatten(),
        'AI': da_ai.values.flatten(),
        'ForestID': da_forest.values.flatten()
    })

    del da_thresh, da_ai, da_forest

    log("🧹 正在清洗与分类数据...")
    df = df.dropna()
    df = df[df['ForestID'].isin([1, 2, 3, 4, 5])]
    df['Forest'] = df['ForestID'].map(FOREST_MAPPING)

    # 划分 AI 梯度
    conditions = [
        (df['AI'] >= 0.2) & (df['AI'] < 0.5),
        (df['AI'] >= 0.5) & (df['AI'] < 0.65),
        (df['AI'] >= 0.65)
    ]
    df['Aridity Gradient'] = np.select(conditions, AI_LABELS, default='Other')
    df_final = df[df['Aridity Gradient'] != 'Other']

    log(f"✅ 数据准备就绪! 有效样本量: {len(df_final):,}")
    return df_final


def plot_light_boxplot(df):
    log("🎨 开始绘图 (High Quality Mode)...")

    # ================= 🌟 字体修复核心代码 🌟 =================
    # 1. 重置 Matplotlib 配置，清除冲突
    mpl.rcParams.update(mpl.rcParamsDefault)

    # 2. 设置通用字体 (Arial 用于英文)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']

    # 3. 关键：设置数学公式字体引擎为 Computer Modern (学术标准)
    plt.rcParams['mathtext.fontset'] = 'cm'
    # =========================================================

    # 应用 Seaborn 风格
    sns.set_theme(style="ticks", font_scale=1.1, rc={"axes.unicode_minus": False})

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # 绘图逻辑
    plot_kwargs = {
        'data': df,
        'x': 'Aridity Gradient',
        'y': 'Threshold',
        'hue': 'Forest',
        'hue_order': FOREST_ORDER,
        'order': AI_LABELS,
        'palette': light_palette,
        'showfliers': False,
        'width': 0.55,
        'linewidth': 1.2,
        'ax': ax
    }

    props = dict(alpha=0.6, edgecolor='#444444')

    try:
        sns.boxplot(**plot_kwargs, gap=0.15, boxprops=props)
    except TypeError:
        log("⚠️ Seaborn 版本过低不支持 gap 参数，已自动忽略")
        sns.boxplot(**plot_kwargs, boxprops=props)

    # ================= 修饰图表 =================
    ax.set_title("")
    ax.set_xlabel("")  # 移除 X 轴标题

    # 🌟 修正 Y 轴标签：使用原始字符串 r"..." 配合 LaTeX 语法
    ax.set_ylabel(r"Critical Root-zone Soil Moisture ($\theta_{crit}$, $m^3/m^3$)", fontsize=14)

    # 背景网格
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.15)
    sns.despine(trim=True, offset=10)

    # 优化图例
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        title=None,
        frameon=False,
        fontsize=11
    )

    out_file = "Fig_AI_Gradient_Fixed_Theta.png"
    out_path = os.path.join(OUTPUT_DIR, out_file)
    plt.savefig(out_path, bbox_inches='tight')
    log(f"💾 图片已保存至: {out_path}")


# ================= 🚀 主程序入口 =================
if __name__ == "__main__":
    start_time = time.time()
    try:
        df_data = load_and_process_data()
        plot_light_boxplot(df_data)
    except Exception as e:
        log(f"❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()

    end_time = time.time()
    duration = end_time - start_time
    log(f"🎉 全部完成! 总耗时: {duration:.2f} 秒")
