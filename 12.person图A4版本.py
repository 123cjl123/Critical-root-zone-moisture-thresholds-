import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import warnings
import traceback

# 忽略警告信息，保持输出清洁
warnings.filterwarnings('ignore')

# ================= 🔧 1. 基础配置与路径 =================

# 设置全局字体为 Arial，并解决负号显示问题
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 🌟🌟🌟 [修改1] 强制全局纯黑设置 🌟🌟🌟
# 虽然会被下面的 sns.set_style 覆盖，但保留作为保险
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'

# 输入文件路径
CSV_FILE = r"D:\Paper_1\paper_all\shuju_outout\xunlian1.csv"
# 输出文件夹路径
OUTPUT_ROOT = r"D:\Paper_1\paper_all\person_plot_15cm_Final_Black"  # 修改一下路径以免覆盖

# 如果输出目录不存在，自动创建
if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

# 字体大小配置 (适配 15cm)
FONT_SIZE_AXIS = 10
FONT_SIZE_NUM = 6
FONT_SIZE_TICK = 4.5
FONT_SIZE_CBAR_TICK = 9

# 布局对齐参数
LAYOUT_TOP = 0.95
LAYOUT_BOTTOM = 0.15
LAYOUT_LEFT = 0.20
LAYOUT_RIGHT = 0.88
CBAR_PAD = 0.015
CBAR_WIDTH = 0.025

# 变量名映射
VAR_MAPPING = {
    'Precipitation': 'Precipitation',
    'SMrz': 'SMrz',
    'T2m': 'T2m',
    'VPD': 'VPD',
    'Radiation': 'Radiation',
    'LAI': 'LAI',
    'TreeHeight': 'TreeHeight',
    'Sand': 'Sand',
    'Clay': 'Clay',
    'CFVO': 'CFVO',
    'BulkDensity': 'BulkDensity',
    'Threshold': 'Threshold'
}

PLOT_COLS = [
    'Precipitation', 'SMrz', 'T2m', 'VPD', 'Radiation',
    'LAI', 'TreeHeight',
    'Sand', 'Clay', 'CFVO', 'BulkDensity',
    'Threshold'
]

AI_ZONES = {1: "Semi-Arid", 2: "Dry_Sub-Humid", 3: "Humid"}
FOREST_TYPES = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "Mixed"}

# ================= 🎨 2. 自定义绘图函数 =================

cmap_corr = mcolors.LinearSegmentedColormap.from_list("custom_div", ["#7CAE00", "#FFFFFF", "#F8766D"], N=100)
norm = plt.Normalize(-1, 1)


def corr_func(x, y, **kws):
    """
    绘制上三角
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2: return

    r, p = stats.pearsonr(x, y)

    p_stars = ''
    if p < 0.05: p_stars = '*'
    if p < 0.01: p_stars = '**'
    if p < 0.001: p_stars = '***'

    ax = plt.gca()
    ax.set_facecolor(cmap_corr(norm(r)))

    # 🌟🌟🌟 [修改2] 矩阵内部文字颜色逻辑 🌟🌟🌟
    # 只有背景非常深时才用白色，其他情况一律用纯黑 #000000，不用 'black'
    text_color = 'white' if abs(r) > 0.6 else '#000000'

    ax.text(0.5, 0.55, f"{r:.2f}", ha='center', va='center', transform=ax.transAxes,
            fontsize=FONT_SIZE_NUM, fontweight='normal', color=text_color)

    ax.text(0.5, 0.25, p_stars, ha='center', va='center', transform=ax.transAxes,
            fontsize=FONT_SIZE_NUM + 1, fontweight='normal', color=text_color)
    ax.grid(False)


def scatter_func(x, y, **kws):
    """
    绘制下三角
    """
    sample_n = min(len(x), 2000)
    idx = np.random.choice(len(x), sample_n, replace=False)
    x_plot, y_plot = x.iloc[idx], y.iloc[idx]

    sns.regplot(x=x_plot, y=y_plot, ax=plt.gca(),
                scatter_kws={'s': 1.5, 'alpha': 0.4, 'color': '#E68FAC', 'edgecolor': 'none'},
                line_kws={'color': '#C71585', 'linewidth': 0.8})


def diag_func(x, **kws):
    """
    绘制对角线
    """
    # 🌟 这里的 edgecolor 虽然是直方图边缘，但也设为纯黑可能太硬，保持 white 或设为 #000000 看你喜好
    # 这里我们只改文字，图形边缘保持原样
    sns.histplot(x=x, kde=True, ax=plt.gca(), color='#8FBC8F', edgecolor='white', linewidth=0.3)


# ================= 🚀 3. 核心绘图逻辑 =================

def plot_correlation_matrix(df_subset, filename, global_ranges):
    print(f"      -> 正在绘图... {filename} (数据量 n={len(df_subset)})")

    valid_cols = [c for c in PLOT_COLS if c in df_subset.columns]
    df_plot = df_subset[valid_cols].dropna(axis=1, how='all').rename(columns=VAR_MAPPING)

    if df_plot.shape[1] < 2: return

    # 🌟🌟🌟 [修改3] 强制 Seaborn 使用纯黑配置 🌟🌟🌟
    # 这一步最关键，因为 sns.set_style 默认会把 rcParams 里的颜色重置为深灰
    sns.set_style("white", rc={
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.edgecolor": "#000000"  # 如果你想让图的边框也是纯黑
    })

    # 建立 PairGrid 网格
    g = sns.PairGrid(df_plot, diag_sharey=False, height=0.8, aspect=1)

    g.map_upper(corr_func)
    g.map_lower(scatter_func)
    g.map_diag(diag_func)

    mapped_cols = list(df_plot.columns)

    # --- 刻度与极值控制 ---
    for i, ax_row in enumerate(g.axes):
        for j, ax in enumerate(ax_row):
            if ax:
                x_var, y_var = mapped_cols[j], mapped_cols[i]

                # 设置 X 轴
                if x_var in global_ranges:
                    xmin, xmax = global_ranges[x_var]
                    ax.set_xlim(xmin, xmax)
                    ax.set_xticks([xmin, xmax])
                    xlabels = ax.get_xticklabels()
                    if len(xlabels) >= 2:
                        xlabels[0].set_ha('left')
                        xlabels[-1].set_ha('right')
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                # 设置 Y 轴
                need_y_format = False
                if i != j and y_var in global_ranges:
                    ymin, ymax = global_ranges[y_var]
                    ax.set_ylim(ymin, ymax)
                    ax.set_yticks([ymin, ymax])
                    need_y_format = True
                elif i == j and j == 0:
                    ymin, ymax = ax.get_ylim()
                    ax.set_yticks([ymin, ymax])
                    need_y_format = True

                if need_y_format:
                    ylabels = ax.get_yticklabels()
                    if len(ylabels) >= 2:
                        plt.setp(ylabels, rotation=90, fontsize=FONT_SIZE_TICK)
                        ylabels[0].set_va('bottom')
                        ylabels[-1].set_va('top')
                        plt.setp(ylabels, ha='center')
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                # 设置刻度参数 (确保颜色为纯黑)
                # 🌟 [修改4] color='#000000'
                ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK, pad=1, colors='#000000')

                # 确保边框也是纯黑 (可选)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#000000')

    # --- 变量标签样式 ---
    for ax in g.axes[:, 0]:
        ylabel = ax.get_ylabel()
        if ylabel:
            # 🌟 [修改5] color='#000000'
            ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXIS,
                          fontweight='bold', color='#000000',
                          rotation=0, ha='right', va='center', labelpad=10)

    for ax in g.axes[-1, :]:
        xlabel = ax.get_xlabel()
        if xlabel:
            # 🌟 [修改6] color='#000000'
            ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXIS,
                          fontweight='bold', color='#000000',
                          rotation=45, ha='right', rotation_mode='anchor', labelpad=5)

    # --- 调整整体布局 ---
    g.fig.subplots_adjust(top=LAYOUT_TOP, bottom=LAYOUT_BOTTOM, left=LAYOUT_LEFT, right=LAYOUT_RIGHT, hspace=0.1,
                          wspace=0.1)

    # --- 颜色条设置 ---
    cbar_ax = g.fig.add_axes([LAYOUT_RIGHT + CBAR_PAD, LAYOUT_BOTTOM, CBAR_WIDTH, LAYOUT_TOP - LAYOUT_BOTTOM])
    sm = plt.cm.ScalarMappable(cmap=cmap_corr, norm=norm)
    sm.set_array([])

    cbar = g.fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # 🌟 [修改7] 颜色条刻度纯黑
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR_TICK, colors='#000000')

    cbar.outline.set_visible(False)

    # --- 强制锁定物理尺寸 (15cm) ---
    g.fig.set_size_inches(5.9, 5.9)

    save_path = os.path.join(OUTPUT_ROOT, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ================= 🔄 4. 主程序 =================

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print("❌ 错误：找不到 CSV 文件")
        exit()
    df = pd.read_csv(CSV_FILE)

    global_ranges = {}
    for col in PLOT_COLS:
        if col in df.columns:
            short_name = VAR_MAPPING.get(col, col)
            global_ranges[short_name] = (df[col].min(), df[col].max())

    print("\n🌍 正在生成：全球总图 (Global)...")
    try:
        plot_correlation_matrix(df, "Corr_Global_15cm.jpg", global_ranges)
    except Exception as e:
        print(f"❌ Global 图生成失败: {e}")
        traceback.print_exc()

    for z_id, z_name in AI_ZONES.items():
        print(f"\n📍 正在处理气候区: {z_name}...")
        try:
            df_zone = df[df['AI_Zone'] == z_id]
            if len(df_zone) < 50:
                print(f"   ⚠️ 数据不足 (<50)，跳过 {z_name}")
                continue

            safe_z = z_name.replace(" ", "_")
            plot_correlation_matrix(df_zone, f"Corr_Zone_{z_id}_{safe_z}_All.jpg", global_ranges)

            for ft_id, ft_name in FOREST_TYPES.items():
                try:
                    df_sub = df_zone[df_zone['ForestType'] == ft_id]
                    if len(df_sub) < 30: continue

                    safe_ft = ft_name.replace(" ", "_")
                    plot_correlation_matrix(df_sub, f"Corr_Zone_{z_id}_{safe_z}_FT_{ft_id}_{safe_ft}.jpg",
                                            global_ranges)
                except Exception as e:
                    print(f"      ❌ 子类型 ({ft_name}) 失败: {e}")
                    continue

        except Exception as e:
            print(f"   ❌ 区域 ({z_name}) 失败: {e}")
            continue

    print(f"\n🎉 所有绘图任务完成！\n📂 输出路径: {OUTPUT_ROOT}")