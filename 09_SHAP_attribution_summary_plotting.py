import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import os
import shutil
import joblib
import warnings

warnings.filterwarnings('ignore')

# ================= 🔧 1. 基础配置 =================

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ================= 🎨 2. 字体与样式配置 =================

# 字体大小常量
FONT_SIZE_TITLE = 24  # 标题
FONT_SIZE_AXIS = 22  # 变量名 (Y轴)
FONT_SIZE_TICK = 18  # 刻度数字 (XY轴)
FONT_SIZE_VAL = 16  # 条形图数值
FONT_SIZE_CBAR_TICK = 20  # 颜色条刻度
FONT_SIZE_ROSE = 14  # 玫瑰图标签

# 变量名样式
UNIFIED_FONT = {
    'fontname': 'Arial',
    'fontsize': FONT_SIZE_AXIS,
    'fontweight': 'bold',
    'color': 'black'
}

# 轴标题样式
TITLE_FONT = {
    'fontname': 'Arial',
    'fontsize': FONT_SIZE_TITLE,
    'fontweight': 'bold'
}

# ================= 📁 3. 路径配置 =================

PKL_ROOT = r""
OUTPUT_ROOT = r""  # 修改输出路径

if os.path.exists(OUTPUT_ROOT): shutil.rmtree(OUTPUT_ROOT)
os.makedirs(OUTPUT_ROOT)

# ================= 🎨 4. 配色方案 =================

PASTEL_COLORS_RAW = ["#F7A6AC", "#F7B2C7", "#F3BBB1", "#EEC78A", "#EEE9A2", "#CBE4B1", "#B3DDCB", "#B8E5FA"]
PASTEL_COLORS_CYCLE = PASTEL_COLORS_RAW * 3
pastel_div_colors = ["#B8E5FA", "#F7A6AC"]
PASTEL_DIV_CMAP = LinearSegmentedColormap.from_list("pastel_div", pastel_div_colors, N=100)


# ================= 🖌️ 5. 绘图函数 =================

def plot_combined_pack(data_pack, save_dir):
    shap_values_obj = data_pack["shap_values"]
    X = data_pack["X"]

    # --- 1. 数据处理 ---
    shap_values_matrix = shap_values_obj.values
    feature_names = X.columns.tolist()
    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)

    sorted_features = shap_series.index.tolist()
    sorted_shap_values = shap_series.values
    plot_colors = PASTEL_COLORS_CYCLE[:len(sorted_features)]

    # --- 2. 创建画布 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [1.3, 1]})
    plt.subplots_adjust(wspace=0.4)

    # =========================================================
    # 🎨 左图 (ax1): 蜂巢图
    # =========================================================
    plt.sca(ax1)

    X_sorted = X[sorted_features]
    indices = [feature_names.index(f) for f in sorted_features]
    shap_values_sorted = shap_values_matrix[:, indices]

    shap.summary_plot(
        shap_values_sorted,
        X_sorted,
        plot_type='dot',
        show=False,
        color_bar=False,
        plot_size=None,
        cmap=PASTEL_DIV_CMAP
    )

    # 强制对称 X 轴
    max_val = np.abs(shap_values_sorted).max()
    limit = max_val * 1.1
    ax1.set_xlim(-limit, limit)

    # 🌟 颜色条设置 🌟
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=PASTEL_DIV_CMAP, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax1, aspect=25, pad=0.02)
    cbar.set_label("")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.outline.set_visible(False)

    # 颜色条刻度数字：加粗 + 纯黑
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR_TICK, width=0)  # width=0 不显示刻度线
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
        label.set_color('black')  # 🌟 强制纯黑

    # 🌟 左图 X轴设置 (SHAP默认是灰色的，这里强制改黑) 🌟
    ax1.set_xlabel('SHAP value (impact on model output)', fontdict=TITLE_FONT)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK, colors='black')  # 强制 tick 颜色

    for label in ax1.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
        label.set_color('black')  # 🌟 强制纯黑，解决"暗"的问题
        label.set_alpha(1)  # 确保不透明

    # 左图 Y轴设置
    ax1.set_yticks(range(len(sorted_features)))
    ax1.set_yticklabels(sorted_features[::-1], **UNIFIED_FONT)
    ax1.tick_params(axis='y', width=0)  # 隐藏Y轴刻度线

    # =========================================================
    # 📊 右图 (ax2): 条形图
    # =========================================================
    y_pos = range(len(sorted_features))
    ax2.barh(y_pos, sorted_shap_values, color=plot_colors, align='center', height=0.6)

    # Y轴
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_features, **UNIFIED_FONT)
    ax2.invert_yaxis()

    # 右图 X轴设置
    ax2.set_xlabel('Mean |SHAP Value|', fontdict=TITLE_FONT)
    ax2.tick_params(axis='x', labelsize=FONT_SIZE_TICK, colors='black')

    for label in ax2.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
        label.set_color('black')  # 🌟 保持一致

    ax2.tick_params(axis='y', length=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    limit_x = max(sorted_shap_values) * 1.2
    ax2.set_xlim(0, limit_x)

    # 数值标签
    for i, v in enumerate(sorted_shap_values):
        ax2.text(v + (limit_x * 0.01), i, f'{v:.4f}',
                 color='black', va='center',
                 fontdict={'family': 'Arial', 'size': FONT_SIZE_VAL, 'weight': 'bold'})

    # =========================================================
    # 🌹 右图嵌入: 玫瑰图
    # =========================================================
    percentages = (sorted_shap_values / sorted_shap_values.sum()) * 100
    base_length, fixed_increment, colored_ring_width = 4.0, 0.5, 2.0
    widths = (sorted_shap_values / sorted_shap_values.sum()) * 2 * np.pi
    thetas = np.cumsum([0] + widths[:-1].tolist()) - np.pi / 21

    num_vars = len(feature_names)
    total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
    inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
    inner_colors = ['#F5F5F5', '#FFFFFF'] * (num_vars // 2 + 1)

    ax_rose = ax2.inset_axes([0.35, 0.15, 0.55, 0.55], projection='polar')
    ax_rose.patch.set_alpha(0)

    ax_rose.bar(x=thetas, height=inner_heights, width=widths, color=inner_colors, align='edge', edgecolor='white')
    ax_rose.bar(x=thetas, height=[colored_ring_width] * num_vars, width=widths, bottom=inner_heights,
                color=plot_colors, align='edge', edgecolor='white', linewidth=1)

    for i in range(num_vars):
        label_angle_rad = thetas[i] + widths[i] / 2
        label_radius = total_lengths[i] + 2.0
        ax_rose.text(label_angle_rad, label_radius, f'{percentages[i]:.1f}%',
                     ha='center', va='center',
                     fontdict={'family': 'Arial', 'weight': 'bold', 'size': FONT_SIZE_ROSE},
                     bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8))

    ax_rose.set_axis_off()

    # =========================================================
    # 💥 终极保险：再次遍历 Y 轴标签强制设置纯黑 💥
    # =========================================================
    for label in ax1.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(FONT_SIZE_AXIS)
        label.set_fontweight('bold')
        label.set_color('black')

    for label in ax2.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(FONT_SIZE_AXIS)
        label.set_fontweight('bold')
        label.set_color('black')

    plt.savefig(os.path.join(save_dir, "Combined_Feature_Analysis_PureBlackLabels.jpg"), bbox_inches='tight')
    plt.close(fig)


# ================= 🚀 主程序 =================
if __name__ == "__main__":
    print(f"🔍 Scanning for .pkl files in: {PKL_ROOT}")

    for root, dirs, files in os.walk(PKL_ROOT):
        for file in files:
            if file.endswith(".pkl"):
                pkl_path = os.path.join(root, file)
                print(f"\n🎨 Plotting: {file} ...")
                try:
                    data_pack = joblib.load(pkl_path)

                    rel_path = os.path.relpath(root, PKL_ROOT)
                    sub_name = file.replace(".pkl", "")
                    save_dir = os.path.join(OUTPUT_ROOT, rel_path, sub_name)
                    if not os.path.exists(save_dir): os.makedirs(save_dir)

                    plot_combined_pack(data_pack, save_dir)

                except Exception as e:
                    print(f"❌ Error plotting {file}: {e}")

    print(f"\n🎉 All plots generated in: {OUTPUT_ROOT}")
