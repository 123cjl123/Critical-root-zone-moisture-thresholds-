import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib.colors as mcolors
import warnings
from tqdm import tqdm
import traceback

# 忽略警告信息
warnings.filterwarnings('ignore')

# ================= 🔧 1. 配置区域 =================

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# 🌟🌟🌟 字体大小独立控制 (New!) 🌟🌟🌟
FONT_SIZE_TITLE = 20  # 标题 (如果有)
FONT_SIZE_AXIS = 20  # 矩阵图的轴标签 (变量名)
FONT_SIZE_TICK = 18  # 刻度数字
FONT_SIZE_VAL = 16  # 矩阵数值
FONT_SIZE_CBAR = 20  # 颜色条刻度

# 👇👇👇 在这里单独调整散点图的横纵轴标题大小 👇👇👇
FONT_SIZE_SCATTER_X = 22  # 散点图-横轴标题大小
FONT_SIZE_SCATTER_Y = 22  # 散点图-纵轴标题大小

# 布局参数
LAYOUT_TOP = 0.95
LAYOUT_BOTTOM = 0.20
LAYOUT_LEFT = 0.20
LAYOUT_RIGHT = 0.85
CBAR_PAD = 0.01
CBAR_WIDTH = 0.03

CSV_FILE = r""
OUTPUT_ROOT = r""  # 修改输出路径

if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

MAX_SAMPLES = 20000

# 变量名映射
VAR_MAPPING = {
    'Threshold': 'Threshold',
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
    'BulkDensity': 'BulkDensity'
}

FEATURE_COLS = [
    'Precipitation', 'SMrz',
    'T2m', 'VPD', 'Radiation',
    'LAI', 'TreeHeight',
    'Sand', 'Clay', 'CFVO', 'BulkDensity'
]

AI_ZONES = {1: "Semi-Arid", 2: "Dry_Sub-Humid", 3: "Humid"}
FOREST_TYPES = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "Mixed"}

custom_reds = mcolors.LinearSegmentedColormap.from_list("pearson_style_red", ["#FFFFFF", "#F8766D"], N=100)


# ================= 🤖 2. 核心分析函数 =================

def run_shap_interaction(df_subset, save_folder, title_suffix):
    X = df_subset[FEATURE_COLS]
    y = df_subset['Threshold']

    if not os.path.exists(save_folder): os.makedirs(save_folder)

    with tqdm(total=4, desc=f"   🔨 处理中: {title_suffix}", leave=False, unit="step") as pbar:

        # --- 步骤 1: 训练 ---
        pbar.set_description(f"   🔥 训练模型...")
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
        model.fit(X, y)

        if len(X) > MAX_SAMPLES:
            X_shap = X.sample(MAX_SAMPLES, random_state=42)
        else:
            X_shap = X

        explainer = shap.TreeExplainer(model)
        shap_interaction_values = explainer.shap_interaction_values(X_shap)
        shap_values = explainer.shap_values(X_shap)
        pbar.update(1)

        global_importance = np.abs(shap_values).mean(0)
        importance_dict = {col: imp for col, imp in zip(X.columns, global_importance)}

        # --- 步骤 2: 绘制矩阵图 (使用 FONT_SIZE_AXIS) ---
        pbar.set_description(f"   🎨 绘制矩阵图...")

        mean_interaction = np.abs(shap_interaction_values).mean(0)
        tmp_interaction = mean_interaction.copy()
        np.fill_diagonal(tmp_interaction, np.nan)

        short_cols = [VAR_MAPPING.get(c, c) for c in X.columns]
        df_inter = pd.DataFrame(tmp_interaction, index=short_cols, columns=short_cols)
        max_val = np.nanmax(tmp_interaction)
        df_norm = df_inter / max_val if max_val > 0 else df_inter

        fig, ax = plt.subplots(figsize=(18, 16))
        fig.subplots_adjust(top=LAYOUT_TOP, bottom=LAYOUT_BOTTOM, left=LAYOUT_LEFT, right=LAYOUT_RIGHT)

        sns.heatmap(df_norm, ax=ax, cbar=False, cmap=custom_reds, annot=True, fmt=".2f", vmin=0, vmax=1,
                    square=True, linewidths=1.5, linecolor='white',
                    annot_kws={"size": FONT_SIZE_VAL, "weight": "bold", "color": "black"})

        cbar_ax = fig.add_axes([LAYOUT_RIGHT + CBAR_PAD, LAYOUT_BOTTOM, CBAR_WIDTH, LAYOUT_TOP - LAYOUT_BOTTOM])
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=custom_reds, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("")

        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])
        cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR)

        ax.set_xticklabels(short_cols, rotation=45, ha='right', rotation_mode='anchor',
                           fontsize=FONT_SIZE_AXIS, fontweight='bold')
        ax.set_yticklabels(short_cols, rotation=0, ha='right',
                           fontsize=FONT_SIZE_AXIS, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.savefig(os.path.join(save_folder, "1_Matrix_Interaction.jpg"), dpi=300, bbox_inches='tight')
        plt.close()
        pbar.update(1)

        # --- 步骤 3: 绘制散点图 (使用独立变量控制横纵轴字体) ---
        pbar.set_description(f"   🎨 绘制散点图...")

        indices = np.dstack(np.unravel_index(np.argsort(tmp_interaction.ravel())[::-1], tmp_interaction.shape))[0]
        top_pairs = []
        seen = set()

        for r, c in indices:
            if np.isnan(tmp_interaction[r, c]): continue
            pair_set = frozenset([FEATURE_COLS[r], FEATURE_COLS[c]])

            if pair_set not in seen:
                seen.add(pair_set)
                raw_feat_a, raw_feat_b = FEATURE_COLS[r], FEATURE_COLS[c]

                if raw_feat_a == 'SMrz':
                    main_axis_feat, color_feat = raw_feat_b, raw_feat_a
                elif raw_feat_b == 'SMrz':
                    main_axis_feat, color_feat = raw_feat_a, raw_feat_b
                else:
                    if importance_dict[raw_feat_a] >= importance_dict[raw_feat_b]:
                        main_axis_feat, color_feat = raw_feat_a, raw_feat_b
                    else:
                        main_axis_feat, color_feat = raw_feat_b, raw_feat_a

                top_pairs.append((main_axis_feat, color_feat))

            if len(top_pairs) >= 3: break

        for rank, (main_feat, interact_feat) in enumerate(top_pairs):
            try:
                fig, ax = plt.subplots(figsize=(10, 8))

                shap.dependence_plot(
                    main_feat, shap_values, X_shap, interaction_index=interact_feat,
                    show=False, ax=ax, cmap=plt.get_cmap("coolwarm"), alpha=0.7, dot_size=40
                )

                short_main = VAR_MAPPING.get(main_feat, main_feat)
                short_interact = VAR_MAPPING.get(interact_feat, interact_feat)

                ax.set_title("")

                # 🌟🌟🌟 独立设置横纵轴字体大小 🌟🌟🌟
                ax.set_xlabel(short_main, fontsize=FONT_SIZE_SCATTER_X, fontweight='bold', fontname='Arial')
                ax.set_ylabel(f"SHAP value for {short_main}", fontsize=FONT_SIZE_SCATTER_Y, fontweight='bold',
                              fontname='Arial')

                ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK, width=2)

                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname('Arial')
                    label.set_fontsize(FONT_SIZE_TICK)
                    label.set_fontweight('bold')

                if len(fig.axes) > 1:
                    cbar_ax_scatter = fig.axes[1]
                    # 颜色条的标签大小也可以复用纵轴的大小，或者单独设置
                    cbar_ax_scatter.set_ylabel(short_interact, fontsize=FONT_SIZE_SCATTER_Y, fontweight='bold',
                                               fontname='Arial')
                    cbar_ax_scatter.tick_params(labelsize=FONT_SIZE_CBAR, width=2)
                    for label in cbar_ax_scatter.get_yticklabels():
                        label.set_fontname('Arial')
                        label.set_fontweight('bold')

                plt.tight_layout()
                plt.savefig(os.path.join(save_folder, f"2_Scatter_Top{rank + 1}_{short_main}_vs_{short_interact}.jpg"),
                            dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                pass

        pbar.update(2)


# ================= 🔄 3. 主程序 =================

if __name__ == "__main__":
    print("步骤 1: 读取数据...")
    if not os.path.exists(CSV_FILE):
        print("❌ 错误: 未找到 CSV 文件")
        exit()
    df = pd.read_csv(CSV_FILE)

    print("\n🌍 全球数据...")
    try:
        run_shap_interaction(df, os.path.join(OUTPUT_ROOT, "Global"), "Global")
    except Exception as e:
        print(f"❌ Global 失败: {e}")
        traceback.print_exc()

    for z_id, z_name in AI_ZONES.items():
        print(f"\n📍 区域: {z_name}...")
        try:
            df_zone = df[df['AI_Zone'] == z_id]
            if len(df_zone) < 50: continue

            safe_z = z_name.replace(" ", "_")
            run_shap_interaction(df_zone, os.path.join(OUTPUT_ROOT, f"Zone_{z_id}_{safe_z}"), f"{z_name} (Whole)")

            for ft_id, ft_name in FOREST_TYPES.items():
                try:
                    df_sub = df_zone[df_zone['ForestType'] == ft_id]
                    if len(df_sub) < 50: continue

                    safe_ft = ft_name.replace(" ", "_")
                    run_shap_interaction(df_sub,
                                         os.path.join(OUTPUT_ROOT, f"Zone_{z_id}_{safe_z}_FT_{ft_id}_{safe_ft}"),
                                         f"{z_name} - {ft_name}")
                except Exception as e:
                    print(f"      ❌ 子类 ({ft_name}) 失败: {e}")
                    continue

        except Exception as e:
            print(f"   ❌ 区域 ({z_name}) 失败: {e}")
            continue

    print(f"\n🎉 完成！输出路径: {OUTPUT_ROOT}")
