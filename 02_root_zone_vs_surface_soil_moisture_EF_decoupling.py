import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rioxarray
import os
import warnings
import pandas as pd
import matplotlib.ticker as ticker
from rasterio.enums import Resampling
from scipy.stats import ttest_ind  # ✨ 引入T检验计算P值

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 🔧 1. 配置区域 =================
FILE_SM_ROOT = r""
FILE_SM_SURF = r""
FILE_EF = r""
FOREST_TIF = r""
FILE_AI = r""

OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 核心阈值
Z_DRY_THRESHOLD = -0.5  # 表层极端干旱标准 (-1.5 std)
Z_SAFE_THRESHOLD = -0.5  # 深层相对安全标准 (-0.5 std)

AI_RANGES = [
    (0.2, 0.5, "Semi-Arid (AI 0.2-0.5)"),
    (0.5, 0.65, "Semi-Arid Subhumid (AI 0.5-0.65)"),
    (0.65, 10, "Humid (AI > 0.65)")
]

# ================= 🎨 2. 绘图样式 (严格与 SIF 对齐) =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

FIG_WIDTH_INCH, FIG_HEIGHT_INCH = 3.8, 3.5

# 字体大小与粗细控制拆分
SIZE_TITLE = 9
SIZE_LABEL = 8
SIZE_TICK_X = 8
SIZE_TICK_Y = 8
SIZE_TEXT = 7

WEIGHT_TITLE = 'bold'
WEIGHT_LABEL = 'bold'
WEIGHT_TICK_X = 'bold'    # 横轴分类加粗
WEIGHT_TICK_Y = 'normal'  # 纵轴数值正常


# ================= 🧰 3. 核心工具函数 =================

def standardize_da(da):
    rename_dict = {}
    dims = [str(d) for d in da.dims]

    for lon_name in ['longitude', 'lon', 'long', 'x']:
        if lon_name in dims: rename_dict[lon_name] = 'lon'
    for lat_name in ['latitude', 'lat', 'y']:
        if lat_name in dims: rename_dict[lat_name] = 'lat'
    for time_name in ['time', 't', 'date', 'valid_time']:
        if time_name in dims: rename_dict[time_name] = 'time'

    da = da.rename(rename_dict)

    if 'lat' in da.dims and 'lon' in da.dims:
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    if 'time' in da.coords:
        if not np.issubdtype(da.time.dtype, np.datetime64):
            da['time'] = pd.to_datetime(da.time.values)
        da = da.assign_coords(time=da.indexes['time'].to_period('M').to_timestamp())
    return da


def load_data(path, var_name, template=None):
    print(f"   📂 Loading {var_name}...")
    if path.lower().endswith(('.tif', '.tiff')):
        da = rioxarray.open_rasterio(path).isel(band=0, drop=True)
    else:
        ds = xr.open_dataset(path)
        da = ds[list(ds.data_vars)[0]]

    da = standardize_da(da)

    if template is not None:
        if template.rio.x_dim != 'lon' or template.rio.y_dim != 'lat':
            template = template.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

        da = da.rio.write_crs("EPSG:4326").rio.reproject_match(
            template, resampling=Resampling.nearest
        )
        da = standardize_da(da)

    return da


# ================= 🚀 4. 主程序 =================

def run_analysis():
    print("🚀 Step 1: Initializing and Aligning Data...")

    forest_raw = rioxarray.open_rasterio(FOREST_TIF).isel(band=0, drop=True)
    forest_da = standardize_da(forest_raw)

    da_ef = load_data(FILE_EF, 'EF_Anom', forest_da)
    da_root = load_data(FILE_SM_ROOT, 'SM_Root', forest_da)
    da_surf = load_data(FILE_SM_SURF, 'SM_Surf', forest_da)
    da_ai = load_data(FILE_AI, 'AI', forest_da)
    if 'time' in da_ai.dims: da_ai = da_ai.mean('time')

    print("   Syncing time coordinates...")
    common_time = np.intersect1d(da_root.time.values, da_surf.time.values)
    common_time = np.intersect1d(common_time, da_ef.time.values)

    da_root = da_root.sel(time=common_time)
    da_surf = da_surf.sel(time=common_time)
    da_ef = da_ef.sel(time=common_time)

    da_ef = da_ef.assign_coords(lat=da_root.lat, lon=da_root.lon)
    da_surf = da_surf.assign_coords(lat=da_root.lat, lon=da_root.lon)

    mask_forest = (forest_da >= 1) & (forest_da <= 5)

    print("\n🚀 Step 2: Regional Analysis Loop...")

    for (ai_min, ai_max, zone_name) in AI_RANGES:
        print(f"🌍 Processing: {zone_name} ...")

        mask_zone = mask_forest & (da_ai > ai_min) & (da_ai <= ai_max)

        v_root = da_root.where(mask_zone).values.flatten()
        v_surf = da_surf.where(mask_zone).values.flatten()
        v_ef = da_ef.where(mask_zone).values.flatten()

        valid = np.isfinite(v_root) & np.isfinite(v_surf) & np.isfinite(v_ef)
        v_root, v_surf, v_ef = v_root[valid], v_surf[valid], v_ef[valid]

        if len(v_ef) < 500:
            print(f"   ⚠️ Skipping: Insufficient samples ({len(v_ef)})")
            continue

        # Z-score 归一化
        z_root = (v_root - np.mean(v_root)) / np.std(v_root)
        z_surf = (v_surf - np.mean(v_surf)) / np.std(v_surf)
        z_ef = (v_ef - np.mean(v_ef)) / np.std(v_ef)

        # 状态判定逻辑
        mask_safe = (z_surf < Z_DRY_THRESHOLD) & (z_root > Z_SAFE_THRESHOLD)
        mask_fail = (z_surf < Z_DRY_THRESHOLD) & (z_root < Z_DRY_THRESHOLD)

        ef_safe = z_ef[mask_safe]
        ef_fail = z_ef[mask_fail]

        if len(ef_safe) < 20 or len(ef_fail) < 20:
            print("   ⚠️ Not enough extreme events.")
            continue

        # ================= 🎨 绘图 (严格一致的风格) =================
        sns.set_style("whitegrid") # 对齐 SIF
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=300)

        df_plot = pd.concat([
            pd.DataFrame({'EF': ef_safe, 'Condition': 'Surface Dry\nRoot Wet (Safe)'}),
            pd.DataFrame({'EF': ef_fail, 'Condition': 'Surface Dry\nRoot Dry (Fail)'})
        ])

        # 保留你的无情截断长尾逻辑！
        p_low = np.percentile(z_ef, 5)
        p_high = np.percentile(z_ef, 95)
        clip_limit = min(max(abs(p_low), abs(p_high)), 1.2)
        df_plot = df_plot[df_plot['EF'].between(-clip_limit, clip_limit)]

        # 对齐 SIF 的颜色
        colors = ["#7FB3D5", "#F1948A"]

        # 1. 小提琴图
        try:
            width_param = {'density_norm': 'width'}
            sns.violinplot(x='Condition', y='EF', data=df_plot, palette=colors,
                           inner=None, linewidth=0, saturation=0.85, ax=ax,
                           cut=0.6, bw_adjust=1.2, width=0.7, **width_param)
        except:
            sns.violinplot(x='Condition', y='EF', data=df_plot, palette=colors,
                           inner=None, linewidth=0, saturation=0.85, ax=ax,
                           cut=0.6, bw=1.2, width=0.7, scale='width')

        # 2. 箱线图 (补齐 SIF 的纯黑边框和 alpha)
        sns.boxplot(x='Condition', y='EF', data=df_plot, width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.8},
                    whiskerprops={'linewidth': 0.8, 'color': 'black'},
                    capprops={'linewidth': 0.8, 'color': 'black'},
                    medianprops={'color': 'black', 'linewidth': 1.2},
                    showfliers=False, zorder=10, ax=ax)

        # 3. 点图
        sns.pointplot(x='Condition', y='EF', data=df_plot, color='black',
                      join=False, scale=0.4, zorder=11, markers="D", ax=ax)

        # 4. 装饰线条
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
        plt.grid(True, axis='y', linestyle=':', alpha=0.5, color='gray') # 补齐 SIF 内部网格虚线

        # 坐标轴范围与刻度
        ax.set_ylim(-1.0, 1.0)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

        # 5. 文字与标签
        ax.set_title(zone_name.split(' (')[0], fontsize=SIZE_TITLE, fontweight=WEIGHT_TITLE, color='black', pad=10)
        ax.set_ylabel("Forest EF Anomaly (Z-score)", fontsize=SIZE_LABEL, fontweight=WEIGHT_LABEL, color='black')
        ax.set_xlabel("")

        # 6. 刻度样式 (引入 SIF 循环强制锁定)
        ax.tick_params(axis='both', labelsize=SIZE_TICK_X, color='black', width=0.8)

        # 横轴分类名加粗
        for label in ax.get_xticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(SIZE_TICK_X)
            label.set_fontweight(WEIGHT_TICK_X)
            label.set_color('black')

        # 纵轴数值正常
        for label in ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(SIZE_TICK_Y)
            label.set_fontweight(WEIGHT_TICK_Y)
            label.set_color('black')

        # 边框纯黑
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')

        # ✨ 核心计算：进行 Welch's t-test 计算 P 值
        t_stat, p_val = ttest_ind(ef_safe, ef_fail, equal_var=False)

        if p_val < 0.001:
            p_str = "P < 0.001"
        else:
            p_str = f"P = {p_val:.3f}"

        diff = np.mean(ef_safe) - np.mean(ef_fail)

        # ✨ 7. 统计信息文本框 (统一 SIF 的边框和圆角样式)
        stats_text = (f"Gap: {diff:.2f}\n"
                      # f"{p_str}\n"
                      f"Mean(Safe): {np.mean(ef_safe):.2f}\n"
                      f"Mean(Fail): {np.mean(ef_fail):.2f}")

        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
                fontsize=SIZE_TEXT, fontfamily='Arial', color='black',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3', linewidth=0.5))

        clean_name = zone_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(">", "GT")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Decoupling_Plot_Perfect_{clean_name}.png"), dpi=300)
        plt.close()

    print(f"\n🎉 分析完成！所有图表已保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_analysis()
