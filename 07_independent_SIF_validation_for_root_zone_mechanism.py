import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rioxarray
import os
import glob
import warnings
import pandas as pd
import matplotlib.ticker as ticker
from datetime import datetime
from rasterio.enums import Resampling
from scipy.stats import ttest_ind  # ✨ 引入T检验计算P值

warnings.filterwarnings('ignore')

# ================= 🔧 1. 配置区域 =================

# SIF 数据文件夹 (重采样后的 0.25 度数据)
DIR_SIF_TIFS = r""

FILE_SM_ROOT = r""
FILE_SM_SURF = r""
FOREST_TIF = r""
FILE_AI = r""

OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_DRY_THRESHOLD = -0.5
Z_SAFE_THRESHOLD = -0.5

AI_RANGES = [
    (0.2, 0.5, "Semi-Arid (AI 0.2-0.5)"),
    (0.5, 0.65, "Semi-Arid Subhumid (AI 0.5-0.65)"),
    (0.65, 10, "Humid (AI > 0.65)")
]

# ================= 🎨 2. 绘图样式独立控制区 (完全保留您的原版) =================

# 1. 字体全局设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# 2. 尺寸控制
FIG_WIDTH_INCH = 3.8
FIG_HEIGHT_INCH = 3.5

# 3. 字体大小控制
SIZE_TITLE = 9
SIZE_LABEL = 8
SIZE_TICK_X = 8
SIZE_TICK_Y = 8
SIZE_TEXT = 7

# 4. 字体粗细控制
WEIGHT_TITLE = 'bold'
WEIGHT_LABEL = 'bold'
WEIGHT_TICK_X = 'bold'  # 横轴分类加粗
WEIGHT_TICK_Y = 'normal'  # 纵轴数值正常


# ================= 🧰 3. 数据处理函数 =================

def load_sif_series(folder_path, master_template):
    print(f"   📂 Loading SIF TIFs from: {folder_path}...")
    tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

    if not tif_files:
        print("      ❌ No .tif files found!")
        return None

    timestamps = []
    valid_files = []

    for f in tif_files:
        try:
            date_str = os.path.basename(f).split('_')[-1].split('.')[0]
            dt = pd.to_datetime(date_str, format='%Y%m')
            timestamps.append(dt)
            valid_files.append(f)
        except:
            continue

    if not valid_files: return None

    # 初始化
    first_da = rioxarray.open_rasterio(valid_files[0]).isel(band=0, drop=True)
    first_da = first_da.rio.reproject_match(master_template, resampling=Resampling.nearest)

    nt, ny, nx = len(valid_files), first_da.shape[0], first_da.shape[1]
    data_cube = np.full((nt, ny, nx), np.nan, dtype='float32')

    print(f"      ⏳ Merging {nt} TIFs...")
    for i, f in enumerate(valid_files):
        try:
            da = rioxarray.open_rasterio(f).isel(band=0, drop=True)
            reprojected = da.rio.reproject_match(master_template, resampling=Resampling.nearest)
            data_cube[i, :, :] = reprojected.values
        except:
            pass

    da_sif = xr.DataArray(
        data_cube,
        coords={'time': timestamps, 'y': master_template.y, 'x': master_template.x},
        dims=['time', 'y', 'x'],
        name='SIF'
    )
    return da_sif


def load_and_align_nc(path, var_name, master_template):
    try:
        ds = xr.open_dataset(path)
        var = list(ds.data_vars)[0]
        da = ds[var]

        rename_dict = {}
        if 'longitude' in da.dims: rename_dict['longitude'] = 'x'
        if 'latitude' in da.dims: rename_dict['latitude'] = 'y'
        if rename_dict: da = da.rename(rename_dict)
        if 'band' in da.dims: da = da.squeeze('band', drop=True)

        da.rio.write_crs("EPSG:4326", inplace=True)
        da = da.rio.reproject_match(master_template, resampling=Resampling.nearest)
        return da
    except Exception as e:
        print(f"      ❌ Error loading {var_name}: {e}")
        return None


# ================= 🚀 4. 主程序 =================

def run_sif_validation():
    print("🚀 Step 1: Loading Template & SIF...")

    try:
        forest_da = rioxarray.open_rasterio(FOREST_TIF).isel(band=0).squeeze()
        if 'longitude' in forest_da.dims: forest_da = forest_da.rename({'longitude': 'x', 'latitude': 'y'})
    except:
        return

    # 加载 SIF 并计算异常
    da_sif_raw = load_sif_series(DIR_SIF_TIFS, forest_da)
    if da_sif_raw is None or da_sif_raw.isnull().all(): return

    # ✨ 核心修复 1：使用正确的 Xarray 广播机制计算去季节化 Z-score
    print("   🔄 Calculating SIF Anomalies...")
    clim_mean = da_sif_raw.groupby('time.month').mean('time')
    clim_std = da_sif_raw.groupby('time.month').std('time')
    clim_std = clim_std.where(clim_std > 1e-6)

    da_sif_anom = (da_sif_raw.groupby('time.month') - clim_mean).groupby('time.month') / clim_std

    # 加载其他变量
    da_root = load_and_align_nc(FILE_SM_ROOT, 'SM_Root', forest_da)
    da_surf = load_and_align_nc(FILE_SM_SURF, 'SM_Surf', forest_da)
    da_ai = load_and_align_nc(FILE_AI, 'AI', forest_da)

    if da_root is None or da_surf is None: return
    if 'time' in da_ai.dims: da_ai = da_ai.mean('time')

    # 时间对齐
    t_common = np.intersect1d(pd.to_datetime(da_sif_anom.time.values), pd.to_datetime(da_root.time.values))
    if len(t_common) == 0: return

    da_sif_anom = da_sif_anom.sel(time=t_common)
    da_root = da_root.sel(time=t_common)
    da_surf = da_surf.sel(time=t_common)

    mask_forest = (forest_da >= 1) & (forest_da <= 5)

    print("\n🚀 Step 3: Analysis Loop...")
    for (ai_min, ai_max, zone_name) in AI_RANGES:
        print(f"\n🌍 Analyzing Zone: {zone_name} ...")

        mask_zone = mask_forest & (da_ai > ai_min) & (da_ai <= ai_max)

        # 数据提取与过滤
        v_root = da_root.where(mask_zone).values.flatten()
        v_surf = da_surf.where(mask_zone).values.flatten()
        v_sif = da_sif_anom.where(mask_zone).values.flatten()

        valid = np.isfinite(v_root) & np.isfinite(v_surf) & np.isfinite(v_sif)
        v_root, v_surf, v_sif = v_root[valid], v_surf[valid], v_sif[valid]

        if len(v_sif) < 1000: continue
        if len(v_sif) > 500000:
            idx = np.random.choice(len(v_sif), 500000, replace=False)
            v_root, v_surf, v_sif = v_root[idx], v_surf[idx], v_sif[idx]

        # ✨ 核心修复 2：根区和表层仍需标准化，但 SIF 已经是真实的异常值了，直接使用！
        z_root = (v_root - np.mean(v_root)) / np.std(v_root)
        z_surf = (v_surf - np.mean(v_surf)) / np.std(v_surf)
        z_sif = v_sif  # 绝对不能再做全局 (x - mean)/std ！

        mask_safe = (z_surf < Z_DRY_THRESHOLD) & (z_root > Z_SAFE_THRESHOLD)
        mask_fail = (z_surf < Z_DRY_THRESHOLD) & (z_root < Z_DRY_THRESHOLD)

        sif_safe = z_sif[mask_safe]
        sif_fail = z_sif[mask_fail]

        if len(sif_safe) < 10 or len(sif_fail) < 10: continue

        # ================= 🎨 绘图 (严格一致的风格) =================
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), dpi=300)

        df_plot = pd.concat([
            pd.DataFrame({'SIF': sif_safe, 'Condition': 'Surface Dry\nRoot Wet (Safe)'}),
            pd.DataFrame({'SIF': sif_fail, 'Condition': 'Surface Dry\nRoot Dry (Fail)'})
        ])
        df_plot = df_plot[df_plot['SIF'].between(-4, 4)]

        colors = ["#7FB3D5", "#F1948A"]

        # 1. 小提琴图
        sns.violinplot(x='Condition', y='SIF', data=df_plot, palette=colors,
                       inner=None, linewidth=0, saturation=0.85, width=0.7, ax=ax)

        # 2. 箱线图 (纯黑边框)
        sns.boxplot(x='Condition', y='SIF', data=df_plot, width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 0.8, 'alpha': 0.8},
                    whiskerprops={'linewidth': 0.8, 'color': 'black'},
                    capprops={'linewidth': 0.8, 'color': 'black'},
                    medianprops={'color': 'black', 'linewidth': 1.2},
                    showfliers=False, zorder=10, ax=ax)

        # 3. 点图 (纯黑)
        sns.pointplot(x='Condition', y='SIF', data=df_plot, color='black',
                      join=False, scale=0.4, zorder=11, markers="D", ax=ax)

        # 4. 装饰 (纯黑)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        plt.grid(True, axis='y', linestyle=':', alpha=0.5, color='gray')

        # 5. 文字与标签
        # 标题
        plt.title(f"{zone_name.split('(')[0]}",
                  fontsize=SIZE_TITLE, fontweight=WEIGHT_TITLE, color='black', pad=10)

        # Y轴变量名 (SIF Anomaly)
        plt.ylabel("SIF Anomaly (Z-score)",
                   fontsize=SIZE_LABEL, fontweight=WEIGHT_LABEL, color='black')
        plt.xlabel("")

        # 6. 刻度样式
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
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

        # 7. 统计信息 (✨ 新增 P 值计算)
        t_stat, p_val = ttest_ind(sif_safe, sif_fail, equal_var=False)
        p_str = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.3f}"

        diff = np.mean(sif_safe) - np.mean(sif_fail)
        msg = (f"Gap: {diff:.2f}\n"
               # f"{p_str}\n"
               f"Mean(Safe): {np.mean(sif_safe):.2f}\n"
               f"Mean(Fail): {np.mean(sif_fail):.2f}")

        plt.text(0.97, 0.95, msg, transform=ax.transAxes,
                 ha='right', va='top', fontsize=SIZE_TEXT, fontfamily='Arial', color='black',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3', linewidth=0.5))

        safe_name = zone_name.split('(')[0].strip().replace('/', '_').replace(' ', '_')
        out_file = os.path.join(OUTPUT_DIR, f"SIF_Fixed_{safe_name}.png")

        plt.tight_layout()
        plt.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f"   ✅ Saved: {out_file}")
        plt.close()

    print("🎉 SIF Validation Completed!")


if __name__ == "__main__":
    run_sif_validation()
