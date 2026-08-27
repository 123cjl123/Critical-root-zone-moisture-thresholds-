import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import rioxarray
from rasterio.enums import Resampling
import os
import warnings
import matplotlib.ticker as ticker

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 🔧 1. 数据路径配置 =================
MIN_PIXEL_COUNT = 100
MIN_SAMPLE_POINTS = 500
TEMP_THRESHOLD_C = 5.0

# ⚠️ 请根据你的实际路径修改
FILE_SM_ABS = r""
FILE_EF_ANOM = r""
FILE_TA = r""
FOREST_TIF = r""
AI_TIF = r""

OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 🎨 2. 绘图样式独立控制区 =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'

FIG_WIDTH_CM = 29.5
FIG_WIDTH_INCH = FIG_WIDTH_CM / 2.54
FIG_HEIGHT_INCH = 4.5

SIZE_TITLE = 10
SIZE_AXIS_LABEL = 10
SIZE_TICK = 8
SIZE_LEGEND = 7  # 缩小一点图例，防止遮挡

WEIGHT_TITLE = 'bold'
WEIGHT_AXIS_LABEL = 'bold'
WEIGHT_TICK = 'normal'


# ================= 🧰 3. 工具函数 =================

def linear_plateau(x, sm_crit, slope, intercept):
    """线性-平台模型物理方程"""
    return np.where(x < sm_crit, slope * x + intercept, slope * sm_crit + intercept)


def unify_and_format_time(da):
    """统一维度名称并规范化时间坐标"""
    rename_map = {}
    if 'valid_time' in da.dims: rename_map['valid_time'] = 'time'
    if 'latitude' in da.dims: rename_map['latitude'] = 'lat'
    if 'longitude' in da.dims: rename_map['longitude'] = 'lon'
    if rename_map:
        da = da.rename(rename_map)

    if 'time' in da.coords:
        idx = da.indexes['time']
        if not isinstance(idx, pd.DatetimeIndex):
            idx = idx.to_datetimeindex()
        da = da.assign_coords(time=pd.to_datetime(idx.values).to_period('M').to_timestamp())
    return da


# ================= 🚀 4. 主程序 =================

def run_analysis():
    print("Step 1: Loading Data...")
    ds_sm = xr.open_dataset(FILE_SM_ABS)
    ds_ef = xr.open_dataset(FILE_EF_ANOM)
    ds_ta = xr.open_dataset(FILE_TA)

    sm_da = ds_sm['SMrz'] if 'SMrz' in ds_sm else ds_sm['SMroot']
    ef_da = ds_ef['EF_anom']
    ta_var = next((v for v in ['Ta', 't2m', 'T2m', 'tmp'] if v in ds_ta), None)
    ta_da = ds_ta[ta_var]

    print("   Unifying dimensions...")
    sm_da = unify_and_format_time(sm_da)
    ef_da = unify_and_format_time(ef_da)
    ta_da = unify_and_format_time(ta_da)

    common_time = np.intersect1d(sm_da.time, ef_da.time)
    common_time = np.intersect1d(common_time, ta_da.time)

    sm_da = sm_da.sel(time=common_time)
    ef_da = ef_da.sel(time=common_time)
    ta_da = ta_da.sel(time=common_time)

    # 同步坐标防止对齐错误
    ef_da = ef_da.assign_coords(lat=sm_da.lat, lon=sm_da.lon)
    ta_da = ta_da.assign_coords(lat=sm_da.lat, lon=sm_da.lon)

    for da in [sm_da, ef_da, ta_da]:
        if da.rio.crs is None: da.rio.write_crs("EPSG:4326", inplace=True)

    print("Step 1.5: Growing Season Filter...")
    ta_mean = ta_da.mean().item()
    thresh_val = TEMP_THRESHOLD_C + 273.15 if ta_mean > 200 else TEMP_THRESHOLD_C
    if ta_da.shape != sm_da.shape:
        ta_da = ta_da.rio.reproject_match(sm_da, resampling=Resampling.nearest)

    mask_gs = ta_da > thresh_val
    sm_da = sm_da.where(mask_gs)
    ef_da = ef_da.where(mask_gs)

    print("Step 2: Loading Masks...")
    forest_da = rioxarray.open_rasterio(FOREST_TIF).isel(band=0).squeeze()
    ai_da = rioxarray.open_rasterio(AI_TIF).isel(band=0).squeeze()
    forest_da = forest_da.rio.reproject_match(ef_da, resampling=Resampling.nearest)
    ai_da = ai_da.rio.reproject_match(ef_da, resampling=Resampling.bilinear)
    forest_da = forest_da.rename({'x': 'lon', 'y': 'lat'}).assign_coords(lon=ef_da.lon, lat=ef_da.lat)
    ai_da = ai_da.rename({'x': 'lon', 'y': 'lat'}).assign_coords(lon=ef_da.lon, lat=ef_da.lat)

    forest_types = {1: "EBF", 2: "DBF", 3: "ENF", 4: "DNF", 5: "MF"}
    colors = {1: '#2ca02c', 2: '#98df8a', 3: '#1f77b4', 4: '#aec7e8', 5: '#d62728'}
    ai_zones = [
        {"name": "Semi-Arid (AI 0.2-0.5)", "range": (0.2, 0.5)},
        {"name": "Semi-arid Subhumid (AI 0.5-0.65)", "range": (0.5, 0.65)},
        {"name": "Humid (AI > 0.65)", "range": (0.65, 100)}
    ]

    percentiles = [5, 10, 20, 25, 30]
    stats_list = []

    print("\nStep 3: Plotting & Regression Loop...")

    for p_val in percentiles:
        print(f"\n >>> Processing {p_val}th Percentile...")
        fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_INCH, FIG_HEIGHT_INCH), sharey=True, dpi=300)
        if not isinstance(axes, np.ndarray): axes = [axes]

        for col_idx, zone in enumerate(ai_zones):
            ax = axes[col_idx]
            z_name, (z_min, z_max) = zone["name"], zone["range"]
            ax.grid(True, which='major', linestyle=':', color='gray', alpha=0.3)
            mask_zone = (ai_da >= z_min) & (ai_da < z_max)
            has_data = False

            for f_id, f_name in forest_types.items():
                final_mask = mask_zone & (forest_da == f_id)
                if np.sum(final_mask.values) < MIN_PIXEL_COUNT: continue

                # 时空池化提取
                sm_v = sm_da.where(final_mask).values.flatten()
                ef_v = ef_da.where(final_mask).values.flatten()
                valid = np.isfinite(sm_v) & np.isfinite(ef_v)
                if not np.any(valid): continue
                x_d, y_d = sm_v[valid], ef_v[valid]

                if len(x_d) < MIN_SAMPLE_POINTS: continue
                if x_d.max() > 1.5: x_d /= 100.0  # 单位修正

                # 等频分箱处理
                _, upper = np.percentile(x_d, [0, 99])
                bins = np.linspace(x_d.min(), upper, 30)
                digitized = np.digitize(x_d, bins)
                bin_x, bin_y = [], []
                for k in range(1, len(bins)):
                    in_bin = y_d[digitized == k]
                    in_bin_x = x_d[digitized == k]
                    # 严格控制分箱样本量以满足边界稳健性
                    if len(in_bin) > 500:
                        bin_x.append(np.mean(in_bin_x))
                        bin_y.append(np.percentile(in_bin, p_val))

                x_fit, y_fit = np.array(bin_x), np.array(bin_y)
                if len(x_fit) < 5: continue

                try:
                    p0 = [np.mean(x_fit), 2.0, -0.1]
                    bounds = ([x_fit.min(), 0, -2], [x_fit.max(), 20, 1])
                    # ✨ 执行拟合
                    popt, pcov = curve_fit(linear_plateau, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=5000)

                    # ✨ 提取参数与 SE (Standard Error)
                    sm_crit = popt[0]
                    # SE = sqrt(方差)，对应 pcov 对角线元素
                    se_sm_crit = np.sqrt(np.diag(pcov))[0] if pcov is not None else np.nan

                    # 计算 R2
                    y_pred = linear_plateau(x_fit, *popt)
                    ss_res = np.sum((y_fit - y_pred) ** 2)
                    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))

                    stats_list.append({
                        'Percentile': p_val, 'Zone': z_name, 'Forest': f_name,
                        'Threshold': sm_crit, 'SE': se_sm_crit, 'R2': r2
                    })

                    c = colors[f_id]
                    ax.scatter(x_fit, y_fit, color=c, s=25, alpha=0.5, edgecolor='none', zorder=2)
                    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
                    y_line = linear_plateau(x_line, *popt)
                    ax.plot(x_line, y_line, color=c, linewidth=1.8, zorder=3,
                            label=f"{f_name} ($\Theta$={sm_crit:.2f}±{se_sm_crit:.3f}, $R^2$={r2:.2f})")
                    ax.axvline(sm_crit, color=c, linestyle=':', linewidth=1.0, alpha=0.7)
                    has_data = True
                except:
                    pass

            # 子图样式控制
            ax.set_title(z_name, fontsize=SIZE_TITLE, fontweight=WEIGHT_TITLE, pad=10)
            ax.set_xlabel(r"Root Soil Moisture ($m^3/m^3$)", fontsize=SIZE_AXIS_LABEL, fontweight=WEIGHT_AXIS_LABEL)
            ax.set_ylim(-0.4, 0.15)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel(f"EF Anomaly ({p_val}th Percentile)", fontsize=SIZE_AXIS_LABEL,
                              fontweight=WEIGHT_AXIS_LABEL)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=SIZE_TICK)
            if has_data:
                ax.legend(loc='lower right', fontsize=SIZE_LEGEND, frameon=True, framealpha=0.8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"Sens_{p_val:02d}th_Result.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {out_path}")

    # ================= 📊 5. 输出汇总统计表 =================
    if stats_list:
        df = pd.DataFrame(stats_list)
        df.to_csv(os.path.join(OUTPUT_DIR, "Detailed_Statistics.csv"), index=False)

        # 生成你想要的那种 Summary 表格
        summary = df.groupby('Percentile').agg({
            'R2': 'mean',
            'SE': 'mean',
            'Threshold': 'mean',
            'Forest': 'count'
        }).rename(
            columns={'Forest': 'Model_Count', 'R2': 'Average R2', 'SE': 'Average SE', 'Threshold': 'Average Threshold'})

        print("\n" + "=" * 50)
        print("📊 MODEL SENSITIVITY SUMMARY TABLE")
        print("=" * 50)
        print(summary.round(4))
        summary.to_csv(os.path.join(OUTPUT_DIR, "Final_Summary_Table.csv"))
        print("\n📄 All results saved to:", OUTPUT_DIR)
    else:
        print("❌ Error: No valid models fitted. Please check data alignment.")


if __name__ == "__main__":
    run_analysis()
