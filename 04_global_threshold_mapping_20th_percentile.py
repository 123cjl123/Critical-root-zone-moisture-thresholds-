import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import os
import warnings
from tqdm import tqdm

# 忽略运行时的一些警告
warnings.filterwarnings('ignore')

# ================= 🔧 1. 参数与路径配置 =================

# 📂 输入文件路径 (请确保路径正确)
FILE_SM_ROOT = r""
FILE_EF_RAW = r""
FILE_TA = r""
FILE_FOREST = r""

# 📂 输出设置
OUTPUT_DIR = r""
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_TIF_RAW = os.path.join(OUTPUT_DIR, "Global_Thresholds_20th_Raw_Optimized.tif")
OUTPUT_TIF_IDW = os.path.join(OUTPUT_DIR, "Global_Thresholds_20th_IDW_Filled_Optimized.tif")

# ⚙️ 核心算法参数 (针对高纬度优化版)
PERCENTILE_STD = 20
TEMP_THRESHOLD = 5.0  # 核心限制：温度必须 > 5°C
MIN_SAMPLES = 12  # 稍微放宽：允许生长季较短的区域（至少12个有效月）
MIN_BIN_POINTS = 2  # 优化：每个分箱最少点数降至2，增加小样本区域的成功率
THRESHOLD_RANGE = (0.02, 0.70)

# 🛡️ 质量控制 (QC) 参数
R2_THRESHOLD = 0.02  # 保留 R2 过滤，剔除无物理意义的拟合
MAX_INTERP_DIST = 2  # 优化：扩大插值半径到3度，更好地填补零星空洞


# ================= 🧰 2. 核心工具函数 =================

def linear_plateau(x, sm_crit, slope, intercept):
    """ 线性-平台模型公式 """
    return np.where(x < sm_crit, slope * x + intercept, slope * sm_crit + intercept)


def fit_pixel_threshold_with_stats(sm, ef):
    """ 执行自适应分箱边界线分析 """
    mask = np.isfinite(sm) & np.isfinite(ef)
    x, y = sm[mask], ef[mask]

    # 1. 基础检查
    if len(x) < MIN_SAMPLES: return np.nan, np.nan
    if x.max() == x.min(): return np.nan, np.nan

    try:
        # 2. 【核心优化：自适应分箱】
        # 根据样本量动态调整箱子数，确保每个箱子不为空
        num_bins = int(np.clip(len(x) // 4, 5, 15))

        bins = np.linspace(x.min(), np.percentile(x, 99.5), num_bins)
        bx, by = [], []
        dig = np.digitize(x, bins)

        # 3. 边界提取
        for k in range(1, len(bins)):
            m = (dig == k)
            if m.sum() >= MIN_BIN_POINTS:
                bx.append(np.mean(x[m]))
                by.append(np.percentile(y[m], PERCENTILE_STD))

        if len(bx) < 3: return np.nan, np.nan

        xf, yf = np.array(bx), np.array(by)

        # 4. 非线性拟合
        # 初始猜测优化：[阈值, 斜率, 截距]
        p0 = [np.median(xf), 1.0, np.min(yf)]
        lower_bounds = [xf.min(), 0, -2]
        upper_bounds = [xf.max(), 50, 2]

        popt, _ = curve_fit(linear_plateau, xf, yf, p0=p0,
                            bounds=(lower_bounds, upper_bounds), maxfev=5000)

        # 5. 计算 R2
        y_pred = linear_plateau(xf, *popt)
        ss_res = np.sum((yf - y_pred) ** 2)
        ss_tot = np.sum((yf - np.mean(yf)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return popt[0], r2
    except:
        return np.nan, np.nan


def load_and_align(path, var_name, master):
    """ 数据加载与空间对齐 """
    print(f"   -> Loading {var_name}...")
    ds = xr.open_dataset(path, chunks='auto')

    rn = {}
    for k in ds.coords:
        if k in ['valid_time', 't', 'date', 'Time']: rn[k] = 'time'
        if k in ['lon', 'longitude', 'X', 'long']: rn[k] = 'x'
        if k in ['lat', 'latitude', 'Y']: rn[k] = 'y'
    if rn: ds = ds.rename(rn)

    target = [v for v in ds.data_vars if 'spatial' not in v and 'bnds' not in v][0]
    da = ds[target]
    if 'band' in da.dims: da = da.squeeze('band', drop=True)
    if 'time' in da.coords: da['time'] = da.indexes['time'].to_period('M').to_timestamp()

    da.rio.write_crs("EPSG:4326", inplace=True)
    return da.interp(x=master.x, y=master.y, method='nearest').rename(var_name)


# ================= 🚀 3. 主程序逻辑 =================

def generate_global_map():
    print("=" * 60)
    print(f"🚀 优化版全球森林阈值制图")
    print("=" * 60)

    # Step 1: 森林掩膜
    forest_da = rioxarray.open_rasterio(FILE_FOREST).isel(band=0).squeeze()
    if 'longitude' in forest_da.dims: forest_da = forest_da.rename({'longitude': 'x', 'latitude': 'y'})
    forest_da.rio.write_crs("EPSG:4326", inplace=True)
    mask_forest = (forest_da >= 1) & (forest_da <= 5)

    # Step 2: 加载数据
    da_sm = load_and_align(FILE_SM_ROOT, 'SM', forest_da)
    da_ef = load_and_align(FILE_EF_RAW, 'EF', forest_da)
    da_ta = load_and_align(FILE_TA, 'Ta', forest_da)

    # 时间交集
    common_time = np.intersect1d(da_sm.time, da_ef.time)
    common_time = np.intersect1d(common_time, da_ta.time)
    da_sm, da_ef, da_ta = da_sm.sel(time=common_time), da_ef.sel(time=common_time), da_ta.sel(time=common_time)

    # Step 3: 生长季筛选
    ta_mean_sample = da_ta.isel(time=0).mean().compute().item()
    t_thresh_final = TEMP_THRESHOLD + 273.15 if ta_mean_sample > 200 else TEMP_THRESHOLD
    mask_gs = da_ta > t_thresh_final

    # 合并并转为表格
    print("   -> Preparing dataframe...")
    ds_merged = xr.merge([da_sm, da_ef]).where(mask_forest).where(mask_gs)
    df = ds_merged.to_dataframe().dropna().reset_index()

    # Step 4: 逐像元拟合
    pixel_groups = df.groupby(['y', 'x'])
    valid_results, stats = [], {"Success": 0, "Dropped_R2": 0, "Dropped_Range": 0, "Failed_Fit": 0}

    for (lat, lon), group in tqdm(pixel_groups, desc="Fitting", unit="px"):
        th_val, r2_val = fit_pixel_threshold_with_stats(group['SM'].values, group['EF'].values)

        if np.isfinite(th_val):
            if not (THRESHOLD_RANGE[0] < th_val < THRESHOLD_RANGE[1]):
                stats["Dropped_Range"] += 1
                continue
            if r2_val < R2_THRESHOLD:
                stats["Dropped_R2"] += 1
                continue
            stats["Success"] += 1
            valid_results.append({'y': lat, 'x': lon, 'Threshold': th_val})
        else:
            stats["Failed_Fit"] += 1

    # Step 5: 报告
    print(
        f"\n✅ 成功: {stats['Success']} | 🗑️ R2低: {stats['Dropped_R2']} | 🗑️ 范围异常: {stats['Dropped_Range']} | ❌ 拟合失败: {stats['Failed_Fit']}")

    # Step 6: 保存与插值
    if valid_results:
        df_res = pd.DataFrame(valid_results)
        da_raw = df_res.set_index(['y', 'x']).to_xarray()['Threshold'].reindex_like(forest_da)
        da_raw.rio.write_nodata(np.nan, inplace=True).rio.write_crs("EPSG:4326")
        da_raw.rio.to_raster(OUTPUT_TIF_RAW, compress='lzw')

        # 智能插值
        coords_valid, values_valid = df_res[['x', 'y']].values, df_res['Threshold'].values
        grid_x, grid_y = np.meshgrid(forest_da.x.values, forest_da.y.values)
        grid_z = griddata(coords_valid, values_valid, (grid_x, grid_y), method='linear')

        # 距离约束
        tree = cKDTree(coords_valid)
        dists, _ = tree.query(np.column_stack((grid_x.ravel(), grid_y.ravel())), k=1)
        grid_z[dists.reshape(grid_x.shape) > MAX_INTERP_DIST] = np.nan

        da_filled = da_raw.combine_first(
            xr.DataArray(grid_z, coords=[forest_da.y, forest_da.x], dims=['y', 'x']).where(mask_forest))
        da_filled.rio.write_nodata(np.nan, inplace=True).rio.write_crs("EPSG:4326")
        da_filled.rio.to_raster(OUTPUT_TIF_IDW, compress='lzw')
        print(f"🎉 完成! 结果保存至: {OUTPUT_DIR}")
    else:
        print("❌ 未生成有效结果。")


if __name__ == "__main__":
    generate_global_map()
