import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rioxarray
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import t
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 🔧 1. 参数配置 (Configuration) =================

# 📂 输入数据路径
FILE_SM_NC = r""
VAR_SM = 'SMrz'
FILE_THRESHOLD_TIF = r""
FILE_TA = r""
FILE_FOREST = r""

# 📂 输出目录
OUTPUT_DIR = r""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ⚙️ 核心参数
TEMP_THRESHOLD = 5.0

# 🎮 IDW 插值控制开关 (在这里修改!)
# True  = 开启插值 (填补空洞，图像连续)
# False = 关闭插值 (显示原始数据，可能有空洞)
ENABLE_IDW = True

# 插值半径 (仅当 ENABLE_IDW = True 时生效)
MAX_INTERP_DIST = 0.2  # 度 (约 20km)

# 绘图字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


# ================= 🧰 2. 核心工具函数 =================

def standardize_dims(da):
    """标准化维度名称"""
    rn = {}
    for k in da.dims:
        if k in ['y', 'lat', 'Y']: rn[k] = 'latitude'
        if k in ['x', 'lon', 'long', 'X']: rn[k] = 'longitude'
        if k in ['valid_time', 't', 'date']: rn[k] = 'time'
    if rn: da = da.rename(rn)
    return da


def force_align(da_slave, da_master):
    """强制对齐坐标"""
    da_slave = standardize_dims(da_slave)
    da_master = standardize_dims(da_master)
    if 'latitude' not in da_slave.dims: return da_slave

    if da_slave.shape[-2:] != da_master.shape[-2:]:
        da_slave = da_slave.interp(
            latitude=da_master.latitude,
            longitude=da_master.longitude,
            method='nearest'
        )
    return da_slave.assign_coords({
        'latitude': da_master.latitude,
        'longitude': da_master.longitude
    })


def advanced_idw_fill(da_target, da_mask, max_dist=MAX_INTERP_DIST):
    """
    🛠️ 高级 IDW 插值函数 (带距离控制)
    """
    if not ENABLE_IDW:
        return da_target  # 如果开关关闭，直接返回原数据

    print(f"      ...Applying IDW Interpolation (Dist={max_dist})...")

    # 1. 提取有效点
    df = da_target.to_dataframe(name='value').reset_index()
    df_valid = df.dropna(subset=['value'])

    if len(df_valid) == 0: return da_target

    coords_valid = df_valid[['longitude', 'latitude']].values
    values_valid = df_valid['value'].values

    # 2. 生成网格
    grid_x, grid_y = np.meshgrid(da_target.longitude.values, da_target.latitude.values)

    # 3. 插值
    grid_z = griddata(coords_valid, values_valid, (grid_x, grid_y), method='linear')

    # 4. 距离掩膜
    tree = cKDTree(coords_valid)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dists, _ = tree.query(grid_points, k=1, workers=-1)
    dists_grid = dists.reshape(grid_x.shape)

    grid_z[dists_grid > max_dist] = np.nan

    # 5. 合并
    da_filled = xr.DataArray(grid_z, coords=[da_target.latitude, da_target.longitude], dims=['latitude', 'longitude'])
    da_final = da_target.combine_first(da_filled)

    return da_final.where(da_mask)


def save_wgs84_tif(da, filename, output_dir):
    """仅保存 WGS84 格式 TIF"""
    print(f"   💾 Saving TIF: {filename}...")
    da.rio.write_nodata(np.nan, inplace=True)
    if da.rio.crs is None: da.rio.write_crs("EPSG:4326", inplace=True)
    save_path = os.path.join(output_dir, filename)
    da.rio.to_raster(save_path, compress='LZW')


def load_and_align():
    print("Step 1: Loading Data...")
    ds_sm = xr.open_dataset(FILE_SM_NC)
    da_sm = standardize_dims(ds_sm[VAR_SM])
    try:
        da_sm = da_sm.assign_coords(time=da_sm.indexes['time'].to_datetimeindex())
    except:
        pass

    da_thresh = rioxarray.open_rasterio(FILE_THRESHOLD_TIF).isel(band=0).squeeze()
    da_thresh = force_align(da_thresh, da_sm)

    da_forest = rioxarray.open_rasterio(FILE_FOREST).isel(band=0).squeeze()
    da_forest = force_align(da_forest, da_sm)

    ds_ta = xr.open_dataset(FILE_TA)
    var_ta = 'Ta' if 'Ta' in ds_ta else 't2m'
    da_ta = standardize_dims(ds_ta[var_ta])
    try:
        da_ta = da_ta.assign_coords(time=da_ta.indexes['time'].to_datetimeindex())
    except:
        pass

    common_time = np.intersect1d(da_sm.time, da_ta.time)
    da_sm = da_sm.sel(time=common_time)
    da_ta = da_ta.sel(time=common_time)
    da_ta = force_align(da_ta, da_sm)

    return da_sm, da_thresh, da_forest, da_ta


# ================= 🚀 3. 主分析程序 =================

def run_main_analysis():
    # 1. 加载
    da_sm, da_thresh, da_forest, da_ta = load_and_align()

    # 2. 掩膜
    print("Step 2: Masking...")
    mask_gs = (da_ta > (TEMP_THRESHOLD + 273.15))
    mask_forest = (da_forest >= 1) & (da_forest <= 5)
    mask_valid_thresh = np.isfinite(da_thresh) & (da_thresh > 0)
    final_mask = mask_gs & mask_forest & mask_valid_thresh

    da_sm_masked = da_sm.where(final_mask)
    is_stressed = da_sm_masked < da_thresh

    # ================= PART 1: Risk Map =================
    print("\nStep 3: Calculating Risk Map...")
    valid_months = da_sm_masked.count(dim='time')
    stressed_months = is_stressed.where(da_sm_masked.notnull()).sum(dim='time')

    freq_map = (stressed_months / valid_months) * 100
    freq_map = freq_map.where(mask_forest)

    # 🎮 插值控制
    freq_map_final = advanced_idw_fill(freq_map, mask_forest)
    freq_map_final.rio.write_crs("EPSG:4326", inplace=True)

    # 绘图 & 保存
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    freq_map_final.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=0, vmax=80,
                        cbar_kwargs={'label': 'Freq (%)', 'shrink': 0.7})
    ax.set_title("Risk Map", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_Risk_Map.png"), bbox_inches='tight')
    plt.close()

    save_wgs84_tif(freq_map_final, "Fig1_Risk_Map.tif", OUTPUT_DIR)

    # ================= PART 2: Trend Analysis (Sen+MK Method) =================
    print("\nStep 4: Calculating Trends (Polyfit + t-test)...")
    da_annual_freq = is_stressed.groupby('time.year').mean() * 100
    da_annual_freq = da_annual_freq.where(mask_forest)

    def linear_trend_with_p_sen(y):
        x = xr.DataArray(np.arange(len(y.year)), dims='year', coords={'year': y.year})
        n = y.count(dim='year')
        valid_mask = n >= 7  # 至少5年数据

        try:
            fit = y.polyfit(dim='year', deg=1, full=True)
            slope = fit.polyfit_coefficients.sel(degree=1)

            rss = fit.polyfit_residuals
            if rss.ndim == 0: rss = rss.expand_dims(dim={})

            if rss.size == 0:
                return xr.full_like(y.isel(year=0), np.nan), xr.full_like(y.isel(year=0), np.nan)

            x_mean = x.mean()
            ss_x = ((x - x_mean) ** 2).sum()
            se = np.sqrt(rss / (n - 2) / ss_x)
            t_stat = slope / se

            p_val = xr.apply_ufunc(
                lambda t_val, df: 2 * t.sf(np.abs(t_val), df),
                t_stat, (n - 2),
                input_core_dims=[[], []],
                output_core_dims=[[]],
                vectorize=True,
                dask='parallelized'
            )
            return slope.where(valid_mask), p_val.where(valid_mask)
        except Exception:
            return xr.full_like(y.isel(year=0), np.nan), xr.full_like(y.isel(year=0), np.nan)

    slope_map, p_value_map = linear_trend_with_p_sen(da_annual_freq)

    # 🎮 插值控制
    slope_map_final = advanced_idw_fill(slope_map, mask_forest)
    p_value_map_final = advanced_idw_fill(p_value_map, mask_forest)

    slope_map_final.rio.write_crs("EPSG:4326", inplace=True)
    p_value_map_final.rio.write_crs("EPSG:4326", inplace=True)

    # 显著性 (使用最终插值后的结果计算)
    slope_sig_masked = slope_map_final.where(p_value_map_final < 0.05)
    slope_sig_masked.rio.write_crs("EPSG:4326", inplace=True)

    # 绘图 & 保存
    print("   -> Plotting Trends...")
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    slope_map_final.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-1.5, vmax=1.5,
                         cbar_kwargs={'label': 'Trend (% year⁻¹)', 'shrink': 0.7})
    ax.set_title("Global Trends", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Trend_Map.png"), bbox_inches='tight')
    plt.close()

    save_wgs84_tif(slope_map_final, "Fig2_Trend_Map.tif", OUTPUT_DIR)
    save_wgs84_tif(p_value_map_final, "Fig2_Trend_P_Value.tif", OUTPUT_DIR)
    save_wgs84_tif(slope_sig_masked, "Fig2_Trend_Significant.tif", OUTPUT_DIR)

    # ================= PART 3: Cooling Loss =================
    print("\nStep 5: Calculating Cooling Loss...")
    climatology = da_ta.groupby('time.month').mean('time')
    da_ta_anom = da_ta.groupby('time.month') - climatology

    # 提取有效干旱月
    penalty_raw = da_ta_anom.where(is_stressed & final_mask)
    valid_count = penalty_raw.count(dim='time')

    # 至少3个月记录才计算
    cooling_loss_map = penalty_raw.mean(dim='time').where(valid_count >= 3)
    cooling_loss_map = cooling_loss_map.where(mask_forest)

    # 🎮 插值控制
    cooling_loss_final = advanced_idw_fill(cooling_loss_map, mask_forest)
    cooling_loss_final.rio.write_crs("EPSG:4326", inplace=True)

    print("   -> Plotting Cooling Loss...")
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    cooling_loss_final.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='inferno_r', vmin=0, vmax=2.0,
                            cbar_kwargs={'label': 'Temp. Rise (°C)', 'shrink': 0.7})
    ax.set_title("Thermal Penalty", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Cooling_Loss_Map.png"), bbox_inches='tight')
    plt.close()

    save_wgs84_tif(cooling_loss_final, "Fig3_Cooling_Loss_Map.tif", OUTPUT_DIR)

    print(f"\n✅ All Done! Results saved to: {OUTPUT_DIR}")
    print(f"   ℹ️ IDW Interpolation Status: {'ENABLED' if ENABLE_IDW else 'DISABLED'}")


if __name__ == "__main__":
    run_main_analysis()
