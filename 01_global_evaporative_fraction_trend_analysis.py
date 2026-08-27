import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import rioxarray
from rasterio.enums import Resampling
import os

# ================= 🔧 配置区域 =================
INPUT_NC = r""
FOREST_TIF = r""
# 输出目录
OUTPUT_DIR = r""
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_hybrid_trend():
    print("⏳ [1/6] 读取数据...")
    ds = xr.open_dataset(INPUT_NC)
    if 'time' in ds.dims: ds = ds.rename({'time': 'valid_time'})

    # ================= 2. 空间对齐与森林掩膜 =================
    print("🌲 [2/6] 森林对齐处理...")
    ef_da = ds['EF']
    if ef_da.rio.crs is None: ef_da.rio.write_crs("EPSG:4326", inplace=True)

    forest_da = rioxarray.open_rasterio(FOREST_TIF, mask_and_scale=False)
    if 'band' in forest_da.dims: forest_da = forest_da.isel(band=0).squeeze()
    if forest_da.rio.crs is None: forest_da.rio.write_crs("EPSG:4326", inplace=True)

    # 对齐
    forest_aligned = forest_da.rio.reproject_match(ef_da, resampling=Resampling.nearest)
    forest_aligned = forest_aligned.rename({'x': 'longitude', 'y': 'latitude'})
    forest_aligned = forest_aligned.assign_coords({"longitude": ef_da.longitude, "latitude": ef_da.latitude})

    # 森林掩膜 (1-5)
    mask_forest = (forest_aligned >= 1) & (forest_aligned <= 5)

    # 只 Mask 非森林区域 (此时 P 和 Slope 都是全森林覆盖)
    ds_masked = ef_da.where(mask_forest)

    # ================= 3. 去季节化 =================
    print("📉 [3/6] 计算异常值...")
    climatology = ds_masked.groupby('valid_time.month').mean('valid_time', skipna=True)
    anomaly_da = ds_masked.groupby('valid_time.month') - climatology

    # ================= 4. 回归计算 =================
    print("🧮 [4/6] 计算 Slope 和 P-value...")
    n_months = len(anomaly_da.valid_time)
    x_axis = np.arange(n_months)

    def linear_trend_func(y):
        mask = ~np.isnan(y)
        if np.sum(mask) < (n_months * 0.5):
            return np.nan, np.nan
        slope, intercept, r, p, err = stats.linregress(x_axis[mask], y[mask])
        return slope * 12, p

    result = xr.apply_ufunc(
        linear_trend_func,
        anomaly_da,
        input_core_dims=[['valid_time']],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, float]
    )
    slope_da, p_value_da = result

    # 确保只保留森林区
    slope_da = slope_da.where(mask_forest)
    p_value_da = p_value_da.where(mask_forest)

    # ================= 5. 【关键差异处理】 =================
    print("⚙️ [5/6] 执行差异化处理...")

    # A. Slope 图：不做任何额外处理 (保持全覆盖)
    slope_final = slope_da.astype(np.float32)

    # B. P-value 图：剔除 > 0.05 的区域 (设为 NaN)
    # 逻辑：只保留 p < 0.05 的值，其他的变成 NaN
    p_value_final = p_value_da.where(p_value_da < 0.05).astype(np.float32)

    # 切除南极
    slope_final = slope_final.sel(latitude=slice(90, -60))
    p_value_final = p_value_final.sel(latitude=slice(90, -60))

    # ================= 6. 保存结果 =================
    print("💾 [6/6] 保存 NC 和 TIF 文件...")

    # --- 1. 保存 Slope (Full) ---
    # NC
    slope_final.name = 'slope'
    slope_final.attrs = {'units': 'year^-1', 'description': 'Full Annual Trend (All Pixels)'}
    slope_nc_path = os.path.join(OUTPUT_DIR, "Forest_EF_Slope_Full.nc")
    slope_final.to_netcdf(slope_nc_path, encoding={'slope': {'_FillValue': np.nan, 'zlib': True}})

    # TIF
    slope_final.rio.write_crs("EPSG:4326", inplace=True)
    slope_tif_path = os.path.join(OUTPUT_DIR, "Forest_EF_Slope_Full.tif")
    slope_final.rio.to_raster(slope_tif_path, compress='lzw', nodata=np.nan)
    print(f"   ✔ Slope (全覆盖): {slope_tif_path}")

    # --- 2. 保存 P-value (Significant Only) ---
    # NC
    p_value_final.name = 'p_value'
    p_value_final.attrs = {'description': 'Significance P-value (Only P < 0.05)'}
    pval_nc_path = os.path.join(OUTPUT_DIR, "Forest_EF_Pvalue_SigOnly.nc")
    p_value_final.to_netcdf(pval_nc_path, encoding={'p_value': {'_FillValue': np.nan, 'zlib': True}})

    # TIF
    p_value_final.rio.write_crs("EPSG:4326", inplace=True)
    pval_tif_path = os.path.join(OUTPUT_DIR, "Forest_EF_Pvalue_SigOnly.tif")
    p_value_final.rio.to_raster(pval_tif_path, compress='lzw', nodata=np.nan)
    print(f"   ✔ P-value (仅显著): {pval_tif_path}")

    # --- 3. 生成一张示意图 ---
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.axes()

    # 画全覆盖的 Slope
    slope_plot = slope_final.plot(ax=ax, cmap='RdYlGn', vmin=-0.005, vmax=0.005, add_colorbar=False)

    # 将处理过的 P-value (SigOnly) 作为纹理层叠加
    # 因为 p_value_final 里不显著的已经是 NaN 了，所以直接画就行
    # 这里我们只用来打点，或者你可以直接不画，因为你的需求是文件处理
    if not np.isnan(p_value_final).all():
        ax.contourf(slope_final.longitude, slope_final.latitude, p_value_final,
                    levels=[0, 0.05], hatches=['...'], colors='none')

    plt.title("Slope (Full) + Stippling (P < 0.05)")
    cb = plt.colorbar(slope_plot, orientation='vertical', pad=0.02)
    cb.set_label("Trend (year$^{-1}$)")
    plt.savefig(os.path.join(OUTPUT_DIR, "Preview_Hybrid.png"), bbox_inches='tight')
    plt.close()

    print("✅ 全部完成！检查 Hybrid_Output 文件夹。")


if __name__ == "__main__":
    analyze_hybrid_trend()
