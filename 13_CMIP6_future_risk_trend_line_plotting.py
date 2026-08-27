import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline

# ================= 1. 参数配置区域 =================
INPUT_CSV = r""

# 尺寸 (A4 宽度适配)
FIG_WIDTH = 11
FIG_HEIGHT = 6

# 字体与粗细
FONT_FAMILY = 'Arial'
FONT_WEIGHT_LABEL = 'bold'  # 轴标题加粗
FONT_WEIGHT_TICK = 'normal' # 刻度数字正常

# 字号
SIZE_AXIS_LABEL = 14
SIZE_TICK_LABEL = 14
SIZE_LEGEND = 12

# 颜色
COLOR_SMOOTH = '#008080' # 深青色 (主曲线)
COLOR_SHADE = '#008080'  # 阴影色
COLOR_TREND = '#CD5C5C'  # 印度红 (趋势线)

# ================= 2. 数据处理 =================
# 读取数据
df = pd.read_csv(INPUT_CSV)
df = df.sort_values('Year')
x = df['Year'].values
y = df['Global_Avg_Risk_Freq_Pct'].values

# 1. 计算滑动统计量 (5年窗口)
s_y = pd.Series(y)
y_roll_mean = s_y.rolling(window=5, center=True, min_periods=1).mean()
y_roll_std = s_y.rolling(window=5, center=True, min_periods=1).std()

# 2. 样条插值 (Spline Interpolation) -> 实现“圆滑”效果
# 创建高分辨率的 X 轴 (从 2022 到 2100，生成 500 个点)
x_new = np.linspace(x.min(), x.max(), 500)

# 对 Mean 进行插值
spl_mean = make_interp_spline(x, y_roll_mean.values, k=3) # k=3 表示三次样条
y_smooth = spl_mean(x_new)

# 对 Std 进行插值 (为了阴影也能平滑)
# 填充 NaN 以防首尾计算问题
spl_std = make_interp_spline(x, y_roll_std.fillna(0).values, k=3)
y_std_smooth = spl_std(x_new)
# 修正插值可能产生的负值
y_std_smooth = np.maximum(y_std_smooth, 0)

# 3. 计算线性趋势 (基于原始数据)
slope, intercept, r_value, p_value, std_err = linregress(x, y)
trend_line = slope * x + intercept
# 趋势线标签
trend_text = f"Trend: {slope:+.3f}% year$^{{-1}}$"

# ================= 3. 绘图核心 =================
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)

# A. 绘制平滑阴影 (Smooth Error Band) - 保留图例
ax.fill_between(x_new, y_smooth - y_std_smooth, y_smooth + y_std_smooth,
                color=COLOR_SHADE, alpha=0.2, linewidth=0,
                label='5-Year Variability (±1 SD)')

# B. 绘制平滑曲线 (Smooth Mean Line) - 【图例已移除】
ax.plot(x_new, y_smooth, color=COLOR_SMOOTH, linewidth=2.5)

# C. 绘制线性趋势线 (Trend Line) - 保留图例
ax.plot(x, trend_line, color=COLOR_TREND, linestyle='--', linewidth=2.0,
        label=trend_text)

# ================= 4. 样式精细控制 =================

# --- 坐标轴范围 ---
ax.set_xlim(2020, 2100) # 从 2020 开始
# 动态调整 Y 轴范围
y_lower = (y_smooth - y_std_smooth).min()
y_upper = (y_smooth + y_std_smooth).max()
pad = (y_upper - y_lower) * 0.1
ax.set_ylim(y_lower - pad, y_upper + pad)

# --- 刻度设置 ---
ax.xaxis.set_major_locator(ticker.MultipleLocator(10)) # 每10年
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))  # 每2年

# --- 标签设置 (Y轴保留 Future) ---
ax.set_ylabel("Projected Future Global Hydraulic\nFailure Risk (%)",
              fontsize=SIZE_AXIS_LABEL,
              fontweight=FONT_WEIGHT_LABEL,
              family=FONT_FAMILY,
              labelpad=10)

ax.set_xlabel("Year",
              fontsize=SIZE_AXIS_LABEL,
              fontweight=FONT_WEIGHT_LABEL,
              family=FONT_FAMILY,
              labelpad=5)

# --- 刻度外观 ---
ax.tick_params(axis='both', direction='in', length=6, width=1.2,
               labelsize=SIZE_TICK_LABEL, top=False, right=False)

# 刻度字体控制
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname(FONT_FAMILY)
    label.set_fontweight(FONT_WEIGHT_TICK)

# --- 边框加粗 ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# --- 图例 (右上角) ---
ax.legend(loc='upper right', frameon=False, fontsize=SIZE_LEGEND,
          prop={'family': FONT_FAMILY, 'size': SIZE_LEGEND})

# ================= 5. 保存 =================
plt.tight_layout()
# 修改为 .tif 格式
output_path = r""
# 保存为 300 DPI 的 TIF 文件，PIL库支持更好的压缩（如果需要更小体积可以加 pil_kwargs）
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='tif')
plt.show()
print(f"✅ 高清TIF图表已生成: {output_path}")
