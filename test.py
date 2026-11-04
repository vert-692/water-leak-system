import pandas as pd
import numpy as np

# 设置随机种子，保证每次生成的规律一致
np.random.seed(2024)

# 生成 2023 年全年的每小时时间戳（共 8760 小时）
dates = pd.date_range(start="2023-01-01", periods=8760, freq="h")

# 构建基础数据表
mock_df = pd.DataFrame({"timestamp": dates})

# 模拟几个核心传感器的异常分数（在 0-100 之间，日常平稳，偶尔小幅波动）
mock_df["temp_site1_anomaly_score"] = np.random.normal(20, 3, 8760).clip(0, 100)
mock_df["sfc_temp_site1_anomaly_score"] = np.random.normal(25, 4, 8760).clip(0, 100)
mock_df["gw_temp_site2_anomaly_score"] = np.random.normal(30, 2, 8760).clip(0, 100)
mock_df["gw_lvl_site2_anomaly_score"] = np.random.normal(35, 2, 8760).clip(0, 100)

# 核心：人为植入两段“漏损危机”
# 第1段：3月中旬（第 1800 到 1950 小时），地下水和温度异常开始急剧上升
mock_df.loc[1800:1950, "gw_temp_site2_anomaly_score"] += np.linspace(0, 45, 151)
mock_df.loc[1800:1950, "gw_lvl_site2_anomaly_score"] += np.linspace(0, 40, 151)

# 第2段：9月下旬（第 6500 到 6700 小时），温度异常大幅度飙升
mock_df.loc[6500:6700, "temp_site1_anomaly_score"] += np.linspace(0, 55, 201)
mock_df.loc[6500:6700, "sfc_temp_site1_anomaly_score"] += np.linspace(0, 50, 201)

# 添加一些随机的设备能耗和缺失值（模拟真实环境）
mock_df["4700008966_kW"] = np.random.normal(10, 5, 8760).clip(0, 100)
mock_df["238045_ws_temp"] = np.random.normal(15, 2, 8760).clip(0, 100)
# 模拟某几个传感器偶尔断线
mock_df.loc[500:550, "sfc_temp_site1_anomaly_score"] = np.nan

# 格式化时间戳，使其与你系统的解析格式完全一致 (DD-MM-YYYY HH:MM)
mock_df["timestamp"] = mock_df["timestamp"].dt.strftime("%d-%m-%Y %H:%M")

# 保存为 CSV 文件
mock_df.to_csv("mock_city_water_data.csv", index=False)
print("成功生成测试数据：mock_city_water_data.csv")