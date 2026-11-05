import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 页面配置：设置浏览器标签页标题与宽屏模式
st.set_page_config(page_title="供水管网漏损智能预警系统", layout="wide")


# 缓存加载模型函数：避免每次交互重复加载模型，提升响应速度
@st.cache_resource
def load_models():
    main = joblib.load("main_model.pkl")
    domain = joblib.load("domain_shift_models.pkl")
    return main, domain


# 模型加载及异常处理
try:
    main_model_data, domain_model_data = load_models()
    model = main_model_data["model"]
    feature_cols = main_model_data["feature_cols"]
except FileNotFoundError:
    st.error("未找到模型文件。请先运行 competition.py 生成 main_model.pkl 和 domain_shift_models.pkl。")
    st.stop()

# 侧边栏导航配置
with st.sidebar:
    st.title("系统导航")
    page = st.radio("功能模块", [
        "智能漏损预警",
        "跨域适配中心",
        "系统与方法说明"
    ])
    st.markdown("---")
    st.info("核心优势：\n\n不依赖特定传感器，兼容多源异常分数；少样本微调，快速适配新城市管网。")

# ==========================================
# 页面 1：智能漏损预警
# ==========================================
if page == "智能漏损预警":
    st.title("管网实时漏损预警看板")
    st.markdown("上传区域管网的异常分数时序数据，系统将自动融合多源特征，提前研判漏损风险。")

    # 使用 session_state 实现跨页面数据持久化
    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None
    if "file_name" not in st.session_state:
        st.session_state["file_name"] = ""

    # 文件上传组件
    uploaded_file = st.file_uploader("请上传管网数据 (CSV格式)", type=["csv"])

    # 演示数据加载按钮
    demo_clicked = st.button("无数据？点击加载演示样例数据")

    # 数据处理逻辑：读取用户上传文件或生成演示数据
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state["file_name"]:
            st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)
            st.session_state["file_name"] = uploaded_file.name
    elif demo_clicked:
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=168, freq="h")
        demo_df = pd.DataFrame({"timestamp": dates})
        for col in ["gw_temp_site2_anomaly_score", "temp_site1_anomaly_score"]:
            demo_df[col] = np.random.normal(30, 5, 168)
        demo_df.loc[100:120, "gw_temp_site2_anomaly_score"] += np.linspace(0, 30, 21)

        st.session_state["uploaded_df"] = demo_df
        st.session_state["file_name"] = "内置演示数据"

    # 获取当前内存中的数据
    df_input = st.session_state["uploaded_df"]

    if df_input is not None:
        st.success(f"当前已加载数据：{st.session_state['file_name']}")

        try:
            # 1. 特征工程：数据拷贝并补全滑动窗口特征
            df_proc = df_input.copy()


            if "timestamp" in df_proc.columns:
                ts_str = df_proc["timestamp"].astype(str).str.strip().str.replace("/", "-")
                df_proc["plot_time"] = pd.to_datetime(ts_str, format="mixed", dayfirst=True, errors="coerce")

            core_cols = [c for c in df_proc.columns if "anomaly_score" in c]
            for col in core_cols:
                df_proc[f"{col}_mean_24h"] = df_proc[col].rolling(24, min_periods=1).mean()
                df_proc[f"{col}_std_24h"] = df_proc[col].rolling(24, min_periods=1).std().fillna(0)
                df_proc[f"{col}_diff_24h"] = df_proc[col] - df_proc[f"{col}_mean_24h"]
                df_proc[f"{col}_mean_72h"] = df_proc[col].rolling(72, min_periods=1).mean()
                df_proc[f"{col}_diff_72h"] = df_proc[col] - df_proc[f"{col}_mean_72h"]

            # 补全模型所需的缺失特征列，防止推理报错
            for col in feature_cols:
                if col not in df_proc.columns:
                    df_proc[col] = 0

            # 2. 模型推理：计算漏损风险概率
            X_input = df_proc[feature_cols].fillna(0)
            df_proc["漏损风险概率"] = model.predict_proba(X_input)[:, 1]

            # 3. 概览指标：渲染风险统计卡片
            st.markdown("### 研判结果概览")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("分析时段", f"{len(df_proc)} 小时")

            high_risk_count = (df_proc["漏损风险概率"] > 0.6).sum()
            c2.metric("高风险时段", f"{high_risk_count} 小时", delta="-预警" if high_risk_count > 0 else "安全",
                      delta_color="inverse")

            max_prob = df_proc["漏损风险概率"].max()
            c3.metric("最高风险概率", f"{max_prob:.1%}")

            if max_prob > 0.6:
                c4.error("建议派发巡检工单")
            else:
                c4.success("管网运行正常")

            # 4. 趋势图渲染：全量数据 + 纯平移底部滑块 + 滚轮缩放
            st.markdown("### 漏损风险趋势图")

            if "plot_time" in df_proc.columns:
                df_valid = df_proc.dropna(subset=["plot_time"]).sort_values("plot_time").copy()
                df_valid["漏损风险概率"] = pd.to_numeric(df_valid["漏损风险概率"], errors="coerce")
                df_valid = df_valid.dropna(subset=["漏损风险概率"])

                if len(df_valid) == 0:
                    st.warning("图表渲染失败：时间格式无法识别或数据为空，请检查原始 CSV 的 timestamp 列。")
                else:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df_valid["plot_time"],
                        y=df_valid["漏损风险概率"],
                        mode='lines',
                        name='漏损概率',
                        line=dict(color='#00bfff', width=1.2)
                    ))

                    fig.add_hline(y=0.6, line_dash="dash", line_color="#ff4b4b", annotation_text="高风险阈值")

                    min_time = df_valid["plot_time"].min()
                    initial_end_time = min_time + pd.Timedelta(days=90)

                    fig.update_layout(
                        title="全周期漏损风险趋势 (底部滑块平移，图表区域滚轮缩放)",
                        xaxis_title="时间",
                        yaxis_title="风险概率",
                        height=550,
                        margin=dict(l=40, r=20, t=50, b=40),
                        yaxis=dict(range=[0, 1.0], fixedrange=True),
                        xaxis=dict(
                            type='date',
                            tickformat="%Y年%m月",
                            range=[min_time, initial_end_time],
                            rangeslider=dict(
                                visible=True,
                                thickness=0.12,
                                bgcolor="#1e1e1e",
                                bordercolor="#444444",
                                borderwidth=1
                            )
                        )
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'displaylogo': False,
                            'modeBarButtonsToAdd': ['togglefullscreen'],
                            'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d']
                        }
                    )

            # 5. 工单列表：展示高风险时段明细
            if high_risk_count > 0:
                st.markdown("### 巡检工单建议")
                warn_df = df_proc[df_proc["漏损风险概率"] > 0.6].copy()
                warn_df = warn_df[["timestamp", "漏损风险概率"]].sort_values("漏损风险概率", ascending=False)
                warn_df["建议动作"] = "派发巡检"
                st.dataframe(warn_df.head(50), use_container_width=True)

        except Exception as e:
            st.error(f"分析出错，请检查数据格式：{e}")

# ==========================================
# 页面 2：跨域适配中心
# ==========================================
elif page == "跨域适配中心":
    st.title("跨域泛化与本地化适配")
    st.markdown("""
    针对“不同城市管网差异大，模型水土不服”的痛点，本系统采用“槽位对齐 + 少样本微调”技术。
    水司只需提供极少量本地的真实漏损数据，即可将模型快速适配到新管网。
    """)

    st.markdown("### 跨域迁移实验数据（基于芬兰 A/B 站点）")

    col1, col2 = st.columns([1, 1])
    with col1:
        try:
            st.image("fig6_domain_shift.png", caption="源域测试 vs 零样本迁移 vs 微调后", use_container_width=True)
        except:
            st.warning("图表未生成，请先运行主程序。")

    with col2:
        st.markdown("""
        #### 实验解读
        *   **源域测试 (0.861)**：模型在本地表现优秀。
        *   **直接跨域 (0.513)**：直接预测新城市，性能跌至盲猜水平。
        *   **少样本微调 (0.641)**：只需提供 2-3 个本地真实漏损记录，模型性能迅速回升至可用状态。

        #### 落地价值
        将系统部署到新城市时，无需收集长达一年的数据。
        *   第一步：接入历史异常分数。
        *   第二步：积累 2-3 次真实漏损记录。
        *   第三步：执行一键适配，模型即刻适应当地管网并开始预警。
        """)

    st.markdown("---")
    st.markdown("### 上传本地漏损事件进行微调 (演示功能)")
    st.file_uploader("上传本地标注数据 (包含 fault_short 列)", type=["csv"])
    st.button("开始微调模型")
    st.caption("（演示版中该按钮为占位符。实际业务中，后台将使用增量学习自动更新模型参数。）")

# ==========================================
# 页面 3：系统与方法说明
# ==========================================
else:
    st.title("系统架构与算法说明")

    st.markdown("""
    ### 为什么本系统能适配不同数据？

    实际工程中，传感器厂商不同，采集到的数据格式（压力、流量、声波）完全不同。
    本系统不处理原始信号，而是接收“异常分数”（0-100）。
    通过多源分数融合与二次研判，屏蔽底层硬件差异。

    ### 核心算法三把斧

    1. **时间切片（滑动窗口）**：捕捉渐进式漏损趋势。基于过去 24 到 72 小时的波动研判，而非依赖瞬时值。
    2. **标签重构**：将原始标签的 ±7 天窗口，重构为“漏损发生前 24 小时”，降低噪声，提高模型精度。
    3. **事件切分验证**：按真实漏损事件切分数据集，而非按时间切分，保证评估结果的真实性和可靠性。

    ### 系统价值

    *   **精准巡检**：从被动巡线转变为接收系统工单的精准排查。
    *   **提前止损**：提前 2-4 天发现地下水温/水位异动，将漏损控制在萌芽阶段。
    """)