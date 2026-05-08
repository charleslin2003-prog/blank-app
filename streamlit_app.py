import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go

# --- 網頁配置 ---
st.set_page_config(page_title="專業理財決策 v6.4", layout="wide")
st.title("🧪 專業理財決策模擬：v6.4 友善視覺版")

# --- 1. 左側參數設定 ---
with st.sidebar:
    st.header("⚙️ 模擬環境設定")

    with st.expander("📊 經濟環境與相關性", expanded=True):
        n_sims = st.slider("模擬路徑次數", 50, 500, 200)
        avg_inf = st.number_input("年通膨率 (%)", value=1.5) / 100
        inf_vol = st.slider("通膨波動度 (%)", 0.0, 3.0, 1.1) / 100
        fail_buffer = st.slider("破產緩衝 (年支出倍數)", 0.0, 5.0, 0.5)

        st.write("---")
        st.write("**資產相關性 (Correlation)**")
        rho_sb = st.slider("股-債相關係數", -1.0, 1.0, -0.2)
        rho_sc = 0.0
        rho_bc = 0.1

    with st.expander("💼 資產配置 (名目報酬)", expanded=True):
        stock_w = st.slider("股票比例 (%)", 0, 100, 60)
        bond_w = st.slider("債券比例 (%)", 0, 100 - stock_w, 30)
        cash_w = 100 - stock_w - bond_w

        stock_ret = st.number_input("股票預期報酬 (%)", value=9.0) / 100
        stock_vol = st.slider("股市年波動度 (%)", 5.0, 30.0, 15.0) / 100
        bond_ret = st.number_input("債券預期報酬 (%)", value=3.5) / 100
        bond_vol = st.slider("債券年波動度 (%)", 0.0, 10.0, 3.0) / 100
        cash_ret = st.number_input("現金報酬 (%)", value=1.5) / 100

    with st.expander("👤 個人收支 (今日價值)", expanded=True):
        current_age = st.number_input("當前年齡", value=23, min_value=18)
        retire_age = st.number_input("退休年齡", value=65, min_value=int(current_age) + 1)
        init_cash = st.number_input("初始資金 (萬)", value=10.0) * 10000
        inc_today = st.number_input("目前月收 (萬)", value=3.5) * 10000
        exp_today = st.number_input("月生活費 (萬)", value=2.0) * 10000
        rent_today = st.number_input("月租金 (萬)", value=1.5) * 10000
        real_inc_grow = st.slider("實質薪資成長 (%)", 0.0, 5.0, 1.5) / 100

    st.header("🏠 房產決策參數")
    with st.expander("購屋與持有成本", expanded=True):
        buy_age = st.number_input("購屋年齡", value=35, min_value=int(current_age), max_value=int(retire_age))
        h_price_today = st.number_input("房價 (今日萬)", value=1800) * 10000
        h_appr = st.slider("房屋年增值 (%)", 0.0, 5.0, 2.0) / 100
        buy_txn_pct = st.slider("購屋交易成本 (佔房價 %)", 0.0, 5.0, 3.0) / 100

        use_adv_h = st.checkbox("使用進階持有成本拆項", value=False)
        if use_adv_h:
            tax_pct = st.slider("稅金 (%/年)", 0.0, 2.0, 0.5) / 100
            maint_pct = st.slider("維修/折舊 (%/年)", 0.0, 2.0, 0.5) / 100
            mgmt_pct = st.slider("管理費 (%/年)", 0.0, 1.0, 0.2) / 100
            h_maint_total = tax_pct + maint_pct + mgmt_pct
        else:
            h_maint_total = st.slider("固定持有成本 (%/年)", 0.0, 3.0, 0.5) / 100
        m_rate, m_years = 0.021, 30


# --- 2. 核心計算引擎 ---
def simulate(buy_house_scenario=True):
    np.random.seed(42)
    ages = np.arange(current_age, 86)
    n_years = len(ages)

    mu_vec = np.array([stock_ret, bond_ret, cash_ret])
    sigma_vec = np.array([stock_vol, bond_vol, 0.01])
    R = np.array([[1.0, rho_sb, rho_sc], [rho_sb, 1.0, rho_bc], [rho_sc, rho_bc, 1.0]])

    eigvals = np.linalg.eigvals(R)
    if np.min(eigvals) <= 0:
        st.error("❌ 相關性矩陣不是正定矩陣。請調低相關係數。")
        st.stop()

    D = np.diag(sigma_vec)
    cov_matrix = D @ R @ D
    L = np.linalg.cholesky(cov_matrix)

    all_liq = np.zeros((n_sims, n_years))
    all_nw = np.zeros((n_sims, n_years))
    all_pp = np.zeros((n_sims, n_years))
    all_exp = np.zeros((n_sims, n_years))

    for s in range(n_sims):
        liq, h_val, m_bal, m_pay, cpi = init_cash, 0, 0, 0, 100.0
        for i, age in enumerate(ages):
            z = np.random.normal(size=3)
            rets = np.clip(mu_vec + L @ z, -0.8, 1.2)
            y_ret = (rets[0] * stock_w + rets[1] * bond_w + rets[2] * cash_w) / 100

            y_inf = np.random.normal(avg_inf, inf_vol)
            cpi *= (1 + y_inf)
            c_f = cpi / 100.0

            y_inc = (inc_today * 12 * (1 + real_inc_grow) ** i * c_f) if age < retire_age else 0
            y_exp = exp_today * 12 * c_f

            h_cost = 0
            if buy_house_scenario:
                y_rent = (rent_today * 12 * c_f) if age <= buy_age else 0
                if h_val > 0:
                    h_avg_val = h_val * (1 + h_appr / 2)
                    h_cost = h_avg_val * h_maint_total
            else:
                y_rent = rent_today * 12 * c_f

            all_exp[s, i] = y_exp + y_rent + h_cost

            if buy_house_scenario and age == buy_age:
                h_nom = h_price_today * c_f
                txn_fee = h_nom * buy_txn_pct
                down_pay = h_nom * 0.2
                liq -= (down_pay + txn_fee)
                h_val, m_bal = h_nom, h_nom * 0.8
                m_pay = (-npf.pmt(m_rate / 12, m_years * 12, m_bal)) * 12

            net_cf = y_inc - (y_exp + y_rent + h_cost) - m_pay
            liq = (liq * (1 + y_ret if liq > 0 else 0.06)) + net_cf

            if h_val > 0:
                h_val *= (1 + h_appr)
                m_bal = max(0, m_bal * (1 + m_rate / 12) ** 12 - m_pay)
                if age >= buy_age + m_years: m_pay = 0

            all_liq[s, i], all_nw[s, i], all_pp[s, i] = liq, liq + h_val - m_bal, liq / c_f

    return ages, all_liq, all_nw, all_pp, all_exp


# --- 3. 模擬執行與數據處理 ---
ages, l_buy, n_buy, p_buy, e_buy = simulate(True)
_, l_rent, n_rent, p_rent, e_rent = simulate(False)


def get_stats(liq, nw, pp, exp):
    ret_idx = retire_age - current_age
    success = np.mean(np.all(liq[:, ret_idx:] > (exp[:, ret_idx:] * fail_buffer), axis=1)) * 100
    return success, np.median(pp[:, -1]), np.median(nw[:, -1])


s_b, pp_b, nw_b = get_stats(l_buy, n_buy, p_buy, e_buy)
s_r, pp_r, nw_r = get_stats(l_rent, n_rent, p_rent, e_rent)

# --- 4. UI 數據總覽表 ---
st.subheader("🏁 專業對決修正版總覽")
st.table(pd.DataFrame({
    "決策指標": ["退休成功率 (含緩衝)", "85歲實質購買力 (中位數)", "85歲名目身價 (中位數)"],
    "買房 (Buy)": [f"{s_b:.1f}%", f"${pp_b / 10000:,.0f} 萬", f"${nw_b / 10000:,.0f} 萬"],
    "租屋 (Rent)": [f"{s_r:.1f}%", f"${pp_r / 10000:,.0f} 萬", f"${nw_r / 10000:,.0f} 萬"]
}))

# --- 5. 圖表區：動態顯示模式 ---
st.write("---")
chart_mode = st.radio("圖表顯示模式", ["簡潔模式 (僅中位數)", "風險模式 (顯示 10-90 分位區間)"], horizontal=True)


def plot_compare(data_b, data_r, title, ylabel):
    fig = go.Figure()

    # 提取統計量
    med_b, p10_b, p90_b = np.percentile(data_b, [50, 10, 90], axis=0)
    med_r, p10_r, p90_r = np.percentile(data_r, [50, 10, 90], axis=0)

    # 如果是風險模式，先畫陰影帶 (層級在下)
    if "風險" in chart_mode:
        # 買房情境區間 (綠色系)
        fig.add_trace(go.Scatter(x=ages, y=p90_b, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=ages, y=p10_b, fill='tonexty', fillcolor='rgba(0,209,178,0.15)', line=dict(width=0),
                                 name="買房風險區間"))

        # 租屋情境區間 (紅色系)
        fig.add_trace(go.Scatter(x=ages, y=p90_r, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=ages, y=p10_r, fill='tonexty', fillcolor='rgba(255,56,96,0.15)', line=dict(width=0),
                                 name="租屋風險區間"))

    # 畫中位數線 (層級在上)
    fig.add_trace(go.Scatter(x=ages, y=med_b, name="買房：預期中位數", line=dict(color='#00d1b2', width=4)))
    fig.add_trace(go.Scatter(x=ages, y=med_r, name="租屋：預期中位數", line=dict(color='#ff3860', width=4, dash='dash')))

    fig.update_layout(title=title, template="plotly_dark", hovermode="x unified", yaxis_title=ylabel)
    return fig


st.plotly_chart(plot_compare(p_buy, p_rent, "實質流動購買力對比 (今日價值)", "TWD (折現)"), use_container_width=True)
st.caption(
    "💡 **如何閱讀此圖？** 陰影區域代表 10–90 分位風險區間。在 90% 的模擬情境下，你的資產會落在該範圍內。區間越寬，代表該決策在極端市場環境下的不確定性越大。")

st.plotly_chart(plot_compare(n_buy, n_rent, "名目總淨資產對比 (身價走勢)", "TWD (名目)"), use_container_width=True)

# --- 6. 自動決策洞察 ---
st.write("---")
st.subheader("💡 決策分析洞察")
s_diff = s_b - s_r

if abs(s_diff) < 5:
    res_text = "⚖️ **兩者財務成功率相近。** 決策重點應放在生活型態的偏好。"
elif s_diff > 5:
    res_text = f"🏠 **買房情境較具優勢。** 成功率高出 {s_diff:.1f}%。"
else:
    res_text = f"🏢 **租屋情境較具優勢。** 成功率高出 {-s_diff:.1f}%。"

st.info(res_text)

if pp_b < pp_r and nw_b > nw_r:
    st.warning("⚠️ **淨資產陷阱警告**：買房身價雖高，但手頭可用購買力低於租屋，需注意老年流動性。")

with st.expander("📘 方法論摘要"):
    st.markdown("""
    - **蒙地卡羅模擬**：透過 200 次隨機路徑模擬市場與通膨波動。
    - **10-90 分位**：排除前後 10% 的極端運氣，呈現最可能的 80% 發生範圍。
    - **實質/名目**：實質購買力已扣除通膨；名目淨資產包含房屋市值的增長。
    """)
