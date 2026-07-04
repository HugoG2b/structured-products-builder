from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.instruments import (
    Underlying, ZeroCouponBond, EuropeanCall, EuropeanPut,
    DigitalCall, DigitalPut, BarrierOption,
)
from src.portfolio import Portfolio
from src.presets import PRESETS, PRESET_DESCRIPTIONS
from src.autocall import (
    AutocallSpec, simulate_events, autocall_greeks, simulate_display_paths,
    ATHENA, PHOENIX,
)


# ---------- Theme constants ----------
CLR = {
    "text": "#dce3ee", "muted": "#7d8898",
    "amber": "#f0a93b", "blue": "#5aa2ff", "green": "#2fd585",
    "red": "#ff5d6c", "purple": "#9b8cff", "grid": "#1a212e",
    "border": "#222837",
}
LEG_COLORS = ["#5aa2ff", "#9b8cff", "#4dd0c4", "#c7a0ff", "#7da7d9", "#e08baf"]

FREQ_PER_YEAR = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4, "Monthly": 12}


# ---------- Page config ----------
st.set_page_config(
    page_title="Structured Products Desk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def style_fig(fig, height=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#a8b3c4", size=12),
        height=height, margin=dict(l=30, r=20, t=30, b=40),
        xaxis=dict(gridcolor=CLR["grid"], zerolinecolor=CLR["border"]),
        yaxis=dict(gridcolor=CLR["grid"], zerolinecolor=CLR["border"]),
        hoverlabel=dict(font_family="JetBrains Mono, monospace"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    )
    return fig


def section(label, color=""):
    st.markdown(f"<div class='section-header {color}'>{label}</div>", unsafe_allow_html=True)


# ---------- Cached autocall computations ----------
@st.cache_data(show_spinner=False)
def ac_events(style, maturity, n_obs, trigger, stepdown, cb, prot, memory,
              S, vol, r, n_paths):
    spec = AutocallSpec(style=style, maturity=maturity, n_obs=n_obs,
                        ac_trigger=trigger, stepdown=stepdown,
                        coupon_barrier=cb, protection=prot, memory=memory)
    return simulate_events(spec, S, vol, r, n_paths=n_paths)


@st.cache_data(show_spinner=False)
def ac_greeks_cached(style, maturity, n_obs, trigger, stepdown, cb, prot, memory,
                     S, vol, r, coupon, n_paths):
    spec = AutocallSpec(style=style, maturity=maturity, n_obs=n_obs,
                        ac_trigger=trigger, stepdown=stepdown,
                        coupon_barrier=cb, protection=prot, memory=memory)
    return autocall_greeks(spec, S, vol, r, coupon, n_paths=n_paths)


@st.cache_data(show_spinner=False)
def ac_display_paths(style, maturity, n_obs, trigger, stepdown, cb, prot, memory,
                     S, vol, r, n_display):
    spec = AutocallSpec(style=style, maturity=maturity, n_obs=n_obs,
                        ac_trigger=trigger, stepdown=stepdown,
                        coupon_barrier=cb, protection=prot, memory=memory)
    return simulate_display_paths(spec, S, vol, r, n_display=n_display)


# ---------- Session state ----------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio()
if "last_preset" not in st.session_state:
    st.session_state.last_preset = None

AC_DEFAULTS = {
    "ac_style": "Athena", "ac_mat": 5.0, "ac_freq": "Annual",
    "ac_trigger": 100.0, "ac_stepdown": 0.0, "ac_cb": 70.0,
    "ac_prot": 60.0, "ac_memory": True, "ac_coupon": 5.0,
}
# Re-assigning each key marks it as user-owned, so Streamlit does not garbage-
# collect the widget state when the Autocall workspace is not rendered.
for k, v in AC_DEFAULTS.items():
    st.session_state[k] = st.session_state.get(k, v)

# Apply a coupon solved on the previous run (widget state can only be set
# before the widget is instantiated).
if "ac_pending_coupon" in st.session_state:
    st.session_state.ac_coupon = st.session_state.pop("ac_pending_coupon")
    st.session_state.ac_show_solved = True


def apply_ac_preset(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.session_state.pop("ac_show_solved", None)


# ---------- Utility: put-call parity sanity check ----------
def sanity_put_call_parity(S, vol, r, T, portfolio):
    calls = [i for i in portfolio.instruments if isinstance(i, EuropeanCall)]
    puts = [i for i in portfolio.instruments if isinstance(i, EuropeanPut)]
    for c in calls:
        for p in puts:
            if abs(c.strike - p.strike) < 1e-6 and abs(c.quantity + p.quantity) < 1e-6:
                theoretical = c.quantity * (S - c.strike * np.exp(-r * T))
                actual = c.price(S, vol, r, T) + p.price(S, vol, r, T)
                return actual - theoretical
    return None


# ---------- Header ----------
st.markdown(
    "<div class='desk-header'>"
    "<div class='desk-title'>STRUCTURED PRODUCTS <span class='accent'>DESK</span></div>"
    "<div class='desk-badge'>BS CLOSED-FORM · MONTE CARLO</div>"
    "</div>"
    "<div class='desk-sub'>Build, price and risk-analyze structured products — "
    "vanilla bricks, barriers, and autocallable notes (Athena / Phoenix).</div>",
    unsafe_allow_html=True,
)


# ---------- Sidebar ----------
with st.sidebar:
    section("Workspace")
    workspace = st.radio(
        "Workspace",
        ["Brick Builder", "Autocall Pricer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    if workspace == "Brick Builder":
        section("Presets")
        preset_keys = list(PRESETS.keys())
        preset_choice = st.selectbox("Load a preset structure", preset_keys, index=0)

        if preset_choice in PRESET_DESCRIPTIONS:
            st.markdown(
                f"<div class='preset-desc'>{PRESET_DESCRIPTIONS[preset_choice]}</div>",
                unsafe_allow_html=True,
            )

        load_col, clear_col = st.columns(2)
        with load_col:
            load_clicked = st.button("Load", use_container_width=True, type="primary")
        with clear_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.portfolio.clear()
                st.session_state.last_preset = None
                st.rerun()
    else:
        section("Product templates")
        if st.button("Classic Athena — 5Y annual", use_container_width=True):
            apply_ac_preset(ac_style="Athena", ac_mat=5.0, ac_freq="Annual",
                            ac_trigger=100.0, ac_stepdown=0.0, ac_prot=60.0,
                            ac_coupon=8.0)
            st.rerun()
        if st.button("Athena step-down — 6Y semi-annual", use_container_width=True):
            apply_ac_preset(ac_style="Athena", ac_mat=6.0, ac_freq="Semi-annual",
                            ac_trigger=100.0, ac_stepdown=2.0, ac_prot=60.0,
                            ac_coupon=4.0)
            st.rerun()
        if st.button("Phoenix memory — 3Y quarterly", use_container_width=True):
            apply_ac_preset(ac_style="Phoenix", ac_mat=3.0, ac_freq="Quarterly",
                            ac_trigger=100.0, ac_stepdown=0.0, ac_cb=70.0,
                            ac_prot=60.0, ac_memory=True, ac_coupon=2.0)
            st.rerun()
        if st.button("Defensive Phoenix — 5Y semi-annual", use_container_width=True):
            apply_ac_preset(ac_style="Phoenix", ac_mat=5.0, ac_freq="Semi-annual",
                            ac_trigger=90.0, ac_stepdown=0.0, ac_cb=60.0,
                            ac_prot=50.0, ac_memory=True, ac_coupon=3.0)
            st.rerun()

    st.markdown("---")
    section("Model notes", "blue")
    if workspace == "Brick Builder":
        st.caption(
            "Vanilla & digital legs: Black-Scholes closed form. "
            "Barrier legs: Monte Carlo (GBM, continuous monitoring). "
            "Flat vol, no dividends — a learning tool, not a production pricer."
        )
    else:
        st.caption(
            "Underlying sampled **exactly** at observation dates (discrete "
            "triggers, European protection barrier — market standard). "
            "Events are coupon-independent, so price is *linear in coupon*: "
            "the fair coupon is solved in closed form from one simulation."
        )


# ---------- Market inputs ----------
section("Market")
if workspace == "Brick Builder":
    c1, c2, c3, c4, c5 = st.columns(5)
else:
    c1, c2, c3, c5 = st.columns(4)
with c1:
    S = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0, format="%.2f")
with c2:
    vol_pct = st.number_input("Volatility (%)", min_value=0.1, max_value=200.0, value=25.0, step=1.0)
    vol = vol_pct / 100.0
with c3:
    r_pct = st.number_input("Rate (%)", min_value=-5.0, max_value=20.0, value=4.0, step=0.25)
    r = r_pct / 100.0
if workspace == "Brick Builder":
    with c4:
        T = st.number_input("Maturity (years)", min_value=0.01, max_value=30.0, value=1.0, step=0.25)
with c5:
    n_paths = st.number_input("MC paths", min_value=1000, max_value=200000, value=20000, step=1000)

st.markdown("---")

# Deferred preset load (needs S from the market bar)
if workspace == "Brick Builder" and load_clicked:
    builder = PRESETS.get(preset_choice)
    if builder is not None:
        st.session_state.portfolio.clear()
        for inst in builder(spot=S):
            st.session_state.portfolio.add(inst)
        st.session_state.last_preset = preset_choice
        st.rerun()


# ============================================================
# WORKSPACE 1 — BRICK BUILDER
# ============================================================
if workspace == "Brick Builder":
    build_col, analyze_col = st.columns([1, 2], gap="large")

    with build_col:
        section("Builder")

        with st.expander("➕  Add a brick", expanded=True):
            brick_type = st.selectbox(
                "Brick type",
                ["Underlying", "Zero-Coupon Bond", "European Call", "European Put",
                 "Digital Call", "Digital Put", "Barrier Option"],
                key="brick_type_select",
            )

            qty = st.number_input("Quantity (+ long / − short)", value=1.0, step=0.5, key="qty_input")

            if brick_type == "Underlying":
                st.caption("Spot underlying held until maturity. Delta = 1, no other Greeks.")
                if st.button("Add to portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio.add(Underlying(quantity=qty))
                    st.rerun()

            elif brick_type == "Zero-Coupon Bond":
                notional = st.number_input("Notional", value=100.0, step=10.0, key="zcb_notional")
                if st.button("Add to portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio.add(ZeroCouponBond(quantity=qty, notional=notional))
                    st.rerun()

            elif brick_type in ("European Call", "European Put"):
                K = st.number_input("Strike", value=float(S), step=1.0, key="euro_strike")
                if st.button("Add to portfolio", type="primary", use_container_width=True):
                    inst = EuropeanCall(quantity=qty, strike=K) if brick_type == "European Call" \
                        else EuropeanPut(quantity=qty, strike=K)
                    st.session_state.portfolio.add(inst)
                    st.rerun()

            elif brick_type in ("Digital Call", "Digital Put"):
                K = st.number_input("Strike", value=float(S), step=1.0, key="dig_strike")
                cash = st.number_input("Cash payout", value=1.0, step=0.5, key="dig_cash")
                if st.button("Add to portfolio", type="primary", use_container_width=True):
                    inst = DigitalCall(quantity=qty, strike=K, cash=cash) if brick_type == "Digital Call" \
                        else DigitalPut(quantity=qty, strike=K, cash=cash)
                    st.session_state.portfolio.add(inst)
                    st.rerun()

            elif brick_type == "Barrier Option":
                opt_type = st.selectbox("Option type", ["call", "put"], key="bar_opt")
                K = st.number_input("Strike", value=float(S), step=1.0, key="bar_strike")
                B = st.number_input("Barrier level", value=float(S) * 1.2, step=1.0, key="bar_level")
                bar_type = st.selectbox(
                    "Barrier type",
                    ["up-and-out", "down-and-out", "up-and-in", "down-and-in"],
                    key="bar_type",
                )
                rebate = st.number_input("Rebate on KO (if applicable)", value=0.0, step=0.5, key="bar_rebate")
                if st.button("Add to portfolio", type="primary", use_container_width=True):
                    st.session_state.portfolio.add(BarrierOption(
                        quantity=qty, strike=K, barrier=B,
                        option_type=opt_type, barrier_type=bar_type, rebate=rebate,
                    ))
                    st.rerun()

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        section("Current legs")

        if st.session_state.portfolio.is_empty():
            st.info("No bricks yet. Add one above or load a preset from the sidebar.")
        else:
            for i, inst in enumerate(list(st.session_state.portfolio.instruments)):
                is_short = inst.quantity < 0
                klass = "leg-card short" if is_short else "leg-card"
                tag = "SHORT" if is_short else "LONG"
                c_desc, c_btn = st.columns([5, 1])
                with c_desc:
                    st.markdown(
                        f"<div class='{klass}'><span>{inst.describe()}</span>"
                        f"<span class='leg-tag'>{tag}</span></div>",
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if st.button("✕", key=f"remove_{i}", help="Remove this leg"):
                        st.session_state.portfolio.remove(i)
                        st.rerun()

    with analyze_col:
        section("Analysis")

        tab_payoff, tab_pricing, tab_greeks, tab_sensi = st.tabs(
            ["Payoff", "Pricing", "Greeks", "Sensitivities"]
        )

        # -------- Payoff tab --------
        with tab_payoff:
            if st.session_state.portfolio.is_empty():
                st.info("Add at least one brick to see the payoff diagram.")
            else:
                s_min = max(0.01, S * 0.4)
                s_max = S * 1.6
                S_grid = np.linspace(s_min, s_max, 300)
                total_payoff = st.session_state.portfolio.payoff_at_maturity(S_grid)

                show_legs = st.checkbox("Show individual legs", value=False)

                fig = go.Figure()
                gain_y = np.where(total_payoff > 0, total_payoff, 0)
                loss_y = np.where(total_payoff < 0, total_payoff, 0)
                fig.add_trace(go.Scatter(
                    x=S_grid, y=gain_y, fill="tozeroy", mode="none",
                    fillcolor="rgba(47, 213, 133, 0.10)", hoverinfo="skip", showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=S_grid, y=loss_y, fill="tozeroy", mode="none",
                    fillcolor="rgba(255, 93, 108, 0.10)", hoverinfo="skip", showlegend=False,
                ))

                if show_legs:
                    for i, (desc, leg_p) in enumerate(st.session_state.portfolio.leg_payoffs(S_grid).items()):
                        fig.add_trace(go.Scatter(
                            x=S_grid, y=leg_p, mode="lines", name=desc,
                            line=dict(width=1.2, color=LEG_COLORS[i % len(LEG_COLORS)], dash="dot"),
                            opacity=0.75,
                        ))

                fig.add_trace(go.Scatter(
                    x=S_grid, y=total_payoff, mode="lines",
                    name="Total payoff",
                    line=dict(color=CLR["amber"], width=3),
                    hovertemplate="Spot: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>",
                ))

                # Strike / barrier markers
                strikes = sorted({inst.strike for inst in st.session_state.portfolio.instruments
                                  if hasattr(inst, "strike") and s_min < inst.strike < s_max})
                barriers = sorted({inst.barrier for inst in st.session_state.portfolio.instruments
                                   if hasattr(inst, "barrier") and s_min < inst.barrier < s_max})
                for k in strikes:
                    fig.add_vline(x=k, line=dict(color="#3a4459", width=1, dash="dot"),
                                  annotation_text=f"K {k:g}",
                                  annotation_font=dict(size=10, color=CLR["muted"]),
                                  annotation_position="bottom")
                for b in barriers:
                    fig.add_vline(x=b, line=dict(color=CLR["red"], width=1, dash="dot"),
                                  annotation_text=f"B {b:g}",
                                  annotation_font=dict(size=10, color=CLR["red"]),
                                  annotation_position="bottom")

                fig.add_vline(x=S, line=dict(color=CLR["muted"], width=1, dash="dash"),
                              annotation_text=f"Spot {S:.2f}",
                              annotation_font=dict(size=10, color=CLR["muted"]),
                              annotation_position="top")
                fig.add_hline(y=0, line=dict(color="#2c3445", width=1))

                style_fig(fig, height=460)
                fig.update_layout(
                    xaxis_title="Underlying at maturity", yaxis_title="Payoff",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                if st.session_state.portfolio.has_path_dependent():
                    st.caption(
                        "⚠️ Portfolio contains path-dependent instruments (barriers). "
                        "The diagram shows payoff conditional on being alive at maturity. "
                        "True expected value is computed in the Pricing tab via Monte Carlo."
                    )

        # -------- Pricing tab --------
        with tab_pricing:
            if st.session_state.portfolio.is_empty():
                st.info("Add at least one brick to price the structure.")
            else:
                with st.spinner("Pricing..."):
                    price = st.session_state.portfolio.price(S, vol, r, T, n_paths=n_paths)
                    breakdown = st.session_state.portfolio.breakdown(S, vol, r, T, n_paths=n_paths)

                k1, k2, k3 = st.columns(3)
                k1.metric("Fair Value (t=0)", f"{price:.4f}")
                k2.metric("Discount factor", f"{np.exp(-r * T):.4f}")
                k3.metric("Underlying", f"{S:.2f}")

                st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                section("Leg breakdown")
                st.dataframe(
                    breakdown.style.format({"Price": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                section("Sanity checks")
                cpp = sanity_put_call_parity(S, vol, r, T, st.session_state.portfolio)
                if cpp is not None:
                    st.caption(f"Put-Call parity check on ATM pair: residual = **{cpp:.6f}** (should be ≈ 0)")
                else:
                    st.caption("Put-Call parity check: N/A for this portfolio (no matching ATM call+put pair).")

        # -------- Greeks tab --------
        with tab_greeks:
            if st.session_state.portfolio.is_empty():
                st.info("Add at least one brick to compute Greeks.")
            else:
                with st.spinner("Computing Greeks..."):
                    g = st.session_state.portfolio.greeks(S, vol, r, T, n_paths=n_paths)

                gc = st.columns(5)
                gc[0].metric("Delta", f"{g['delta']:+.3f}", help="∂P/∂S — directional exposure to underlying")
                gc[1].metric("Gamma", f"{g['gamma']:+.3f}", help="∂²P/∂S² — convexity of the position")
                gc[2].metric("Vega", f"{g['vega']:+.3f}", help="∂P/∂σ per 1 vol pt (1%) — vol sensitivity")
                gc[3].metric("Theta", f"{g['theta']:+.3f}", help="Per-day P&L from time decay")
                gc[4].metric("Rho", f"{g['rho']:+.3f}", help="∂P/∂r per 1 rate pt (1%) — rate sensitivity")

                st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
                section("Delta profile")

                has_barriers = st.session_state.portfolio.has_path_dependent()
                grid_size = 15 if has_barriers else 35
                paths_for_profile = max(n_paths // 10, 2000) if has_barriers else n_paths

                S_grid = np.linspace(max(0.01, S * 0.6), S * 1.4, grid_size)
                deltas = []
                progress = st.progress(0.0) if has_barriers else None
                for idx, s_val in enumerate(S_grid):
                    g_s = st.session_state.portfolio.greeks(s_val, vol, r, T, n_paths=paths_for_profile)
                    deltas.append(g_s["delta"])
                    if progress is not None:
                        progress.progress((idx + 1) / len(S_grid))
                if progress is not None:
                    progress.empty()

                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=S_grid, y=deltas, mode="lines+markers",
                    line=dict(color=CLR["blue"], width=2),
                    marker=dict(size=5),
                    hovertemplate="Spot: %{x:.2f}<br>Delta: %{y:+.4f}<extra></extra>",
                ))
                fig_d.add_vline(x=S, line=dict(color=CLR["muted"], width=1, dash="dash"))
                fig_d.add_hline(y=0, line=dict(color="#2c3445", width=1))
                style_fig(fig_d, height=320)
                fig_d.update_layout(xaxis_title="Spot", yaxis_title="Portfolio Delta", showlegend=False)
                st.plotly_chart(fig_d, use_container_width=True)

        # -------- Sensitivities tab --------
        with tab_sensi:
            if st.session_state.portfolio.is_empty():
                st.info("Add at least one brick to explore sensitivities.")
            else:
                st.caption("Heatmap: fair value as a function of spot × volatility, holding other inputs fixed.")

                s_range_pct = st.slider("Spot range (% around current)", 10, 60, 30, 5, key="sensi_srange")
                v_range_pct = st.slider("Vol range (± vol points)", 5, 40, 15, 5, key="sensi_vrange")

                has_barriers = st.session_state.portfolio.has_path_dependent()
                n_spots = 10 if has_barriers else 20
                n_vols = 8 if has_barriers else 14

                spots = np.linspace(S * (1 - s_range_pct / 100), S * (1 + s_range_pct / 100), n_spots)
                vols = np.linspace(max(0.01, vol - v_range_pct / 100), vol + v_range_pct / 100, n_vols)

                if has_barriers:
                    st.caption("⏳ Portfolio contains barriers — heatmap uses Monte Carlo with reduced grid for speed.")

                Z = np.zeros((len(vols), len(spots)))
                paths_fast = max(n_paths // 6, 3000) if has_barriers else n_paths
                progress = st.progress(0.0)
                total = len(vols) * len(spots)
                done = 0
                for i, v_ in enumerate(vols):
                    for j, s_ in enumerate(spots):
                        Z[i, j] = st.session_state.portfolio.price(s_, v_, r, T, n_paths=paths_fast)
                        done += 1
                        progress.progress(done / total)
                progress.empty()

                fig_h = go.Figure(data=go.Heatmap(
                    z=Z, x=spots, y=vols * 100,
                    colorscale="Viridis",
                    colorbar=dict(title="Fair Value"),
                    hovertemplate="Spot: %{x:.2f}<br>Vol: %{y:.1f}%<br>Price: %{z:.3f}<extra></extra>",
                ))
                style_fig(fig_h, height=420)
                fig_h.update_layout(xaxis_title="Spot", yaxis_title="Volatility (%)")
                st.plotly_chart(fig_h, use_container_width=True)


# ============================================================
# WORKSPACE 2 — AUTOCALL PRICER
# ============================================================
else:
    param_col, out_col = st.columns([1, 2.4], gap="large")

    # ----- Product parameters -----
    with param_col:
        section("Product terms")

        style = st.radio("Structure", ["Athena", "Phoenix"], horizontal=True, key="ac_style")
        is_phoenix = style == "Phoenix"

        mat = st.number_input("Maturity (years)", min_value=0.5, max_value=12.0,
                              step=0.5, key="ac_mat")
        freq = st.selectbox("Observation frequency", list(FREQ_PER_YEAR.keys()), key="ac_freq")
        per_year = FREQ_PER_YEAR[freq]
        n_obs = max(1, int(round(mat * per_year)))

        trigger_pct = st.number_input("Autocall trigger (% of initial)", min_value=10.0,
                                      max_value=200.0, step=1.0, key="ac_trigger")
        stepdown_pct = st.number_input("Trigger step-down per period (%)", min_value=0.0,
                                       max_value=10.0, step=0.25, key="ac_stepdown",
                                       help="Athena step-down: the autocall trigger decreases "
                                            "at each observation, raising early-redemption odds over time.")
        if is_phoenix:
            cb_pct = st.number_input("Coupon barrier (% of initial)", min_value=10.0,
                                     max_value=200.0, step=1.0, key="ac_cb")
            memory = st.toggle("Memory coupons", key="ac_memory",
                               help="Missed coupons are recovered at the next observation "
                                    "where the underlying is back above the coupon barrier.")
        else:
            cb_pct = st.session_state.ac_cb
            memory = st.session_state.ac_memory
            st.caption("Athena: coupons accrue and are paid only at autocall "
                       "(snowball) — no separate coupon barrier.")

        prot_pct = st.number_input("Protection barrier (% of initial)", min_value=1.0,
                                   max_value=100.0, step=1.0, key="ac_prot",
                                   help="European barrier, observed at maturity only. Below it, "
                                        "capital redemption is S_T / S_0 (full downside).")

        st.markdown("---")
        coupon = st.number_input("Coupon per period (% of notional)", min_value=0.0,
                                 max_value=50.0, step=0.05, format="%.3f", key="ac_coupon")
        st.caption(f"= **{coupon * per_year:.2f}% p.a.** · {n_obs} observation"
                   f"{'s' if n_obs > 1 else ''} over {mat:g}y")

        solve_clicked = st.button("⚡ Solve fair coupon (price = 100%)",
                                  type="primary", use_container_width=True)

    # ----- Simulation (cached) -----
    args = (style.lower(), float(mat), int(n_obs), trigger_pct / 100, stepdown_pct / 100,
            cb_pct / 100, prot_pct / 100, bool(memory), float(S), float(vol), float(r),
            int(n_paths))
    with st.spinner("Simulating..."):
        ev = ac_events(*args)

    if solve_clicked:
        c_star = ev.solve_coupon(100.0)
        st.session_state.ac_pending_coupon = float(max(0.0, round(c_star, 3)))
        st.rerun()

    # ----- Output tabs -----
    with out_col:
        if st.session_state.pop("ac_show_solved", False):
            st.markdown(
                f"<div class='solve-banner'>Fair coupon solved: "
                f"<span class='big'>{coupon:.3f}%</span> per period "
                f"(<span class='big'>{coupon * per_year:.2f}%</span> p.a.) "
                f"— the note now prices at par.</div>",
                unsafe_allow_html=True,
            )

        tab_price, tab_sim, tab_risk, tab_ts = st.tabs(
            ["Pricing", "Simulation", "Risk profile", "Term sheet"]
        )

        price = ev.price(coupon)
        se = ev.stderr(coupon)
        fair_c = ev.solve_coupon(100.0)
        dist = ev.redemption_distribution()
        obs_t = ev.obs_times

        # -------- Pricing tab --------
        with tab_price:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fair value", f"{price:.2f}%",
                      delta=f"{price - 100:+.2f} vs par", delta_color="normal",
                      help=f"Monte Carlo 95% CI: ± {1.96 * se:.2f}")
            m2.metric("PV coupon leg", f"{ev.B * coupon:.2f}%",
                      help="Expected discounted coupons. Linear in the coupon: B × c.")
            m3.metric("PV capital leg", f"{ev.A:.2f}%",
                      help="Expected discounted capital redemption (autocall or maturity).")
            m4.metric("Fair coupon", f"{fair_c:.3f}%",
                      help=f"Coupon per period pricing the note at par "
                           f"= {fair_c * per_year:.2f}% p.a. Solved in closed form: "
                           f"c* = (100 − A) / B.")

            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            section("Greeks")
            with st.spinner("Bumping..."):
                gk = ac_greeks_cached(*args[:11], float(coupon), int(n_paths))
            gc = st.columns(5)
            gc[0].metric("Delta", f"{gk['delta']:+.3f}",
                         help="Pts of notional per +1% spot move. Positive: the note is long the market.")
            gc[1].metric("Gamma", f"{gk['gamma']:+.3f}",
                         help="Delta change per +1% spot move. Typically negative near barriers.")
            gc[2].metric("Vega", f"{gk['vega']:+.3f}",
                         help="Pts per +1 vol point. Negative: the holder is short downside vol "
                              "(short the embedded put).")
            gc[3].metric("Theta", f"{gk['theta']:+.4f}",
                         help="Pts per calendar day.")
            gc[4].metric("Rho", f"{gk['rho']:+.3f}",
                         help="Pts per +100bp rate move.")

            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            section("How this is priced")
            st.caption(
                f"One Monte Carlo pass ({n_paths:,} paths) records, per path, the redemption "
                f"date, the capital redeemed and the number of coupon units paid — none of "
                f"which depend on the coupon level. The PV is therefore exactly linear in the "
                f"coupon: **price(c) = A + B·c** with A = {ev.A:.2f} and B = {ev.B:.4f}. "
                f"The fair coupon is closed-form — no root-finding, no extra simulation."
            )

        # -------- Simulation tab --------
        with tab_sim:
            t_grid, paths, _, fate = ac_display_paths(*args[:11], 140)

            fig = go.Figure()
            groups = [
                ("Autocalled early", fate >= 0, CLR["green"]),
                ("Redeemed at maturity", fate == -1, CLR["blue"]),
                ("Capital loss", fate == -2, CLR["red"]),
            ]
            for name, mask, color in groups:
                sel = paths[mask]
                if len(sel) == 0:
                    continue
                xs, ys = [], []
                for row in sel:
                    xs.extend(t_grid); xs.append(None)
                    ys.extend(row); ys.append(None)
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=0.8), opacity=0.45,
                    name=f"{name} ({mask.sum()})", hoverinfo="skip",
                ))

            # Observation dates
            for t_o in obs_t[:-1]:
                fig.add_vline(x=t_o, line=dict(color="#222837", width=1))
            fig.add_vline(x=obs_t[-1], line=dict(color="#2c3445", width=1, dash="dash"))

            # Trigger levels at each observation
            triggers_abs = (trigger_pct / 100 - np.arange(n_obs) * stepdown_pct / 100) * S
            fig.add_trace(go.Scatter(
                x=obs_t, y=triggers_abs, mode="lines+markers",
                line=dict(color=CLR["amber"], width=1.5, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
                name="Autocall trigger",
                hovertemplate="t=%{x:.2f}y · trigger %{y:.1f}<extra></extra>",
            ))
            if is_phoenix:
                fig.add_hline(y=cb_pct / 100 * S,
                              line=dict(color=CLR["blue"], width=1.2, dash="dot"),
                              annotation_text=f"Coupon barrier {cb_pct:g}%",
                              annotation_font=dict(size=10, color=CLR["blue"]))
            fig.add_hline(y=prot_pct / 100 * S,
                          line=dict(color=CLR["red"], width=1.2, dash="dash"),
                          annotation_text=f"Protection {prot_pct:g}% (at maturity)",
                          annotation_font=dict(size=10, color=CLR["red"]),
                          annotation_position="bottom right")

            style_fig(fig, height=500)
            fig.update_layout(
                xaxis_title="Time (years)", yaxis_title="Underlying",
                xaxis=dict(range=[0, mat * 1.02]),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "140 sample paths, colored by outcome. Paths stop at their redemption date — "
                "diamonds mark the autocall trigger at each observation. The protection "
                "barrier is European: only the final fixing matters for capital."
            )

        # -------- Risk tab --------
        with tab_risk:
            ls = ev.loss_stats()
            exp_coupons = float((ev.units.sum(axis=1)).mean()) * coupon

            r1 = st.columns(5)
            r1[0].metric("P(call)", f"{dist['p_autocall_by_date'].sum():.1%}",
                         help="Probability the note autocalls before maturity.")
            r1[1].metric("E[life]", f"{ev.expected_life():.2f}y",
                         help=f"Expected product life vs contractual maturity {mat:g}y.")
            r1[2].metric("P(loss)", f"{ls['p_loss']:.1%}",
                         help="Probability of finishing below the protection barrier (capital loss).")
            r1[3].metric("Avg loss", f"−{ls['avg_loss']:.1f}%",
                         help="Average capital loss conditional on finishing below the protection barrier.")
            r1[4].metric("E[cpns]", f"{exp_coupons:.2f}%",
                         help="Expected total coupons over the product life, undiscounted.")

            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            section("Redemption distribution")

            labels = [f"AC @ {t:.2g}y" for t in obs_t[:-1]] + \
                     ["Maturity · redeemed", "Maturity · loss"]
            values = list(dist["p_autocall_by_date"][:-1]) + \
                     [dist["p_autocall_by_date"][-1] + dist["p_redeemed_at_maturity"],
                      dist["p_capital_loss"]]
            colors = [CLR["green"]] * (n_obs - 1) + [CLR["blue"], CLR["red"]]

            fig_b = go.Figure(go.Bar(
                x=labels, y=values, marker_color=colors,
                text=[f"{v:.1%}" for v in values], textposition="outside",
                textfont=dict(family="JetBrains Mono", size=11),
                hovertemplate="%{x}: %{y:.2%}<extra></extra>",
            ))
            style_fig(fig_b, height=340)
            fig_b.update_layout(yaxis_tickformat=".0%", showlegend=False,
                                yaxis_range=[0, max(values) * 1.25])
            st.plotly_chart(fig_b, use_container_width=True)

            section("Distribution of discounted P&L")
            pv = ev.per_path_pv(coupon)
            var95 = float(np.percentile(pv, 5))
            fig_h = go.Figure(go.Histogram(
                x=pv, nbinsx=70,
                marker=dict(color="rgba(240, 169, 59, 0.55)",
                            line=dict(color="#0a0d12", width=0.5)),
                hovertemplate="PV: %{x:.1f}<br>Paths: %{y}<extra></extra>",
            ))
            fig_h.add_vline(x=float(pv.mean()), line=dict(color=CLR["amber"], width=2),
                            annotation_text=f"mean {pv.mean():.1f}",
                            annotation_font=dict(size=10, color=CLR["amber"]))
            fig_h.add_vline(x=var95, line=dict(color=CLR["red"], width=2, dash="dash"),
                            annotation_text=f"VaR 95% → {var95:.1f}",
                            annotation_font=dict(size=10, color=CLR["red"]),
                            annotation_position="top left")
            style_fig(fig_h, height=320)
            fig_h.update_layout(xaxis_title="Discounted payoff (per 100 notional)",
                                yaxis_title="Paths", showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)
            st.caption(
                "Histogram of discounted per-path payoff (capital + coupons). The left tail is "
                "the down-and-in capital loss; the mass right of par is early autocalls with coupons."
            )

        # -------- Term sheet tab --------
        with tab_ts:
            issue = pd.Timestamp(date.today())
            months = int(12 / per_year)
            obs_dates = [issue + pd.DateOffset(months=months * (i + 1)) for i in range(n_obs)]
            triggers_pct_sched = [trigger_pct - i * stepdown_pct for i in range(n_obs)]

            sched_rows = "".join(
                f"<tr><td class='num'>{i+1}</td>"
                f"<td class='num'>{d.strftime('%d %b %Y')}</td>"
                f"<td class='num'>{tp:.1f}% ({tp / 100 * S:.2f})</td>"
                f"<td>{'Early redemption: 100% + ' + (f'{(i+1)} × {coupon:.3f}%' if style == 'Athena' else f'{coupon:.3f}%') if i < n_obs - 1 else 'Final observation'}</td></tr>"
                for i, (d, tp) in enumerate(zip(obs_dates, triggers_pct_sched))
            )

            if style == "Athena":
                coupon_terms = (
                    f"<tr><td class='k'>Coupon (snowball)</td><td class='v'>{coupon:.3f}% per period "
                    f"({coupon * per_year:.2f}% p.a.) — accrued coupons paid in full at early "
                    f"redemption, or at maturity if the final fixing is at/above the final trigger</td></tr>"
                )
                mat_redemption = (
                    f"<li>Final fixing ≥ final autocall trigger ({triggers_pct_sched[-1]:.1f}%): "
                    f"redemption at <b>100% + {n_obs} × {coupon:.3f}%</b></li>"
                    f"<li>Final fixing ≥ protection barrier ({prot_pct:g}%): redemption at <b>100%</b> (no coupon)</li>"
                )
            else:
                memory_txt = "with memory (missed coupons recovered)" if memory else "without memory"
                coupon_terms = (
                    f"<tr><td class='k'>Conditional coupon</td><td class='v'>{coupon:.3f}% per period "
                    f"({coupon * per_year:.2f}% p.a.), paid if the fixing is at/above the coupon "
                    f"barrier ({cb_pct:g}%), {memory_txt}</td></tr>"
                    f"<tr><td class='k'>Coupon barrier</td><td class='v'>{cb_pct:g}% of initial "
                    f"({cb_pct / 100 * S:.2f})</td></tr>"
                )
                mat_redemption = (
                    f"<li>Final fixing ≥ protection barrier ({prot_pct:g}%): redemption at <b>100%</b> "
                    f"(+ final coupon if ≥ coupon barrier)</li>"
                )

            ts_body = f"""
<h2>{style.upper()} AUTOCALLABLE NOTE</h2>
<div class='ts-sub'>Indicative term sheet · {issue.strftime('%d %B %Y')} · For educational purposes only — not an offer</div>

<h3>General terms</h3>
<table>
<tr><td class='k'>Issuer</td><td class='v'>Demo Issuer (educational)</td></tr>
<tr><td class='k'>Underlying</td><td class='v'>Generic Equity Index (initial fixing {S:.2f})</td></tr>
<tr><td class='k'>Notional</td><td class='v'>EUR 1,000 per note</td></tr>
<tr><td class='k'>Issue price</td><td class='v'>100.00%</td></tr>
<tr><td class='k'>Indicative fair value</td><td class='v'>{price:.2f}% (flat vol {vol_pct:g}%, rate {r_pct:g}%)</td></tr>
<tr><td class='k'>Maturity</td><td class='v'>{mat:g} years — {obs_dates[-1].strftime('%d %b %Y')}</td></tr>
<tr><td class='k'>Observations</td><td class='v'>{freq} ({n_obs} dates)</td></tr>
{coupon_terms}
<tr><td class='k'>Autocall trigger</td><td class='v'>{trigger_pct:g}% of initial{f' − {stepdown_pct:g}% step-down per period' if stepdown_pct > 0 else ''}</td></tr>
<tr><td class='k'>Protection barrier</td><td class='v'>{prot_pct:g}% of initial ({prot_pct / 100 * S:.2f}), European — observed at maturity only</td></tr>
</table>

<h3>Observation schedule</h3>
<table>
<tr><th>#</th><th>Date</th><th>Autocall trigger</th><th>If fixing ≥ trigger</th></tr>
{sched_rows}
</table>

<h3>Redemption at maturity (if never autocalled)</h3>
<ul>
{mat_redemption}
<li>Final fixing &lt; protection barrier: redemption at <b>final fixing / initial fixing</b> — the
holder bears the full downside (e.g. final at 50% of initial → 50% of notional, a −50% capital loss)</li>
</ul>

<h3>Model risk metrics (Monte Carlo, {n_paths:,} paths)</h3>
<table>
<tr><td class='k'>Probability of early autocall</td><td class='v'>{dist['p_autocall_by_date'].sum():.1%}</td></tr>
<tr><td class='k'>Expected life</td><td class='v'>{ev.expected_life():.2f} years</td></tr>
<tr><td class='k'>Probability of capital loss</td><td class='v'>{ev.loss_stats()['p_loss']:.1%}</td></tr>
<tr><td class='k'>Fair coupon (par)</td><td class='v'>{fair_c:.3f}% per period ({fair_c * per_year:.2f}% p.a.)</td></tr>
</table>

<div class='disclaimer'>Educational pricer: flat volatility (no smile), Geometric Brownian Motion,
no dividends/repo, no credit/funding spread, no fees. Real secondary-market prices will differ.
This document is not investment advice and does not constitute an offer.</div>
"""

            st.markdown(f"<div class='ts-doc'>{ts_body}</div>", unsafe_allow_html=True)

            download_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{style} Autocallable Note — Term Sheet</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; color: #1a1a1a; max-width: 820px;
       margin: 40px auto; line-height: 1.5; font-size: 14px; }}
h2 {{ letter-spacing: 0.08em; border-bottom: 3px solid #b8860b; padding-bottom: 8px; }}
h3 {{ text-transform: uppercase; letter-spacing: 0.1em; font-size: 12px; color: #b8860b; margin-top: 26px; }}
.ts-sub {{ color: #777; font-size: 12px; margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
td, th {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #f4f0e6; font-size: 11px; text-transform: uppercase; }}
td.k {{ color: #666; width: 38%; }}
.disclaimer {{ margin-top: 24px; padding-top: 12px; border-top: 1px solid #ccc;
              color: #888; font-size: 11px; font-style: italic; }}
</style></head><body>{ts_body}</body></html>"""

            st.download_button(
                "⬇ Download term sheet (HTML)",
                data=download_html,
                file_name=f"{style.lower()}_term_sheet_{issue.strftime('%Y%m%d')}.html",
                mime="text/html",
            )
