"""
visuals.py — Plotly figure factories for the dashboard.

Four figure types, all returning ``plotly.graph_objects.Figure`` objects so
app.py can render them with ``st.plotly_chart(fig, use_container_width=True)``.

Design notes:
* Colour palette matches the design guide: deep navy #0B2545, accent gold
  #C9A227, soft grey #EEF2F7.
* All figures are responsive (no hard-coded widths) and use ``margin=dict(l=..)``
  tuned for Streamlit's default tab panel.
* No Streamlit imports here — this module is UI-framework agnostic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import FUND_CODES
from engine import efficient_frontier, gmvp


# Brand palette
NAVY = "#0B2545"
MIDNIGHT = "#13315C"
GOLD = "#C9A227"
PAPER_BG = "#FFFFFF"
PLOT_BG = "#F7F9FC"
GRID = "#EEF2F7"
POS = "#2E8B57"
NEG = "#C0392B"
INFO = "#3498DB"

_FONT = dict(family="Inter, -apple-system, system-ui, sans-serif", color=NAVY, size=12)

_LAYOUT_COMMON = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=_FONT,
    margin=dict(l=40, r=20, t=48, b=40),
    hoverlabel=dict(bgcolor=NAVY, font_color="white"),
)


# ---------------------------------------------------------------------------
# 1. Risk profile radar
# ---------------------------------------------------------------------------

def plot_risk_radar(subscores: dict[str, int]) -> go.Figure:
    """Radar over the 5 risk-profile axes.

    Input ``subscores`` maps question category -> raw score. We fold them into
    5 axes: Age, Income/Capacity, Experience, Attitude, Horizon.
    """
    axes = {
        "Age & Capacity": ("age", "income", "investable_ratio"),
        "Experience":     ("experience_breadth", "experience_years"),
        "Attitude":       ("risk_attitude", "prospect_theory"),
        "Horizon":        ("horizon",),
        "Loss Tolerance": ("loss_tolerance", "objective"),
    }
    # Normalize each axis to 0..5 by averaging then scaling.
    values = []
    for _, cats in axes.items():
        vals = [subscores.get(c, 0) for c in cats]
        if not vals:
            values.append(0.0)
        else:
            # Rough normalization: score 1..5 -> 1..5 already; just average.
            values.append(sum(vals) / len(vals))
    labels = list(axes.keys()) + [list(axes.keys())[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels,
        fill="toself",
        fillcolor=f"rgba(201,162,39,0.35)",  # translucent gold
        line=dict(color=GOLD, width=2),
        name="Your Profile",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Risk Profile Radar", x=0.5, xanchor="center",
                   font=dict(size=16, color=NAVY)),
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(range=[0, 5], showline=False, gridcolor=GRID,
                            tickfont=dict(size=10, color=MIDNIGHT)),
            angularaxis=dict(tickfont=dict(size=11, color=NAVY)),
        ),
        showlegend=False,
        **_LAYOUT_COMMON,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Efficient frontier
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    user_point: tuple[float, float] | None = None,
    n_points: int = 40,
) -> go.Figure:
    """Scatter of individual funds + two frontier curves + GMVP + user pick."""
    fr_short = efficient_frontier(mu, sigma, n_points=n_points, allow_short=True)
    fr_long  = efficient_frontier(mu, sigma, n_points=n_points, allow_short=False)
    g_short  = gmvp(mu, sigma, allow_short=True)
    g_long   = gmvp(mu, sigma, allow_short=False)

    fig = go.Figure()

    # Individual funds as annotated markers
    stds = np.sqrt(np.diag(sigma))
    fig.add_trace(go.Scatter(
        x=stds, y=mu, mode="markers+text",
        text=FUND_CODES, textposition="top center",
        marker=dict(size=10, color=INFO, line=dict(color=NAVY, width=1)),
        name="Individual funds",
        hovertemplate="%{text}<br>sigma=%{x:.2%}<br>mu=%{y:.2%}<extra></extra>",
    ))

    # Frontier with shorts (dashed, lighter)
    fig.add_trace(go.Scatter(
        x=fr_short.stds, y=fr_short.target_returns, mode="lines",
        line=dict(color=MIDNIGHT, width=2, dash="dash"),
        name="Frontier (shorts allowed)",
    ))
    # Frontier long-only (solid, prominent)
    fig.add_trace(go.Scatter(
        x=fr_long.stds, y=fr_long.target_returns, mode="lines",
        line=dict(color=NAVY, width=3),
        name="Frontier (no shorts)",
    ))

    # GMVPs
    fig.add_trace(go.Scatter(
        x=[g_short.std], y=[g_short.expected_return],
        mode="markers",
        marker=dict(size=13, color=MIDNIGHT, symbol="diamond", line=dict(color="white", width=1)),
        name="GMVP (shorts ok)",
        hovertemplate="GMVP shorts<br>sigma=%{x:.2%}<br>mu=%{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[g_long.std], y=[g_long.expected_return],
        mode="markers",
        marker=dict(size=13, color=POS, symbol="diamond", line=dict(color="white", width=1)),
        name="GMVP (no shorts)",
        hovertemplate="GMVP long-only<br>sigma=%{x:.2%}<br>mu=%{y:.2%}<extra></extra>",
    ))

    # User's portfolio point (highlighted)
    if user_point:
        std_u, mu_u = user_point
        fig.add_trace(go.Scatter(
            x=[std_u], y=[mu_u], mode="markers",
            marker=dict(size=18, color=GOLD, symbol="star",
                        line=dict(color=NAVY, width=2)),
            name="Your portfolio",
            hovertemplate="Your portfolio<br>sigma=%{x:.2%}<br>mu=%{y:.2%}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Efficient Frontier (annualized)", x=0.5, xanchor="center",
                   font=dict(size=16)),
        xaxis=dict(title="Standard deviation (sigma)", tickformat=".0%",
                   gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title="Expected return (mu)", tickformat=".0%",
                   gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        **_LAYOUT_COMMON,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Weights pie
# ---------------------------------------------------------------------------

def plot_weights_pie(weights: dict[str, float], fund_names: dict[str, str] | None = None) -> go.Figure:
    items = [(k, v) for k, v in weights.items() if v >= 0.005]
    items.sort(key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fund_names = fund_names or {}

    hover_text = [
        f"<b>{k}</b><br>{fund_names.get(k, '')}<br>weight={v * 100:.2f}%"
        for k, v in items
    ]

    color_cycle = [GOLD, NAVY, MIDNIGHT, POS, INFO, "#8E44AD", "#E07A1F", "#C0392B", "#16A085", "#7F8C8D"]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.45,
        hovertext=hover_text, hoverinfo="text",
        textinfo="label+percent", textfont=dict(color="white", size=12),
        marker=dict(colors=color_cycle[:len(labels)],
                    line=dict(color="white", width=2)),
        sort=False,
    )])
    fig.update_layout(
        title=dict(text="Recommended allocation", x=0.5, xanchor="center",
                   font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        **_LAYOUT_COMMON,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Backtest curve
# ---------------------------------------------------------------------------

def plot_backtest(
    prices: pd.DataFrame,
    recommended_weights: dict[str, float],
    initial_capital: float = 10_000.0,
) -> go.Figure:
    """Compare: recommended portfolio vs equal-weight vs best-single-fund."""
    from engine import backtest as bt

    w_rec = np.array([recommended_weights.get(c, 0.0) for c in FUND_CODES])
    if w_rec.sum() > 0:
        w_rec = w_rec / w_rec.sum()
    w_eq = np.full(len(FUND_CODES), 1.0 / len(FUND_CODES))

    bt_rec = bt(w_rec.tolist(), prices[FUND_CODES], initial_capital)
    bt_eq = bt(w_eq.tolist(), prices[FUND_CODES], initial_capital)

    # Best single fund (highest final value)
    finals = prices.iloc[-1] / prices.iloc[0]
    best = finals.idxmax()
    w_best = np.zeros(len(FUND_CODES))
    w_best[FUND_CODES.index(best)] = 1.0
    bt_best = bt(w_best.tolist(), prices[FUND_CODES], initial_capital)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt_rec.index, y=bt_rec["value"],
        mode="lines", name="Your portfolio",
        line=dict(color=GOLD, width=3),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bt_eq.index, y=bt_eq["value"],
        mode="lines", name="Equal-weight",
        line=dict(color=MIDNIGHT, width=2, dash="dash"),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bt_best.index, y=bt_best["value"],
        mode="lines", name=f"Best single ({best})",
        line=dict(color=INFO, width=2, dash="dot"),
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Historical backtest (start S${initial_capital:,.0f})",
                   x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Period", gridcolor=GRID),
        yaxis=dict(title="Portfolio value", gridcolor=GRID, tickformat="$,.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        **_LAYOUT_COMMON,
    )
    return fig


__all__ = [
    "plot_risk_radar",
    "plot_efficient_frontier",
    "plot_weights_pie",
    "plot_backtest",
]
