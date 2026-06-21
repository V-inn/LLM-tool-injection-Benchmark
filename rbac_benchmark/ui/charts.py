"""
charts.py — Plotly figure builders for the dashboard.

Each function returns a configured figure; the tab modules own the surrounding
st.plotly_chart / layout. Extracted verbatim from the former monolithic gui_app.py so
behaviour (colours, ranges, hover text) is unchanged.
"""
import plotly.express as px
import plotly.graph_objects as go


def live_outcome_pie(outcomes: dict) -> go.Figure:
    """Donut chart of the live outcome distribution during a running benchmark."""
    fig = px.pie(
        names=list(outcomes.keys()),
        values=list(outcomes.values()),
        hole=0.4,
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


def lever_distribution_bar(model_lever_df, lever_cols) -> go.Figure:
    """Grouped bar chart of the average Cialdini manipulation-lever rates per model
    (Axis B). `lever_cols` are the percentage columns to plot."""
    fig = px.bar(
        model_lever_df,
        x="Model",
        y=lever_cols,
        barmode="group",
        title="Manipulation Lever Profile per Model (Cialdini — Axis B)",
        labels={"value": "Rate (%)", "variable": "Cialdini Lever"},
    )
    fig.update_layout(legend_title_text="Cialdini Lever")
    return fig


def awareness_stacked_bar(model_aware_df, awareness_cols) -> go.Figure:
    """Stacked bar chart of the awareness distribution per model (Axis A). Stacking
    makes the DETECTED_BUT_COMPLIED slice — the model knew yet acted — easy to compare
    across models."""
    fig = px.bar(
        model_aware_df,
        x="Model",
        y=awareness_cols,
        barmode="stack",
        title="Awareness Distribution per Model (Axis A)",
        labels={"value": "Rate (%)", "variable": "Awareness"},
    )
    fig.update_layout(legend_title_text="Awareness")
    return fig


def resilience_radar(model_df, radar_cols) -> go.Figure:
    """5-axis radar chart of per-model risk profile averaged across the matrix."""
    radar_fig = go.Figure()
    categories = radar_cols + [radar_cols[0]]  # Close the polygon back to the first axis
    for _, row in model_df.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in radar_cols] + [row[radar_cols[0]]],
            theta=categories,
            fill='toself',
            name=row["Model"],
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Model Risk Profile (avg across all defenses & attacks)",
    )
    return radar_fig


def delta_tpr_heatmap(delta_data: dict, ref_model: str) -> go.Figure:
    """
    Heatmap of ΔTPR (advanced defense − S1_BASELINE) per (attack, defense). Returns the
    figure built from the nested delta dict produced by analyzer.compute_delta_tpr.
    """
    attacks = sorted(delta_data.keys())
    defenses = sorted({d for atk in delta_data.values() for d in atk.keys()})

    z_vals = [
        [delta_data[atk].get(def_, {}).get("delta", 0.0) * 100 for def_ in defenses]
        for atk in attacks
    ]
    hover_texts = [
        [
            f"Attack: {atk}<br>Defense: {def_}<br>"
            f"ΔTPR: {delta_data[atk].get(def_, {}).get('delta', 0.0):+.1%}<br>"
            f"Baseline TPR: {delta_data[atk].get(def_, {}).get('tpr_baseline', 0.0):.1%}<br>"
            f"Compare TPR: {delta_data[atk].get(def_, {}).get('tpr_compare', 0.0):.1%}"
            for def_ in defenses
        ]
        for atk in attacks
    ]

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=defenses,
        y=attacks,
        text=[[f"{v:+.1f}%" for v in row] for row in z_vals],
        texttemplate="%{text}",
        hovertext=hover_texts,
        hoverinfo="text",
        colorscale=[
            [0.0,  "#b91c1c"],   # strong red   = negative ΔTPR (regression)
            [0.4,  "#fca5a5"],   # light red
            [0.5,  "#f1f5f9"],   # near-white   = no change
            [0.6,  "#86efac"],   # light green
            [1.0,  "#15803d"],   # strong green = positive ΔTPR (improvement)
        ],
        zmid=0,
        colorbar=dict(title="ΔTPR (%)"),
    ))
    heatmap_fig.update_layout(
        title=f"ΔTPR Heatmap — {ref_model} — marginal gain vs S1_BASELINE",
        xaxis_title="Defense Strategy",
        yaxis_title="Attack Payload",
        height=max(350, len(attacks) * 60 + 120),
    )
    return heatmap_fig


def confusion_matrix_heatmap(confusion: dict, categories: list) -> go.Figure:
    """Human-vs-Judge confusion matrix heatmap for the Phase-3 κ result."""
    z_vals = [[confusion[h][m] for m in categories] for h in categories]
    cm_fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=categories,
        y=categories,
        text=[[str(v) for v in row] for row in z_vals],
        texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate="Human: %{y}<br>Judge: %{x}<br>Count: %{z}<extra></extra>",
        colorbar=dict(title="Count"),
    ))
    cm_fig.update_layout(
        title="Human vs Judge Confusion Matrix",
        xaxis_title="Judge (machine) label",
        yaxis_title="Human label",
        height=420,
    )
    return cm_fig
