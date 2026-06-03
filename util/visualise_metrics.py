"""
plot_results.py  —  Generate Plotly figures from NeuralRecon evaluation results.

Usage:
    python tools/plot_results.py --results results/results_metrics.jsonl --out results/plots
"""
import json
import argparse
import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ── colour palette ────────────────────────────────────────────────────────────
SCALE_COLORS = {
    'scale0': '#B4B2A9',
    'scale1': '#378ADD',
    'scale2': '#185FA5',
}
OUTLIER_COLOR  = '#D85A30'   # room2 or any anomalous scene
TEMPLATE       = 'plotly_white'

SCALE_LABELS = {
    'scale0': 'Scale 0 (coarsest, 16 cm)',
    'scale1': 'Scale 1 (8 cm)',
    'scale2': 'Scale 2 (finest, 4 cm)',
}


# ── load data ────────────────────────────────────────────────────────────────
def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    data = {}   # data[scene][scale] = metrics dict
    for r in records:
        parts = r['scene'].rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('scale0', 'scale1', 'scale2'):
            scene, scale = parts
        else:
            scene, scale = r['scene'], 'scale2'
        data.setdefault(scene, {})[scale] = r

    return data


# ── helpers ──────────────────────────────────────────────────────────────────
def bar_color(scene, scale):
    """Highlight room2 as the anomalous capture-quality scene."""
    if scene == 'room2' and scale == 'scale2':
        return OUTLIER_COLOR
    return SCALE_COLORS[scale]


# ── plot 1 : F-score @5cm, all scales, all scenes ────────────────────────────
def plot_fscore_ablation(data, out_dir):
    scenes = sorted(data.keys())
    scales = ['scale0', 'scale1', 'scale2']

    fig = go.Figure()
    for scale in scales:
        vals   = [data[s].get(scale, {}).get('fscore@5cm', 0) * 100 for s in scenes]
        colors = [bar_color(s, scale) for s in scenes]
        fig.add_trace(go.Bar(
            name=SCALE_LABELS[scale],
            x=scenes,
            y=vals,
            marker_color=colors if scale == 'scale2' else SCALE_COLORS[scale],
            text=[f'{v:.1f}%' for v in vals],
            textposition='outside',
            textfont_size=10,
        ))

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='F-score @5 cm — Ablation Study (all scales)', font_size=15),
        xaxis_title='Scene',
        yaxis_title='F-score @5 cm (%)',
        yaxis_range=[0, 65],
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family='Arial', size=12),
        height=450,
    )
    _save(fig, out_dir, 'fscore_ablation')


# ── plot 2 : Chamfer distance, all scales, all scenes ────────────────────────
def plot_chamfer_ablation(data, out_dir):
    scenes = sorted(data.keys())
    scales = ['scale0', 'scale1', 'scale2']

    fig = go.Figure()
    for scale in scales:
        vals = [data[s].get(scale, {}).get('chamfer', 0) for s in scenes]
        fig.add_trace(go.Bar(
            name=SCALE_LABELS[scale],
            x=scenes,
            y=vals,
            marker_color=SCALE_COLORS[scale],
            text=[f'{v:.3f}m' for v in vals],
            textposition='outside',
            textfont_size=10,
        ))

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Chamfer Distance — Ablation Study (all scales)', font_size=15),
        xaxis_title='Scene',
        yaxis_title='Chamfer distance (m)',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family='Arial', size=12),
        height=450,
    )
    _save(fig, out_dir, 'chamfer_ablation')


# ── plot 3 : Precision vs Recall @5cm (scale2 only) ──────────────────────────
def plot_precision_recall(data, out_dir):
    scenes = sorted(data.keys())
    scale  = 'scale2'

    precision = [data[s].get(scale, {}).get('precision@5cm', 0) * 100 for s in scenes]
    recall    = [data[s].get(scale, {}).get('recall@5cm',    0) * 100 for s in scenes]
    colors    = [OUTLIER_COLOR if s == 'room2' else '#185FA5' for s in scenes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Precision @5cm', x=scenes, y=precision,
        marker_color=colors, opacity=1.0,
        text=[f'{v:.1f}%' for v in precision], textposition='outside', textfont_size=10,
    ))
    fig.add_trace(go.Bar(
        name='Recall @5cm', x=scenes, y=recall,
        marker_color=colors, opacity=0.5,
        text=[f'{v:.1f}%' for v in recall], textposition='outside', textfont_size=10,
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Precision vs Recall @5 cm — Scale 2 (finest)', font_size=15),
        xaxis_title='Scene',
        yaxis_title='(%)',
        yaxis_range=[0, 65],
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family='Arial', size=12),
        height=430,
    )
    _save(fig, out_dir, 'precision_recall_scale2')


# ── plot 4 : Accuracy vs Completeness (scale2 only) ──────────────────────────
def plot_acc_comp(data, out_dir):
    scenes = sorted(data.keys())
    scale  = 'scale2'

    acc  = [data[s].get(scale, {}).get('accuracy',     0) for s in scenes]
    comp = [data[s].get(scale, {}).get('completeness', 0) for s in scenes]
    colors = [OUTLIER_COLOR if s == 'room2' else '#185FA5' for s in scenes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Accuracy (pred→GT)', x=scenes, y=acc,
        marker_color='#185FA5',
        text=[f'{v:.3f}m' for v in acc], textposition='outside', textfont_size=10,
    ))
    fig.add_trace(go.Bar(
        name='Completeness (GT→pred)', x=scenes, y=comp,
        marker_color='#F5C4B3',
        text=[f'{v:.3f}m' for v in comp], textposition='outside', textfont_size=10,
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Accuracy vs Completeness — Scale 2 (lower is better)', font_size=15),
        xaxis_title='Scene',
        yaxis_title='Distance (m)',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        font=dict(family='Arial', size=12),
        height=430,
    )
    _save(fig, out_dir, 'accuracy_completeness_scale2')


# ── plot 5 : Scale contribution — avg F@5cm across all scenes ────────────────
def plot_scale_contribution(data, out_dir):
    scales = ['scale0', 'scale1', 'scale2']
    avgs   = []
    for scale in scales:
        vals = [d.get(scale, {}).get('fscore@5cm', 0) * 100
                for d in data.values() if scale in d]
        avgs.append(np.mean(vals) if vals else 0)

    fig = go.Figure(go.Bar(
        x=[SCALE_LABELS[s] for s in scales],
        y=avgs,
        marker_color=[SCALE_COLORS[s] for s in scales],
        text=[f'{v:.1f}%' for v in avgs],
        textposition='outside',
        textfont_size=12,
        width=0.4,
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='Scale Contribution — Average F-score @5 cm Across All Scenes', font_size=15),
        xaxis_title='Reconstruction Scale',
        yaxis_title='Avg F-score @5 cm (%)',
        yaxis_range=[0, 55],
        font=dict(family='Arial', size=12),
        height=400,
        showlegend=False,
    )
    _save(fig, out_dir, 'scale_contribution_avg')


# ── plot 6 : Multi-threshold F-score for scale2 (heatmap style) ──────────────
def plot_threshold_heatmap(data, out_dir):
    scenes     = sorted(data.keys())
    thresholds = ['fscore@5cm', 'fscore@10cm', 'fscore@20cm']
    t_labels   = ['F@5cm', 'F@10cm', 'F@20cm']

    z = []
    for t in thresholds:
        row = [data[s].get('scale2', {}).get(t, 0) * 100 for s in scenes]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=scenes,
        y=t_labels,
        colorscale='Blues',
        text=[[f'{v:.1f}%' for v in row] for row in z],
        texttemplate='%{text}',
        textfont_size=12,
        colorbar=dict(title='F-score (%)'),
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text='F-score Heatmap — Scale 2, All Thresholds', font_size=15),
        xaxis_title='Scene',
        yaxis_title='Threshold',
        font=dict(family='Arial', size=12),
        height=320,
    )
    _save(fig, out_dir, 'fscore_heatmap_scale2')


# ── save helper ───────────────────────────────────────────────────────────────
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f'{name}.html')
    png_path  = os.path.join(out_dir, f'{name}.png')
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, scale=2, width=900, height=fig.layout.height)
        print(f'  saved: {png_path}  +  {html_path}')
    except Exception as e:
        print(f'  saved: {html_path}  (PNG skipped: {e})')


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/results_metrics.jsonl',
                        help='path to results JSONL file')
    parser.add_argument('--out', default='results/plots',
                        help='output directory for plots')
    args = parser.parse_args()

    print(f'Loading: {args.results}')
    data = load(args.results)
    print(f'Scenes found: {sorted(data.keys())}')

    print('\nGenerating plots...')
    plot_fscore_ablation(data, args.out)
    plot_chamfer_ablation(data, args.out)
    plot_precision_recall(data, args.out)
    plot_acc_comp(data, args.out)
    plot_scale_contribution(data, args.out)
    plot_threshold_heatmap(data, args.out)

    print(f'\nDone. All plots saved to: {args.out}/')