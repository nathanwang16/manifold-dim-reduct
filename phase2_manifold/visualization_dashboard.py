"""
Phase 2.3: Visualization Dashboard

Interactive dashboard for manifold visualization including:
1. 2D scatter plots colored by label
2. Cluster centroids and convex hulls
3. Method comparison views
4. Hierarchical clustering dendrogram
5. Silhouette score visualization

Usage:
    python visualization_dashboard.py --embeddings embeddings/ --analysis analysis/

Runs on http://localhost:8050
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input

# Add parent directory to path for logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import get_logger, LogTimer, configure_logging

# Initialize logger
logger = get_logger(__name__)

# Color palette for 18 classes (colorblind-friendly)
COLORS_18 = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1'
]


def load_embeddings(embeddings_dir: Path) -> dict:
    """Load all embedding files from directory."""
    embeddings = {}
    for f in embeddings_dir.glob("*.npy"):
        if 'labels' not in f.name and 'indices' not in f.name and 'variance' not in f.name:
            embeddings[f.stem] = np.load(f)
    logger.info(f"Loaded {len(embeddings)} embedding files")
    return embeddings


def load_analysis_results(analysis_dir: Path) -> dict:
    """Load cluster analysis results."""
    results_file = analysis_dir / "cluster_analysis_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    logger.warning(f"Analysis results not found at {results_file}")
    return {}


def create_scatter_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    show_centroids: bool = True,
    show_hulls: bool = False
) -> go.Figure:
    """
    Create 2D scatter plot colored by label.

    Args:
        embeddings: 2D embeddings (n_samples, 2)
        labels: Class labels
        title: Plot title
        show_centroids: Whether to show class centroids
        show_hulls: Whether to show convex hulls

    Returns:
        Plotly Figure object
    """
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'label': labels.astype(str)
    })

    fig = go.Figure()

    unique_labels = sorted(df['label'].unique(), key=lambda x: int(x))

    for i, label in enumerate(unique_labels):
        mask = df['label'] == label
        points = df[mask]

        # Add scatter points
        fig.add_trace(go.Scatter(
            x=points['x'],
            y=points['y'],
            mode='markers',
            name=f'Label {label}',
            marker=dict(
                size=4,
                color=COLORS_18[int(label) - 1],
                opacity=0.6
            ),
            hovertemplate=f'Label {label}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>'
        ))

        # Add convex hull
        if show_hulls and len(points) >= 3:
            try:
                hull_points = points[['x', 'y']].values
                hull = ConvexHull(hull_points)
                hull_x = hull_points[hull.vertices, 0].tolist() + [hull_points[hull.vertices[0], 0]]
                hull_y = hull_points[hull.vertices, 1].tolist() + [hull_points[hull.vertices[0], 1]]

                fig.add_trace(go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode='lines',
                    line=dict(color=COLORS_18[int(label) - 1], width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            except Exception:
                pass  # Skip if hull fails

        # Add centroid
        if show_centroids:
            centroid_x = points['x'].mean()
            centroid_y = points['y'].mean()
            fig.add_trace(go.Scatter(
                x=[centroid_x],
                y=[centroid_y],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=COLORS_18[int(label) - 1],
                    symbol='x',
                    line=dict(width=2, color='black')
                ),
                text=[label],
                textposition='top center',
                showlegend=False,
                hovertemplate=f'Centroid {label}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>'
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        height=600,
        template='plotly_white'
    )

    return fig


def create_silhouette_plot(analysis_results: dict, method_name: str) -> go.Figure:
    """Create bar chart of per-class silhouette scores."""
    if method_name not in analysis_results:
        return go.Figure()

    silhouette_data = analysis_results[method_name].get('silhouette', {})
    per_class = silhouette_data.get('per_class', {})

    if not per_class:
        return go.Figure()

    labels = sorted(per_class.keys(), key=int)
    scores = [per_class[l] for l in labels]
    colors = [COLORS_18[int(l) - 1] for l in labels]

    fig = go.Figure(go.Bar(
        x=[f'L{l}' for l in labels],
        y=scores,
        marker_color=colors
    ))

    overall = silhouette_data.get('overall', 0)
    fig.add_hline(
        y=overall,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Overall: {overall:.3f}'
    )

    fig.update_layout(
        title=f'Silhouette Scores by Class ({method_name})',
        xaxis_title='Label',
        yaxis_title='Silhouette Score',
        height=400,
        template='plotly_white'
    )

    return fig


def create_distance_heatmap(analysis_dir: Path, method_name: str) -> go.Figure:
    """Create heatmap of pairwise label distances."""
    dist_file = analysis_dir / f"{method_name}_distance_matrix.npy"

    if not dist_file.exists():
        return go.Figure()

    distance_matrix = np.load(dist_file)
    labels = [str(i) for i in range(1, 19)]

    fig = go.Figure(go.Heatmap(
        z=distance_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        reversescale=True,
        hovertemplate='Label %{x} - Label %{y}<br>Distance: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Pairwise Centroid Distances ({method_name})',
        xaxis_title='Label',
        yaxis_title='Label',
        height=500,
        template='plotly_white'
    )

    return fig


def create_dendrogram(analysis_results: dict, method_name: str) -> go.Figure:
    """Create hierarchical clustering dendrogram."""
    if method_name not in analysis_results:
        return go.Figure()

    hier_data = analysis_results[method_name].get('hierarchical', {})
    linkage_matrix = hier_data.get('linkage', [])

    if not linkage_matrix:
        return go.Figure()

    # Compute dendrogram
    Z = np.array(linkage_matrix)
    labels = [str(i) for i in range(1, 19)]

    # Use scipy dendrogram to get coordinates
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    dend = scipy_dendrogram(Z, labels=labels, no_plot=True)

    # Create plotly figure
    fig = go.Figure()

    # Draw dendrogram lines
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])

    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            line=dict(color='#636EFA', width=1.5),
            showlegend=False
        ))

    # Add leaf labels
    for i, label in enumerate(dend['ivl']):
        fig.add_trace(go.Scatter(
            x=[5 + i * 10],
            y=[0],
            mode='text',
            text=[label],
            textposition='bottom center',
            showlegend=False
        ))

    fig.update_layout(
        title=f'Hierarchical Clustering of Labels ({method_name})',
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis_title='Distance',
        height=400,
        template='plotly_white'
    )

    return fig


def create_comparison_table(analysis_results: dict) -> go.Figure:
    """Create comparison table of all methods."""
    summary = analysis_results.get('comparison_summary', [])

    if not summary:
        return go.Figure()

    methods = [s['method'] for s in summary]
    silhouettes = [s['silhouette'] for s in summary]
    aris = [s['ari'] for s in summary]

    fig = go.Figure(data=[
        go.Bar(name='Silhouette', x=methods, y=silhouettes),
        go.Bar(name='ARI', x=methods, y=aris)
    ])

    fig.update_layout(
        title='Method Comparison: Silhouette Score & Adjusted Rand Index',
        barmode='group',
        xaxis_tickangle=-45,
        height=400,
        template='plotly_white'
    )

    return fig


def create_app(
    embeddings_dir: Path,
    analysis_dir: Path,
    labels: np.ndarray
) -> Dash:
    """
    Create the Dash visualization app.

    Args:
        embeddings_dir: Directory containing embeddings
        analysis_dir: Directory containing analysis results
        labels: Class labels

    Returns:
        Dash application
    """
    logger.info("Creating Dash application...")

    # Load data
    with LogTimer(logger, "Loading embeddings and analysis"):
        embeddings = load_embeddings(embeddings_dir)
        analysis_results = load_analysis_results(analysis_dir)

    method_options = [{'label': m, 'value': m} for m in embeddings.keys()]

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1('Phase 2: Manifold Visualization Dashboard',
                style={'textAlign': 'center', 'marginBottom': 30}),

        # Controls
        html.Div([
            html.Div([
                html.Label('Select Embedding Method:'),
                dcc.Dropdown(
                    id='method-dropdown',
                    options=method_options,
                    value=method_options[0]['value'] if method_options else None,
                    style={'width': '300px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 30}),

            html.Div([
                dcc.Checklist(
                    id='display-options',
                    options=[
                        {'label': ' Show Centroids', 'value': 'centroids'},
                        {'label': ' Show Convex Hulls', 'value': 'hulls'}
                    ],
                    value=['centroids'],
                    inline=True
                )
            ], style={'display': 'inline-block'})
        ], style={'marginBottom': 20, 'padding': 20, 'backgroundColor': '#f8f9fa'}),

        # Main scatter plot
        html.Div([
            dcc.Graph(id='main-scatter', style={'height': '600px'})
        ]),

        # Analysis row
        html.Div([
            html.Div([
                dcc.Graph(id='silhouette-plot')
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id='distance-heatmap')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),

        # Dendrogram and comparison
        html.Div([
            html.Div([
                dcc.Graph(id='dendrogram-plot')
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(id='comparison-plot', figure=create_comparison_table(analysis_results))
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),

        # Predicted confusions
        html.Div([
            html.H3('Predicted Confusion Pairs (based on manifold proximity)'),
            html.Div(id='confusion-table')
        ], style={'padding': 20})
    ])

    @callback(
        [Output('main-scatter', 'figure'),
         Output('silhouette-plot', 'figure'),
         Output('distance-heatmap', 'figure'),
         Output('dendrogram-plot', 'figure'),
         Output('confusion-table', 'children')],
        [Input('method-dropdown', 'value'),
         Input('display-options', 'value')]
    )
    def update_plots(method, display_options):
        if method is None or method not in embeddings:
            empty = go.Figure()
            return empty, empty, empty, empty, "Select a method"

        logger.debug(f"Updating plots for method: {method}")

        emb = embeddings[method]
        if emb.shape[1] > 2:
            emb = emb[:, :2]

        show_centroids = 'centroids' in (display_options or [])
        show_hulls = 'hulls' in (display_options or [])

        # Main scatter
        scatter_fig = create_scatter_plot(
            emb, labels, f'2D Embedding: {method}',
            show_centroids=show_centroids,
            show_hulls=show_hulls
        )

        # Silhouette
        silhouette_fig = create_silhouette_plot(analysis_results, method)

        # Distance heatmap
        heatmap_fig = create_distance_heatmap(analysis_dir, method)

        # Dendrogram
        dendrogram_fig = create_dendrogram(analysis_results, method)

        # Confusion table
        method_results = analysis_results.get(method, {})
        confusions = method_results.get('predicted_confusions', [])
        if confusions:
            rows = [
                html.Tr([
                    html.Td(f"Labels {c['label_1']} & {c['label_2']}"),
                    html.Td(f"{c['distance']:.4f}")
                ]) for c in confusions[:5]
            ]
            confusion_content = html.Table([
                html.Thead(html.Tr([html.Th('Label Pair'), html.Th('Distance')])),
                html.Tbody(rows)
            ], style={'margin': 'auto'})
        else:
            confusion_content = "No confusion data available"

        return scatter_fig, silhouette_fig, heatmap_fig, dendrogram_fig, confusion_content

    logger.info("Dash application created successfully")
    return app


def main():
    parser = argparse.ArgumentParser(description="Run visualization dashboard")
    parser.add_argument("--embeddings", type=str, default="phase2_manifold/embeddings",
                        help="Directory containing embeddings")
    parser.add_argument("--analysis", type=str, default="phase2_manifold/analysis",
                        help="Directory containing analysis results")
    parser.add_argument("--labels", type=str, default="phase2_manifold/embeddings/labels.npy",
                        help="Path to labels .npy file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    args = parser.parse_args()

    # Configure logging
    configure_logging(log_dir=args.log_dir)

    embeddings_dir = Path(args.embeddings)
    analysis_dir = Path(args.analysis)

    # Load labels
    with LogTimer(logger, f"Loading labels from {args.labels}"):
        labels = np.load(args.labels)
        logger.info(f"Loaded {len(labels)} labels")

    # Create and run app
    app = create_app(embeddings_dir, analysis_dir, labels)

    logger.info(f"Starting dashboard on http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
