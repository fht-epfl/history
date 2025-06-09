import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
import math

class CooccurrenceGraphVisualizer:
    def __init__(self, label_sentence_count, label_cooccurrence_graph):
        """
        Initialize the visualizer with your co-occurrence data
        
        Args:
            label_sentence_count: dict mapping label -> count of sentences
            label_cooccurrence_graph: dict mapping (label1, label2) -> co-occurrence count
        """
        self.label_sentence_count = label_sentence_count
        self.label_cooccurrence_graph = label_cooccurrence_graph
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def create_networkx_graph(self, selected_labels, min_edge_weight=1):
        """Create NetworkX graph from selected labels and minimum edge weight"""
        G = nx.Graph()
        
        # Add nodes with size based on sentence count
        for label in selected_labels:
            if label in self.label_sentence_count:
                G.add_node(label, size=self.label_sentence_count[label])
        
        # Add edges with weight based on co-occurrence
        for (label1, label2), weight in self.label_cooccurrence_graph.items():
            if (label1 in selected_labels and label2 in selected_labels and 
                weight >= min_edge_weight):
                G.add_edge(label1, label2, weight=weight)
        
        return G
    
    def create_plotly_graph(self, G, layout_type='spring', node_size_scale=1.0):
        """Convert NetworkX graph to Plotly visualization"""
        if len(G.nodes()) == 0:
            # Return empty graph
            return go.Figure().add_annotation(
                text="No nodes selected or no connections found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
        
        # Choose layout algorithm with better spacing
        if layout_type == 'spring':
            # Use node sizes to influence layout spacing
            node_sizes = [G.nodes[node].get('size', 1) for node in G.nodes()]
            max_size = max(node_sizes) if node_sizes else 1
            k_value = max(2.0, math.sqrt(len(G.nodes())) * 0.5)  # Dynamic spacing
            pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G, scale=2)
        elif layout_type == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(G, scale=2)
            except:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Get edge weights for width calculation
        edge_weights = [G[edge[0]][edge[1]]['weight'] for edge in G.edges()]
        
        # Create individual edge traces with varying widths
        edge_traces = []
        if edge_weights:
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            
            # Create edge traces with different widths and opacity based on strength
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G[edge[0]][edge[1]]['weight']
                
                # Normalize edge width (1-8 pixels)
                if max_weight > min_weight:
                    normalized_width = 1 + (weight - min_weight) / (max_weight - min_weight) * 7
                else:
                    normalized_width = 3
                
                # Normalize opacity (0.3-0.8)
                normalized_opacity = 0.3 + (weight - min_weight) / (max_weight - min_weight + 1e-6) * 0.5
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=normalized_width, color=f'rgba(100,100,100,{normalized_opacity})'),
                    hoverinfo='text',
                    hovertext=f"{edge[0]} ↔ {edge[1]}<br>Co-occurrence: {weight}",
                    mode='lines',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Create node trace with size representing frequency
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_hover_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node size based on sentence count
            size = G.nodes[node].get('size', 1)
            node_sizes.append(size)
            
            # Adjust text based on node size for better visibility
            if size > np.percentile([G.nodes[n].get('size', 1) for n in G.nodes()], 75):
                node_text.append(node)  # Show text for large nodes
            else:
                node_text.append('')  # Hide text for small nodes to reduce clutter
            
            node_hover_text.append(f"{node}<br>Frequency: {size}<br>Connections: {len(list(G.neighbors(node)))}")
        
        # Normalize node sizes for better visual distinction (10-60 pixels)
        if node_sizes:
            max_size = max(node_sizes)
            min_size = min(node_sizes)
            if max_size > min_size:
                normalized_sizes = [10 + (s - min_size) / (max_size - min_size) * 50 * node_size_scale for s in node_sizes]
            else:
                normalized_sizes = [30 * node_size_scale] * len(node_sizes)
        else:
            normalized_sizes = [30 * node_size_scale]
        
        # Use a categorical color scheme for better distinction
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        node_colors = [colors[i % len(colors)] for i in range(len(node_sizes))]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=8, color="black", family="Arial Black"),
            marker=dict(
                color=node_colors,
                size=normalized_sizes,
                line=dict(width=2, color="white"),
                opacity=0.8
            ),
            hovertext=node_hover_text,
            showlegend=False
        )
        
        # Combine all traces
        all_traces = edge_traces + [node_trace]
        
        # Create figure with improved layout
        fig = go.Figure(data=all_traces,
                       layout=go.Layout(
                           title=dict(
                               text='Co-occurrence Graph<br><sub>Node size = frequency, Edge width = connection strength</sub>',
                               x=0.5,
                               font=dict(size=18)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=40,l=40,r=40,t=80),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           paper_bgcolor='white',
                           font=dict(size=12)
                       ))
        
        # Adjust layout range to prevent node overlap
        if pos:
            x_values = [pos[node][0] for node in pos]
            y_values = [pos[node][1] for node in pos]
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            
            # Add padding based on largest node size
            max_node_size = max(normalized_sizes) if normalized_sizes else 30
            padding = max_node_size * 0.002  # Scale padding with node size
            
            fig.update_xaxes(range=[min(x_values) - x_range*0.1 - padding, 
                                   max(x_values) + x_range*0.1 + padding])
            fig.update_yaxes(range=[min(y_values) - y_range*0.1 - padding, 
                                   max(y_values) + y_range*0.1 + padding])
        
        return fig
    
    def setup_layout(self):
        """Setup the Dash layout"""
        # Get all available labels
        all_labels = list(self.label_sentence_count.keys())
        
        self.app.layout = html.Div([
            html.H1("Interactive Co-occurrence Graph", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.Label("Select Labels to Display:", 
                              style={'fontWeight': 'bold', 'marginBottom': 10}),
                    html.Div([
                        dcc.Checklist(
                            id='label-selector',
                            options=[{'label': label, 'value': label} for label in sorted(all_labels)],
                            value=sorted(all_labels)[:10],  # Default to first 10 labels
                            style={'marginBottom': 15, 'maxHeight': '300px', 'overflowY': 'auto'},
                            labelStyle={'display': 'block', 'marginBottom': '5px', 'fontSize': '12px'},
                            inputStyle={'marginRight': '8px'}
                        )
                    ], style={'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px', 
                             'backgroundColor': '#f9f9f9', 'maxHeight': '320px', 'overflowY': 'auto'}),
                    
                    html.Div([
                        html.Button('Select All', id='select-all-btn', n_clicks=0,
                                   style={'marginRight': '10px', 'padding': '5px 10px', 
                                         'backgroundColor': '#007bff', 'color': 'white', 
                                         'border': 'none', 'borderRadius': '3px', 'cursor': 'pointer'}),
                        html.Button('Clear All', id='clear-all-btn', n_clicks=0,
                                   style={'marginRight': '10px', 'padding': '5px 10px',
                                         'backgroundColor': '#dc3545', 'color': 'white',
                                         'border': 'none', 'borderRadius': '3px', 'cursor': 'pointer'}),
                        html.Button('Top 10', id='top-10-btn', n_clicks=0,
                                   style={'padding': '5px 10px', 'backgroundColor': '#28a745', 
                                         'color': 'white', 'border': 'none', 'borderRadius': '3px', 
                                         'cursor': 'pointer'})
                    ], style={'marginTop': '10px', 'marginBottom': '15px'}),
                    
                    html.Label("Minimum Edge Weight:", 
                              style={'fontWeight': 'bold', 'marginBottom': 10}),
                    dcc.Slider(
                        id='edge-weight-slider',
                        min=1,
                        max=max(self.label_cooccurrence_graph.values()) if self.label_cooccurrence_graph else 10,
                        value=1,
                        marks={i: str(i) for i in range(1, min(11, max(self.label_cooccurrence_graph.values()) + 1))},
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=1
                    ),
                    
                    html.Label("Node Size Scale:", 
                              style={'fontWeight': 'bold', 'marginTop': 15, 'marginBottom': 10}),
                    dcc.Slider(
                        id='node-size-scale',
                        min=0.5,
                        max=2.0,
                        value=1.0,
                        marks={0.5: '0.5x', 1.0: '1x', 1.5: '1.5x', 2.0: '2x'},
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=0.1
                    ),
                    
                    html.Label("Layout Algorithm:", 
                              style={'fontWeight': 'bold', 'marginTop': 20, 'marginBottom': 10}),
                    dcc.RadioItems(
                        id='layout-selector',
                        options=[
                            {'label': 'Spring Layout', 'value': 'spring'},
                            {'label': 'Circular Layout', 'value': 'circular'},
                            {'label': 'Kamada-Kawai Layout', 'value': 'kamada_kawai'}
                        ],
                        value='spring',
                        labelStyle={'display': 'block', 'marginBottom': 5}
                    ),
                    
                    html.Div(id='graph-stats', 
                            style={'marginTop': 20, 'padding': 10, 'backgroundColor': '#f0f0f0', 'borderRadius': 5})
                    
                ], style={'width': '25%', 'padding': 20, 'verticalAlign': 'top', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='cooccurrence-graph', style={'height': '80vh'})
                ], style={'width': '75%', 'display': 'inline-block'})
                
            ], style={'display': 'flex'})
        ])
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        # Callback for button functionality
        @self.app.callback(
            Output('label-selector', 'value'),
            [Input('select-all-btn', 'n_clicks'),
             Input('clear-all-btn', 'n_clicks'),
             Input('top-10-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def update_label_selection(select_all_clicks, clear_all_clicks, top_10_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            all_labels = sorted(list(self.label_sentence_count.keys()))
            
            if button_id == 'select-all-btn':
                return all_labels
            elif button_id == 'clear-all-btn':
                return []
            elif button_id == 'top-10-btn':
                # Get top 10 labels by frequency
                top_labels = sorted(self.label_sentence_count.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
                return [label for label, count in top_labels]
            
            return dash.no_update
        
        @self.app.callback(
            [Output('cooccurrence-graph', 'figure'),
             Output('graph-stats', 'children')],
            [Input('label-selector', 'value'),
             Input('edge-weight-slider', 'value'),
             Input('layout-selector', 'value'),
             Input('node-size-scale', 'value')]
        )
        def update_graph(selected_labels, min_edge_weight, layout_type, node_size_scale):
            if not selected_labels:
                empty_fig = go.Figure().add_annotation(
                    text="Please select at least one label",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font_size=16
                )
                return empty_fig, "No labels selected"
            
            # Create graph
            G = self.create_networkx_graph(selected_labels, min_edge_weight)
            fig = self.create_plotly_graph(G, layout_type, node_size_scale)
            
            # Create enhanced stats
            if G.nodes():
                node_sizes = [G.nodes[node].get('size', 1) for node in G.nodes()]
                edge_weights = [G[edge[0]][edge[1]]['weight'] for edge in G.edges()]
                
                stats = html.Div([
                    html.H4("Graph Statistics", style={'marginBottom': 15}),
                    html.P([html.Strong("Nodes: "), f"{len(G.nodes())}"]),
                    html.P([html.Strong("Edges: "), f"{len(G.edges())}"]),
                    html.P([html.Strong("Avg Degree: "), f"{np.mean([d for n, d in G.degree()]):.2f}"]),
                    html.P([html.Strong("Density: "), f"{nx.density(G):.3f}"]),
                    html.Hr(),
                    html.P([html.Strong("Node Frequency Range: "), f"{min(node_sizes)} - {max(node_sizes)}"]),
                    html.P([html.Strong("Edge Weight Range: "), f"{min(edge_weights) if edge_weights else 0} - {max(edge_weights) if edge_weights else 0}"]),
                ])
            else:
                stats = html.Div([
                    html.H4("Graph Statistics"),
                    html.P("No connections found with current settings"),
                ])
            
            return fig, stats
    
    def run(self, debug=True, port=8057):
        """Run the Dash app"""
        self.app.run("0.0.0.0", port=port, debug=debug, use_reloader=False)

# Example usage with your data
if __name__ == "__main__":
    # Example data structure - replace with your actual data
    # Assuming you have your label_sentence_count and label_cooccurrence_graph from your code
    import pickle
    
    # read pickle
    with open("../proc/label_sentence_count_gudu.pkl", "rb") as f:
        label_sentence_count = pickle.load(f)
    with open("../proc/label_cooccurrence_graph_gudu.pkl", "rb") as f:
        label_cooccurrence_graph = pickle.load(f)

    # Initialize and run the visualizer
    visualizer = CooccurrenceGraphVisualizer(
        label_sentence_count, 
        label_cooccurrence_graph
    )
    
    print("Starting the co-occurrence graph visualizer...")
    print("Open http://localhost:8052 in your browser to view the interactive graph")
    
    # To use with your actual data, replace the example data above with:
    # visualizer = CooccurrenceGraphVisualizer(label_sentence_count, label_cooccurrence_graph)
    
    visualizer.run()