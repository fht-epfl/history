import dash
from dash import dcc, html, Input, Output
import pandas as pd
import re
import random

# Load data
df_books = pd.read_pickle("passive_voice.pkl")
df_imagery = pd.read_pickle("imagery_dictionary.pkl")

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Enhanced color generation for better visual distinction
def generate_enhanced_colors():
    """Generate distinct colors for big_labels and small_labels with better visual separation"""
    random.seed(42)
    
    # Big label colors - more saturated and distinct
    big_label_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#AED6F1", "#F8C471",
        "#BB8FCE", "#85C1E9", "#82E0AA", "#F9E79F", "#FADBD8"
    ]
    
    # Small label colors - lighter variations and pastels
    small_label_colors = [
        "#FFE0E0", "#E0F8F8", "#E0F2FF", "#E8F5E8", "#FFF8E0",
        "#F0E8FF", "#E8F8F5", "#FEF9E7", "#EBF5FB", "#FEF2E7",
        "#F4ECF7", "#D6EAF8", "#D5F4E6", "#FCF3CF", "#FADBD8",
        "#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFBA",
        "#FFB3FF", "#B3FFFF", "#E6E6FA", "#F0FFF0", "#FFF8DC"
    ]
    
    big_labels = sorted(df_imagery['big_label'].unique())
    small_labels = sorted(df_imagery['small_label'].unique())
    
    # Create color mappings
    big_color_map = {label: big_label_colors[i % len(big_label_colors)] 
                    for i, label in enumerate(big_labels)}
    small_color_map = {label: small_label_colors[i % len(small_label_colors)] 
                      for i, label in enumerate(small_labels)}
    
    # Combine both mappings
    combined_colors = {**big_color_map, **small_color_map}
    return combined_colors, big_color_map, small_color_map

label_colors, big_label_colors, small_label_colors = generate_enhanced_colors()

# Dropdown options - filter for 朱天心 books only
book_options = [{'label': f"{row['title']} - {row['year']} - {row['author']}", 'value': row['title']} 
               for _, row in df_books.iterrows() 
               if '朱天心' in row['author'] and row['year'] >= 1987]
# print(f"Available books: {book_options}")

# Enhanced Layout with better organization
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("文学文本意象词可视化系统", 
               style={
                   'textAlign': 'center', 
                   'color': '#2C3E50',
                   'marginBottom': '30px',
                   'fontFamily': 'Arial, sans-serif'
               })
    ]),
    
    # Main content container
    html.Div([
        # Left sidebar for controls
        html.Div([
            html.Div([
                html.H3("控制面板", style={
                    'color': '#34495E', 
                    'borderBottom': '2px solid #3498DB',
                    'paddingBottom': '10px',
                    'marginBottom': '20px'
                }),
                
                # Book selection
                html.Div([
                    html.Label("选择书籍", style={
                        'fontWeight': 'bold', 
                        'color': '#2C3E50',
                        'display': 'block',
                        'marginBottom': '8px'
                    }),
                    dcc.Dropdown(
                        id="book-selector", 
                        options=book_options, 
                        value=book_options[0]['value'] if book_options else None,
                        style={'marginBottom': '20px'}
                    ),
                ]),

                # Label selection with checkboxes
                html.Div([
                    html.Label("选择意象标签", style={
                        'fontWeight': 'bold', 
                        'color': '#2C3E50',
                        'display': 'block',
                        'marginBottom': '15px'
                    }),
                    html.Div(id="label-checklist-container", style={
                        'maxHeight': '400px',
                        'overflowY': 'auto',
                        'border': '1px solid #BDC3C7',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'backgroundColor': '#FFFFFF'
                    }),
                ]),
                
                # Color legend
                html.Div([
                    html.H4("颜色图例", style={
                        'color': '#34495E',
                        'marginTop': '30px',
                        'marginBottom': '15px'
                    }),
                    html.Div(id="color-legend")
                ])
                
            ], style={
                'padding': '20px',
                'backgroundColor': '#F8F9FA',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            })
        ], style={
            'width': '28%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'padding': '10px'
        }),

        # Right side for text display
        html.Div([
            html.Div([
                html.H3("文本高亮显示", style={
                    'color': '#34495E',
                    'borderBottom': '2px solid #E74C3C',
                    'paddingBottom': '10px',
                    'marginBottom': '20px'
                }),
                html.Div(id="highlighted-text"),
            ], style={
                'padding': '20px',
                'backgroundColor': '#FFFFFF',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'minHeight': '600px'
            })
        ], style={
            'width': '70%', 
            'display': 'inline-block', 
            'padding': '10px', 
            'verticalAlign': 'top'
        }),
    ], style={
        'display': 'flex',
        'gap': '10px',
        'maxWidth': '1400px',
        'margin': '0 auto'
    })
], style={
    'backgroundColor': '#ECF0F1',
    'minHeight': '100vh',
    'padding': '20px',
    'fontFamily': 'Arial, sans-serif'
})

# Create hierarchical checklist based on selected book
@app.callback(
    Output("label-checklist-container", "children"),
    Input("book-selector", "value"),
)
def create_hierarchical_checklist(selected_book):
    if not selected_book:
        return html.P("请先选择书籍", style={'color': '#7F8C8D', 'fontStyle': 'italic'})
    
    # Filter imagery data for the selected book
    # book_imagery = df_imagery[df_imagery['book'] == selected_book]
    book_imagery = df_imagery.copy()
    
    # Group by big_label and collect small_labels
    label_hierarchy = {}
    for _, row in book_imagery.iterrows():
        big_label = row['big_label']
        small_label = row['small_label']
        
        if big_label not in label_hierarchy:
            label_hierarchy[big_label] = set()
        label_hierarchy[big_label].add(small_label)
    
    # Create checklist items
    checklist_items = []
    
    for big_label in sorted(label_hierarchy.keys()):
        # Big label checkbox
        checklist_items.append(
            html.Div([
                dcc.Checklist(
                    id={'type': 'big-label-check', 'index': big_label},
                    options=[{'label': big_label, 'value': big_label}],
                    value=[],
                    style={'fontWeight': 'bold', 'color': '#2C3E50'},
                    labelStyle={'display': 'block', 'marginBottom': '5px'}
                )
            ])
        )
        
        # Small label checkboxes (indented)
        small_labels = sorted(label_hierarchy[big_label])
        if small_labels:
            small_checklist_items = []
            for small_label in small_labels:
                small_checklist_items.append(
                    html.Div([
                        dcc.Checklist(
                            id={'type': 'small-label-check', 'index': f"{big_label}::{small_label}"},
                            options=[{'label': small_label, 'value': small_label}],
                            value=[],
                            style={'fontSize': '14px', 'color': '#34495E'},
                            labelStyle={'display': 'block', 'marginBottom': '3px'}
                        )
                    ], style={'marginLeft': '25px', 'marginBottom': '2px'})
                )
            
            checklist_items.extend(small_checklist_items)
        
        # Add spacing between big label groups
        checklist_items.append(html.Div(style={'height': '10px'}))
    
    return checklist_items

# Combined callback for both color legend and text highlighting
@app.callback(
    [Output("color-legend", "children"),
     Output("highlighted-text", "children")],
    Input("book-selector", "value"),
    Input({'type': 'big-label-check', 'index': dash.dependencies.ALL}, 'value'),
    Input({'type': 'small-label-check', 'index': dash.dependencies.ALL}, 'value'),
    prevent_initial_call=True
)
def update_legend_and_highlight_text(selected_book, big_label_values, small_label_values):
    if not selected_book:
        legend = html.P("请先选择书籍", style={'color': '#7F8C8D', 'fontStyle': 'italic'})
        text_display = html.Div([
            html.P("请先选择书籍。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '18px',
                      'marginTop': '100px'
                  })
        ])
        return legend, text_display
    
    # Collect selected big labels
    selected_big_labels = []
    for values in big_label_values:
        selected_big_labels.extend(values)
    
    # Collect selected small labels
    selected_small_labels = []
    for values in small_label_values:
        selected_small_labels.extend(values)
    
    # Update color legend
    if not selected_big_labels and not selected_small_labels:
        legend = html.P("请选择意象标签", style={'color': '#7F8C8D', 'fontStyle': 'italic'})
    else:
        legend_items = []
        
        # Show big label colors
        for big_label in selected_big_labels:
            legend_items.append(
                html.Div([
                    html.Span(style={
                        'display': 'inline-block',
                        'width': '20px',
                        'height': '20px',
                        'backgroundColor': big_label_colors[big_label],
                        'border': '1px solid #BDC3C7',
                        'marginRight': '10px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span(f"{big_label} (大类)", style={'verticalAlign': 'middle'})
                ], style={'marginBottom': '8px'})
            )
        
        # Show small label colors
        for small_label in selected_small_labels:
            legend_items.append(
                html.Div([
                    html.Span(style={
                        'display': 'inline-block',
                        'width': '20px',
                        'height': '20px',
                        'backgroundColor': small_label_colors[small_label],
                        'border': '1px solid #BDC3C7',
                        'marginRight': '10px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span(f"{small_label} (小类)", style={'verticalAlign': 'middle', 'fontSize': '14px'})
                ], style={'marginBottom': '6px', 'marginLeft': '15px'})
            )
        
        legend = legend_items
    
    # Update highlighted text
    if not selected_big_labels and not selected_small_labels:
        text_display = html.Div([
            html.P("请选择一个或多个意象标签来开始文本高亮显示。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '18px',
                      'marginTop': '100px'
                  })
        ])
    else:
        row = df_books[df_books['title'] == selected_book].iloc[0]
        text = row['text']

        # Filter imagery data for the selected book first
        # book_imagery = df_imagery[df_imagery['book'] == selected_book]
        book_imagery = df_imagery.copy()
        
        # Filter by selected labels (both big and small)
        filtered = book_imagery[
            (book_imagery['big_label'].isin(selected_big_labels)) |
            (book_imagery['small_label'].isin(selected_small_labels))
        ]

        filtered = filtered.drop_duplicates(subset=['word', 'big_label', 'small_label'])

        # Sort by word length (longest first) to avoid partial replacements
        filtered = filtered.sort_values(by='word', key=lambda x: x.str.len(), ascending=False)

        # Build highlight mapping with priority for small_label colors
        highlight_map = {}
        for _, r in filtered.iterrows():
            word = r['word']
            color = small_label_colors.get(r['small_label'], big_label_colors.get(r['big_label'], "#DDDDDD"))
            label_info = f"{r['small_label']} ({r['big_label']})" if r['small_label'] in selected_small_labels else r['big_label']
            if word not in highlight_map:
                highlight_map[word] = (color, label_info)


        def create_highlighted_elements(text, highlight_map):
            # Sort by length to avoid partial matches
            sorted_words = sorted(highlight_map.keys(), key=len, reverse=True)

            # Build pattern to match any of the highlight words
            pattern = '|'.join(re.escape(word) for word in sorted_words)
            parts = re.split(f'({pattern})', text)

            # Construct the final output using html.Span
            elements = []
            for part in parts:
                if part in highlight_map:
                    color, label_info = highlight_map[part]
                    elements.append(html.Span(
                        part,
                        style={
                            'backgroundColor': color,
                            'padding': '2px 4px',
                            'borderRadius': '3px',
                            'border': '1px solid #BDC3C7',
                            'cursor': 'pointer'
                        },
                        title=label_info
                    ))
                else:
                    elements.append(part)
            return elements


        highlighted_text = create_highlighted_elements(text, highlight_map)
        text_display = html.Div(highlighted_text, style={
            'whiteSpace': 'pre-wrap',
            'fontFamily': "'Microsoft YaHei', SimSun, serif",
            'lineHeight': '1.8',
            'fontSize': '16px',
            'padding': '20px',
            'color': '#2C3E50',
            'backgroundColor': '#FEFEFE',
            'maxWidth': '100%',
            'wordWrap': 'break-word',
            'height': '700px',
            'overflowY': 'scroll',
            'border': '1px solid #BDC3C7',
            'borderRadius': '5px'
        })

    
    return legend, text_display

# Run app
if __name__ == '__main__':
    app.run(debug=True, port=8050)