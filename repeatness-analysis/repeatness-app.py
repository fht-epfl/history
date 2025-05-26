import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import random
from collections import defaultdict, Counter

# Load data
df_books = pd.read_pickle("../proc/df_books.pkl")
df_imagery = pd.read_pickle("../proc/imagery_dictionary.pkl")

# Initialize app
app = dash.Dash(__name__)
server = app.server

# 動態意象分析器類別
class DynamicImageryAnalyzer:
    """動態意象分析器 - 根據用戶選擇的標籤進行分析"""
    
    def __init__(self, df_ima, df_books):
        self.df_ima = df_ima.copy()
        self.df_books = df_books.copy()
    
    def get_book_sentences(self, book_title):
        """獲取指定書籍的句子列表"""
        print(f"🔍 查找書籍句子: {book_title}")
        
        book_data = self.df_books[self.df_books['title'] == book_title]
        
        if book_data.empty:
            print(f"❌ 在df_books中找不到書籍: '{book_title}'")
            return []
        
        if 'text_chunk_smallest' not in book_data.columns:
            print(f"❌ 找不到 text_chunk_smallest 欄位")
            return []
        
        sentences = book_data['text_chunk_smallest'].iloc[0]
        
        if not isinstance(sentences, list):
            print(f"❌ text_chunk_smallest 不是列表格式，類型: {type(sentences)}")
            return []
        
        print(f"✅ 成功載入 {len(sentences)} 個句子")
        return sentences
    
    def get_selected_imagery_words(self, book_title, selected_big_labels, selected_small_labels):
        """根據用戶選擇的標籤獲取相關詞彙"""
        print(f"🔍 搜尋意象數據: {book_title}")
        print(f"選中的大標籤: {selected_big_labels}")
        print(f"選中的小標籤: {selected_small_labels}")
        
        # 獲取該書的意象數據
        book_imagery = self.df_ima[self.df_ima['book'] == book_title]
        
        if book_imagery.empty:
            print(f"❌ 書籍 '{book_title}' 在意象數據中不存在")
            return {}
        
        print(f"✅ 找到意象數據: {len(book_imagery)} 個意象")
        
        # 根據用戶選擇篩選意象
        filtered_imagery = book_imagery[
            (book_imagery['big_label'].isin(selected_big_labels)) |
            (book_imagery['small_label'].isin(selected_small_labels))
        ]
        
        if filtered_imagery.empty:
            print(f"❌ 沒有找到與選中標籤相關的意象")
            return {}
        
        print(f"✅ 篩選後的意象數據: {len(filtered_imagery)} 個意象")
        
        # 按照標籤分組收集詞彙
        label_words = {}
        
        # 處理大標籤
        for big_label in selected_big_labels:
            matching_rows = filtered_imagery[filtered_imagery['big_label'] == big_label]
            if not matching_rows.empty:
                words = matching_rows['word'].unique().tolist()
                label_words[big_label] = words
                print(f"  ✅ {big_label}: {len(words)}個詞彙")
        
        # 處理小標籤
        for small_label in selected_small_labels:
            matching_rows = filtered_imagery[filtered_imagery['small_label'] == small_label]
            if not matching_rows.empty:
                words = matching_rows['word'].unique().tolist()
                label_words[small_label] = words
                print(f"  ✅ {small_label}: {len(words)}個詞彙")
        
        return label_words
    
    def count_words_in_text(self, text, word_list):
        """統計詞彙在文本中的出現次數"""
        if not text or not word_list:
            return 0
        
        total_count = 0
        for word in word_list:
            if word and word.strip():
                count = len(re.findall(re.escape(word), text))
                total_count += count
        
        return total_count
    
    def analyze_imagery_patterns(self, book_title, selected_big_labels, selected_small_labels, window_size=5):
        """分析用戶選擇的意象模式"""
        print(f"\n🚀 開始動態分析: {book_title}")
        print(f"🎯 分析目標: 大標籤{len(selected_big_labels)}個, 小標籤{len(selected_small_labels)}個")
        
        # 檢查是否有選擇的標籤
        if not selected_big_labels and not selected_small_labels:
            print("❌ 沒有選擇任何標籤")
            return None
        
        # 獲取句子數據
        book_data = self.df_books[self.df_books['title'] == book_title]
        
        if book_data.empty:
            print("❌ 在df_books中找不到對應書籍")
            return None
        
        sentences = book_data['text_chunk_smallest'].iloc[0]
        
        if not isinstance(sentences, list):
            print(f"❌ 句子數據格式錯誤: {type(sentences)}")
            return None
        
        print(f"✅ 成功載入 {len(sentences)} 個句子")
        
        # 獲取選中標籤的詞彙
        label_words = self.get_selected_imagery_words(book_title, selected_big_labels, selected_small_labels)
        
        if not label_words:
            print("❌ 沒有找到相關詞彙")
            return None
        
        total_words = sum(len(words) for words in label_words.values())
        print(f"✅ 找到 {total_words} 個相關詞彙，開始窗口分析...")
        
        # 滑動窗口分析
        window_analysis = []
        total_sentences = len(sentences)
        all_labels = list(label_words.keys())
        
        for start_idx in range(0, total_sentences, window_size):
            end_idx = min(start_idx + window_size, total_sentences)
            
            window_sentences = sentences[start_idx:end_idx]
            window_text = ' '.join(window_sentences)
            
            label_counts = {}
            total_mentions = 0
            
            for label, words in label_words.items():
                count = self.count_words_in_text(window_text, words)
                label_counts[label] = count
                total_mentions += count
            
            window_info = {
                'window_start': start_idx,
                'window_end': end_idx - 1,
                'window_center': (start_idx + end_idx - 1) / 2,
                'total_mentions': total_mentions,
                'label_counts': label_counts
            }
            
            # 為每個標籤添加具體計數
            for label in all_labels:
                window_info[f'{label}_count'] = label_counts.get(label, 0)
            
            window_analysis.append(window_info)
        
        # 句子級別統計
        sentence_stats = []
        for idx, sentence in enumerate(sentences):
            sentence_label_counts = {}
            total_count = 0
            
            for label, words in label_words.items():
                count = self.count_words_in_text(sentence, words)
                sentence_label_counts[label] = count
                total_count += count
            
            if total_count > 0:
                sentence_stats.append({
                    'sentence_idx': idx,
                    'sentence': sentence,
                    'total_count': total_count,
                    'label_counts': sentence_label_counts
                })
        
        print(f"✅ 分析完成: 找到 {len(sentence_stats)} 個含相關意象的句子")
        
        return {
            'book_title': book_title,
            'total_sentences': len(sentences),
            'selected_labels': {
                'big_labels': selected_big_labels,
                'small_labels': selected_small_labels
            },
            'label_words': label_words,
            'window_analysis': window_analysis,
            'sentence_stats': sentence_stats
        }

# 初始化動態分析器
analyzer = DynamicImageryAnalyzer(df_imagery, df_books)

# Enhanced color generation with scientific journal standards
def generate_enhanced_colors():
    """Generate colors based on scientific journal standards - high contrast yet aesthetically pleasing"""
    
    # 科學期刊常用的高質量配色方案
    # 基於Nature, Science, Cell等頂級期刊的圖表配色
    scientific_colors = [
        "#1f77b4",  # 深藍 - 經典科學藍
        "#ff7f0e",  # 橙色 - 溫暖對比色
        "#2ca02c",  # 綠色 - 自然綠
        "#d62728",  # 紅色 - 警示紅
        "#9467bd",  # 紫色 - 優雅紫
        "#8c564b",  # 棕色 - 大地色
        "#e377c2",  # 粉色 - 柔和粉
        "#7f7f7f",  # 灰色 - 中性灰
        "#bcbd22",  # 橄欖綠 - 自然色
        "#17becf",  # 青藍 - 清新藍
        "#aec7e8",  # 淺藍 - 柔和藍
        "#ffbb78",  # 淺橙 - 暖橙
        "#98df8a",  # 淺綠 - 春綠
        "#ff9896",  # 淺紅 - 溫和紅
        "#c5b0d5",  # 淺紫 - 薰衣草
        "#c49c94",  # 淺棕 - 米棕
        "#f7b6d3",  # 淺粉 - 櫻花粉
        "#c7c7c7",  # 淺灰 - 銀灰
        "#dbdb8d",  # 淺黃綠 - 檸檬綠
        "#9edae5"   # 淺青 - 天藍
    ]
    
    def create_scientific_light_color(base_color, alpha=0.15):
        """創建科學期刊風格的高亮顏色"""
        hex_color = base_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # 使用更精緻的混合算法，保持色相飽和度平衡
        # 參考 Nature 期刊的高亮色彩處理
        lightness_factor = 0.88  # 更高的亮度
        r = int(r * (1 - lightness_factor) + 255 * lightness_factor)
        g = int(g * (1 - lightness_factor) + 255 * lightness_factor)
        b = int(b * (1 - lightness_factor) + 255 * lightness_factor)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def create_scientific_dark_color(base_color, factor=0.85):
        """創建科學期刊風格的深色版本"""
        hex_color = base_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # 保持足夠的對比度和專業感
        min_brightness = 45  # 適中的最低亮度
        r = max(int(r * factor), min_brightness)
        g = max(int(g * factor), min_brightness)
        b = max(int(b * factor), min_brightness)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    big_labels = sorted(df_imagery['big_label'].unique())
    small_labels = sorted(df_imagery['small_label'].unique())
    
    # 為大標籤創建顏色映射
    big_label_colors_light = {}
    big_label_colors_dark = {}
    
    for i, label in enumerate(big_labels):
        base_color = scientific_colors[i % len(scientific_colors)]
        big_label_colors_light[label] = create_scientific_light_color(base_color)
        big_label_colors_dark[label] = create_scientific_dark_color(base_color, 0.9)
    
    # 為小標籤創建顏色映射 - 使用不同起始點
    small_label_colors_light = {}
    small_label_colors_dark = {}
    
    # 從第5個顏色開始，確保與大標籤有足夠區別
    offset = max(5, len(big_labels) // 2)
    
    for i, label in enumerate(small_labels):
        color_index = (i + offset) % len(scientific_colors)
        base_color = scientific_colors[color_index]
        small_label_colors_light[label] = create_scientific_light_color(base_color)
        small_label_colors_dark[label] = base_color  # 小標籤使用原色
    
    return {
        'big_light': big_label_colors_light,
        'big_dark': big_label_colors_dark, 
        'small_light': small_label_colors_light,
        'small_dark': small_label_colors_dark,
        'base_colors': scientific_colors
    }

def get_dynamic_colors_for_selection(selected_big_labels, selected_small_labels):
    """根據當前選擇動態分配最優科學期刊風格顏色組合"""
    all_selected = selected_big_labels + selected_small_labels
    
    if len(all_selected) <= 1:
        return color_schemes
    
    # 使用科學期刊推薦的顏色間隔策略
    base_colors = color_schemes['base_colors']
    num_colors_needed = len(all_selected)
    
    if num_colors_needed <= len(base_colors):
        # 優化的顏色選擇算法 - 確保最大視覺區別
        if num_colors_needed <= 4:
            # 對於少量顏色，使用經典的科學期刊四色組合
            selected_indices = [0, 1, 2, 3]  # 藍、橙、綠、紅
        elif num_colors_needed <= 8:
            # 中等數量，使用擴展的經典組合
            selected_indices = [0, 1, 2, 3, 4, 6, 8, 9]  # 跳過相似顏色
        else:
            # 大量顏色，使用均勻分布
            interval = len(base_colors) // num_colors_needed
            selected_indices = [(i * interval) % len(base_colors) for i in range(num_colors_needed)]
        
        selected_base_colors = [base_colors[i] for i in selected_indices[:num_colors_needed]]
        
        # 重新生成顏色映射
        dynamic_colors = {
            'big_light': {},
            'big_dark': {},
            'small_light': {},
            'small_dark': {}
        }
        
        def create_light_color(base_color):
            hex_color = base_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            lightness_factor = 0.88
            r = int(r * (1 - lightness_factor) + 255 * lightness_factor)
            g = int(g * (1 - lightness_factor) + 255 * lightness_factor)
            b = int(b * (1 - lightness_factor) + 255 * lightness_factor)
            
            return f"#{r:02x}{g:02x}{b:02x}"
        
        def create_dark_color(base_color):
            hex_color = base_color.lstrip('#')
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            min_brightness = 45
            factor = 0.9
            r = max(int(r * factor), min_brightness)
            g = max(int(g * factor), min_brightness)
            b = max(int(b * factor), min_brightness)
            
            return f"#{r:02x}{g:02x}{b:02x}"
        
        # 為選中的標籤分配最優顏色
        color_index = 0
        for label in selected_big_labels:
            if color_index < len(selected_base_colors):
                base_color = selected_base_colors[color_index]
                dynamic_colors['big_light'][label] = create_light_color(base_color)
                dynamic_colors['big_dark'][label] = create_dark_color(base_color)
                color_index += 1
        
        for label in selected_small_labels:
            if color_index < len(selected_base_colors):
                base_color = selected_base_colors[color_index]
                dynamic_colors['small_light'][label] = create_light_color(base_color)
                dynamic_colors['small_dark'][label] = base_color
                color_index += 1
        
        return dynamic_colors
    
    return color_schemes

color_schemes = generate_enhanced_colors()
big_label_colors = color_schemes['big_light']      # 用於文本高亮
small_label_colors = color_schemes['small_light']  # 用於文本高亮

# Dropdown options - filter for 朱天心 books only
book_options = [{'label': row['title'], 'value': row['title']} 
               for _, row in df_books.iterrows() 
               if '朱天心' in row['author']]

# Enhanced Layout with analysis section
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
                
                # 意象分析控制
                html.Div([
                    html.H4("意象重複性分析", style={
                        'color': '#34495E',
                        'marginTop': '30px',
                        'marginBottom': '15px'
                    }),
                    
                    # Debug 按鈕
                    html.Button("🔍 Debug 數據結構", id="debug-button", 
                              style={
                                  'backgroundColor': '#F39C12',
                                  'color': 'white',
                                  'border': 'none',
                                  'padding': '8px 16px',
                                  'borderRadius': '5px',
                                  'marginBottom': '10px',
                                  'cursor': 'pointer',
                                  'width': '100%',
                                  'fontSize': '14px'
                              }),
                    
                    html.Label("窗口大小（句子數）", style={
                        'fontWeight': 'bold', 
                        'color': '#2C3E50',
                        'display': 'block',
                        'marginBottom': '8px'
                    }),
                    dcc.Slider(
                        id='window-size-slider',
                        min=3,
                        max=10,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Button("執行意象分析", id="analyze-button", 
                              style={
                                  'backgroundColor': '#E74C3C',
                                  'color': 'white',
                                  'border': 'none',
                                  'padding': '10px 20px',
                                  'borderRadius': '5px',
                                  'marginTop': '15px',
                                  'cursor': 'pointer',
                                  'width': '100%'
                              }),
                    
                    # Debug 結果顯示區域
                    html.Div(id="debug-info", style={
                        'marginTop': '15px',
                        'padding': '10px',
                        'backgroundColor': '#FEF9E7',
                        'borderRadius': '5px',
                        'fontSize': '12px',
                        'maxHeight': '300px',
                        'overflowY': 'auto',
                        'display': 'none'
                    })
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
                'minHeight': '600px',
                'marginBottom': '20px'
            }),
            
            # 意象分析結果區域
            html.Div([
                html.H3("意象重複性分析", style={
                    'color': '#34495E',
                    'borderBottom': '2px solid #3498DB',
                    'paddingBottom': '10px',
                    'marginBottom': '20px'
                }),
                html.Div(id="ethnic-analysis-results"),
            ], style={
                'padding': '20px',
                'backgroundColor': '#FFFFFF',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'minHeight': '400px',
                'marginBottom': '20px'
            }),
            
            # 主語省略分析區域
            html.Div([
                html.H3("主語省略分析", style={
                    'color': '#34495E',
                    'borderBottom': '2px solid #9B59B6',
                    'paddingBottom': '10px',
                    'marginBottom': '20px'
                }),
                html.Div(id="omission-analysis-results"),
            ], style={
                'padding': '20px',
                'backgroundColor': '#FFFFFF',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'minHeight': '400px',
                'marginBottom': '20px'
            }),
            
            # 時態分析區域
            html.Div([
                html.H3("敘述時態分析", style={
                    'color': '#34495E',
                    'borderBottom': '2px solid #E67E22',
                    'paddingBottom': '10px',
                    'marginBottom': '20px'
                }),
                html.Div(id="anachrony-analysis-results"),
            ], style={
                'padding': '20px',
                'backgroundColor': '#FFFFFF',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'minHeight': '400px'
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
    book_imagery = df_imagery[df_imagery['book'] == selected_book]
    
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
        # 使用動態顏色分配
        dynamic_colors = get_dynamic_colors_for_selection(selected_big_labels, selected_small_labels)
        
        legend_items = []
        
        # Show big label colors (使用動態淺色用於圖例顯示)
        for big_label in selected_big_labels:
            light_color = dynamic_colors['big_light'].get(big_label, color_schemes['big_light'].get(big_label, '#DDDDDD'))
            dark_color = dynamic_colors['big_dark'].get(big_label, color_schemes['big_dark'].get(big_label, '#999999'))
            
            legend_items.append(
                html.Div([
                    html.Span(style={
                        'display': 'inline-block',
                        'width': '20px',
                        'height': '20px',
                        'backgroundColor': light_color,
                        'border': f'2px solid {dark_color}',
                        'marginRight': '10px',
                        'verticalAlign': 'middle'
                    }),
                    html.Span(f"{big_label} (大类)", style={'verticalAlign': 'middle'})
                ], style={'marginBottom': '8px'})
            )
        
        # Show small label colors (使用動態淺色用於圖例顯示)
        for small_label in selected_small_labels:
            light_color = dynamic_colors['small_light'].get(small_label, color_schemes['small_light'].get(small_label, '#DDDDDD'))
            dark_color = dynamic_colors['small_dark'].get(small_label, color_schemes['small_dark'].get(small_label, '#999999'))
            
            legend_items.append(
                html.Div([
                    html.Span(style={
                        'display': 'inline-block',
                        'width': '20px',
                        'height': '20px',
                        'backgroundColor': light_color,
                        'border': f'2px solid {dark_color}',
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
        book_imagery = df_imagery[df_imagery['book'] == selected_book]
        
        # Filter by selected labels (both big and small)
        filtered = book_imagery[
            (book_imagery['big_label'].isin(selected_big_labels)) |
            (book_imagery['small_label'].isin(selected_small_labels))
        ]

        # Sort by word length (longest first) to avoid partial replacements
        filtered = filtered.sort_values(by='word', key=lambda x: x.str.len(), ascending=False)

        # Build highlight mapping with priority for small_label colors
        highlight_map = {}
        for _, r in filtered.iterrows():
            word = r['word']
            # Use small_label color if the small_label is selected, otherwise use big_label color
            if r['small_label'] in selected_small_labels:
                color = color_schemes['small_light'][r['small_label']]
                label_info = f"{r['small_label']} ({r['big_label']})"
            else:
                color = color_schemes['big_light'][r['big_label']]
                label_info = r['big_label']
            
            if word not in highlight_map:
                highlight_map[word] = f"<span style='background-color:{color}; padding:2px 4px; border-radius:3px; border:1px solid #BDC3C7;' title='{label_info}'>{word}</span>"

        def highlight_words(text, highlight_map):
            for word, span in highlight_map.items():
                text = text.replace(word, span)
            return text

        highlighted_text = highlight_words(text, highlight_map)

        # Enhanced HTML styling
        html_output = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    white-space: pre-wrap;
                    font-family: 'Microsoft YaHei', SimSun, serif;
                    line-height: 1.8;
                    font-size: 16px;
                    padding: 20px;
                    color: #2C3E50;
                    background-color: #FEFEFE;
                    max-width: 100%;
                    word-wrap: break-word;
                }}
                span {{
                    transition: all 0.2s ease;
                }}
                span:hover {{
                    transform: scale(1.05);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }}
            </style>
        </head>
        <body>{highlighted_text}</body>
        </html>
        """

        text_display = html.Iframe(
            srcDoc=html_output,
            style={
                'width': '100%', 
                'height': '700px', 
                'border': '1px solid #BDC3C7',
                'borderRadius': '5px'
            }
        )
    
    return legend, text_display

# Debug 回調函數
@app.callback(
    Output("debug-info", "children"),
    Output("debug-info", "style"),
    [Input("debug-button", "n_clicks")],
    [State("book-selector", "value")],
    prevent_initial_call=True
)
def debug_data_structure(n_clicks, selected_book):
    if not n_clicks or not selected_book:
        return "", {'display': 'none'}
    
    debug_info = []
    
    debug_info.append(html.H5("🔍 Debug 資訊", style={'color': '#E67E22', 'marginBottom': '10px'}))
    
    # 1. 檢查選中的書籍
    debug_info.append(html.P(f"📖 選中書籍: {selected_book}", style={'margin': '5px 0'}))
    
    # 2. 檢查 df_books 結構
    debug_info.append(html.P("📊 df_books 資訊:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    debug_info.append(html.P(f"  • 總行數: {len(df_books)}", style={'margin': '2px 0', 'marginLeft': '15px'}))
    debug_info.append(html.P(f"  • 欄位: {list(df_books.columns)}", style={'margin': '2px 0', 'marginLeft': '15px'}))
    
    # 3. 檢查 df_imagery 結構
    debug_info.append(html.P("🎨 df_imagery 資訊:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    debug_info.append(html.P(f"  • 總行數: {len(df_imagery)}", style={'margin': '2px 0', 'marginLeft': '15px'}))
    debug_info.append(html.P(f"  • 欄位: {list(df_imagery.columns)}", style={'margin': '2px 0', 'marginLeft': '15px'}))
    
    # 4. 檢查書籍匹配
    debug_info.append(html.P("🔍 書籍匹配測試:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    
    # 方法1: 用 title 匹配
    if 'title' in df_books.columns:
        title_match = df_books[df_books['title'] == selected_book]
        debug_info.append(html.P(f"  • title 匹配: {len(title_match)} 行", style={'margin': '2px 0', 'marginLeft': '15px'}))
        if not title_match.empty:
            debug_info.append(html.P(f"    完整 book 值: {title_match['book'].iloc[0]}", style={'margin': '2px 0', 'marginLeft': '25px'}))
    
    # 方法2: 用 book 包含匹配
    book_contain_match = df_books[df_books['book'].str.contains(selected_book, na=False, regex=False)]
    debug_info.append(html.P(f"  • book 包含匹配: {len(book_contain_match)} 行", style={'margin': '2px 0', 'marginLeft': '15px'}))
    
    # 5. 顯示相關書籍範例
    debug_info.append(html.P("📚 相關書籍範例:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    related_books = df_books[df_books['book'].str.contains('朱天心', na=False, regex=False)]
    for i, (_, row) in enumerate(related_books.head(5).iterrows()):
        book_info = f"book: {row['book']}"
        if 'title' in row:
            book_info += f", title: {row['title']}"
        debug_info.append(html.P(f"  {i+1}. {book_info}", style={'margin': '2px 0', 'marginLeft': '15px', 'fontSize': '11px'}))
    
    # 6. 檢查 text_chunk_smallest
    debug_info.append(html.P("📝 句子資料檢查:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    
    if 'title' in df_books.columns:
        book_data = df_books[df_books['title'] == selected_book]
    else:
        book_data = df_books[df_books['book'].str.contains(selected_book, na=False, regex=False)]
    
    if not book_data.empty:
        if 'text_chunk_smallest' in book_data.columns:
            sentences = book_data['text_chunk_smallest'].iloc[0]
            debug_info.append(html.P(f"  • text_chunk_smallest 類型: {type(sentences)}", style={'margin': '2px 0', 'marginLeft': '15px'}))
            
            if isinstance(sentences, list):
                debug_info.append(html.P(f"  • 句子數量: {len(sentences)}", style={'margin': '2px 0', 'marginLeft': '15px'}))
                if sentences:
                    debug_info.append(html.P(f"  • 第一句範例: {sentences[0][:50]}...", style={'margin': '2px 0', 'marginLeft': '15px', 'fontSize': '11px'}))
            else:
                debug_info.append(html.P(f"  • 資料內容 (前100字): {str(sentences)[:100]}...", style={'margin': '2px 0', 'marginLeft': '15px', 'fontSize': '11px'}))
        else:
            debug_info.append(html.P("  ❌ 找不到 text_chunk_smallest 欄位", style={'margin': '2px 0', 'marginLeft': '15px', 'color': 'red'}))
    else:
        debug_info.append(html.P("  ❌ 找不到對應的書籍資料", style={'margin': '2px 0', 'marginLeft': '15px', 'color': 'red'}))
    
    # 7. 檢查意象資料匹配
    debug_info.append(html.P("🎭 意象資料匹配:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    
    if not book_data.empty and 'book' in book_data.columns:
        full_book_name = book_data['book'].iloc[0]
        imagery_match1 = df_imagery[df_imagery['book'] == full_book_name]
        debug_info.append(html.P(f"  • 用完整名稱匹配: {len(imagery_match1)} 行", style={'margin': '2px 0', 'marginLeft': '15px'}))
        
        imagery_match2 = df_imagery[df_imagery['book'] == selected_book]
        debug_info.append(html.P(f"  • 用書名匹配: {len(imagery_match2)} 行", style={'margin': '2px 0', 'marginLeft': '15px'}))
        
        imagery_match3 = df_imagery[df_imagery['book'].str.contains(selected_book, na=False, regex=False)]
        debug_info.append(html.P(f"  • 用包含匹配: {len(imagery_match3)} 行", style={'margin': '2px 0', 'marginLeft': '15px'}))
    
    # 8. 顯示意象資料中的書籍範例
    debug_info.append(html.P("🎨 意象資料書籍範例:", style={'fontWeight': 'bold', 'margin': '10px 0 5px 0'}))
    unique_imagery_books = df_imagery['book'].unique()[:10]
    for i, book in enumerate(unique_imagery_books):
        debug_info.append(html.P(f"  {i+1}. {book}", style={'margin': '2px 0', 'marginLeft': '15px', 'fontSize': '11px'}))
    
    return debug_info, {
        'marginTop': '15px',
        'padding': '10px',
        'backgroundColor': '#FEF9E7',
        'borderRadius': '5px',
        'fontSize': '12px',
        'maxHeight': '400px',
        'overflowY': 'auto',
        'border': '1px solid #F39C12',
        'display': 'block'
    }

# 簡化的時態分析回調函數 - 只保留餅圖、統計摘要和詞彙範例
@app.callback(
    Output("anachrony-analysis-results", "children"),
    Input("book-selector", "value"),
    prevent_initial_call=True
)
def update_anachrony_analysis(selected_book):
    if not selected_book:
        return html.Div([
            html.P("請選擇書籍以查看敘述時態分析。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 獲取書籍資料
    book_data = df_books[df_books['title'] == selected_book]
    
    if book_data.empty:
        return html.Div([
            html.P("找不到對應的書籍資料。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    book_row = book_data.iloc[0]
    
    # 檢查是否有時態分析資料
    if 'anachrony_type' not in book_row or 'anachrony_terms' not in book_row:
        return html.Div([
            html.P("該書籍缺少敘述時態分析資料。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    anachrony_types = book_row['anachrony_type']
    anachrony_terms = book_row['anachrony_terms']
    
    # 檢查資料格式
    if not isinstance(anachrony_types, list) or not isinstance(anachrony_terms, list):
        return html.Div([
            html.P("敘述時態資料格式錯誤。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    if len(anachrony_types) == 0:
        return html.Div([
            html.P("該書籍沒有檢測到時態變化。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 處理 list of lists 資料
    def flatten_anachrony_data(data_list):
        """展平時態資料並統計"""
        flattened = []
        sentence_counts = {'追述': 0, '预述': 0, '混合': 0}
        
        for i, sentence_data in enumerate(data_list):
            if isinstance(sentence_data, list):
                if len(sentence_data) > 1:
                    # 混合時態
                    sentence_counts['混合'] += 1
                    flattened.extend(sentence_data)
                elif len(sentence_data) == 1:
                    # 單一時態
                    tense = sentence_data[0]
                    sentence_counts[tense] = sentence_counts.get(tense, 0) + 1
                    flattened.append(tense)
            elif isinstance(sentence_data, str):
                # 直接是字符串
                sentence_counts[sentence_data] = sentence_counts.get(sentence_data, 0) + 1
                flattened.append(sentence_data)
                
        return flattened, sentence_counts
    
    # 處理時態類型和詞彙
    flat_types, sentence_type_counts = flatten_anachrony_data(anachrony_types)
    flat_terms, _ = flatten_anachrony_data(anachrony_terms)
    
    # 統計總體分佈
    from collections import Counter
    type_counts = Counter(flat_types)
    
    # 準備圖表資料
    tense_labels = list(type_counts.keys())
    tense_counts = list(type_counts.values())
    
    # 創建時態分佈餅圖
    colors_map = {
        '追述': '#E67E22',  # 橘色 - 回望過去
        '预述': '#3498DB',  # 藍色 - 展望未來  
        '混合': '#9B59B6'   # 紫色 - 混合時態
    }
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=tense_labels,
        values=tense_counts,
        hole=0.3,
        marker=dict(colors=[colors_map.get(label, '#95A5A6') for label in tense_labels]),
        textinfo='label+percent+value',
        textposition='outside',
        textfont=dict(size=14)
    )])
    
    fig_pie.update_layout(
        title=f"《{selected_book}》敘述時態分佈",
        font=dict(size=14),
        height=450,
        margin=dict(t=80, b=60, l=60, r=60),
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # 統計摘要
    total_tense_instances = sum(type_counts.values())
    total_sentences_with_tense = len([t for t in sentence_type_counts.values() if t > 0])
    total_sentences = len(anachrony_types)
    tense_coverage = (sum(sentence_type_counts.values()) / total_sentences * 100) if total_sentences > 0 else 0
    
    most_common_tense = type_counts.most_common(1)[0] if type_counts else ("無", 0)
    
    # 創建統計表格
    stats_table = []
    stats_table.append(html.Tr([
        html.Th("統計項目", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}),
        html.Th("數值", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("總時態標記數", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{total_tense_instances}個", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("含時態變化的句子", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{sum(sentence_type_counts.values())}句", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("時態覆蓋率", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{tense_coverage:.1f}%", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("最常見時態", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"「{most_common_tense[0]}」({most_common_tense[1]}次)", 
               style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    # 混合時態句子數
    mixed_sentences = sentence_type_counts.get('混合', 0)
    stats_table.append(html.Tr([
        html.Td("混合時態句子", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{mixed_sentences}句", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    # 時態詞彙範例（顯示前10個）
    term_examples = []
    shown_terms = []
    for i, terms_list in enumerate(anachrony_terms[:10]):
        if isinstance(terms_list, list) and terms_list:
            for term in terms_list:
                if term and term not in shown_terms:
                    shown_terms.append(term)
                    if len(shown_terms) <= 10:
                        # 判斷對應的時態類型
                        tense_type = '未知'
                        if i < len(anachrony_types):
                            sentence_tense = anachrony_types[i]
                            if isinstance(sentence_tense, list) and sentence_tense:
                                tense_type = '/'.join(sentence_tense)
                            elif isinstance(sentence_tense, str):
                                tense_type = sentence_tense
                        
                        term_color = colors_map.get(tense_type.split('/')[0], '#95A5A6')
                        term_examples.append(
                            html.Div([
                                html.Span(f"「{term}」", style={
                                    'fontWeight': 'bold', 
                                    'color': term_color,
                                    'marginRight': '10px',
                                    'fontSize': '16px'
                                }),
                                html.Span(f"({tense_type})", style={
                                    'color': '#6c757d',
                                    'fontSize': '14px'
                                })
                            ], style={
                                'marginBottom': '10px', 
                                'padding': '8px 12px', 
                                'backgroundColor': '#f8f9fa', 
                                'borderRadius': '6px',
                                'border': f'1px solid {term_color}',
                                'display': 'inline-block',
                                'marginRight': '10px'
                            })
                        )
    
    return html.Div([
        # 餅圖和統計摘要並排 - 左餅右表
        html.Div([
            # 左側：餅圖
            html.Div([
                dcc.Graph(figure=fig_pie)
            ], style={
                'width': '55%', 
                'display': 'inline-block', 
                'verticalAlign': 'top'
            }),
            
            # 右側：統計摘要和詞彙範例
            html.Div([
                html.H4("📊 統計摘要", style={
                    'color': '#34495E', 
                    'marginBottom': '15px'
                }),
                html.Table(stats_table, style={
                    'width': '100%', 
                    'borderCollapse': 'collapse',
                    'marginBottom': '25px'
                }),
                
                # 時態詞彙範例
                html.H4("🎯 時態詞彙範例", style={
                    'color': '#34495E', 
                    'marginBottom': '15px'
                }),
                html.Div(term_examples, style={
                    'lineHeight': '1.6'
                }),
                html.P(f"以上顯示前 10 個時態標記詞彙範例", 
                      style={
                          'color': '#6c757d', 
                          'fontStyle': 'italic', 
                          'marginTop': '15px',
                          'fontSize': '12px'
                      })
            ], style={
                'width': '45%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'paddingLeft': '20px'
            })
        ])
    ])

# 簡化的主語省略分析回調函數 - 只保留餅圖和統計摘要
@app.callback(
    Output("omission-analysis-results", "children"),
    Input("book-selector", "value"),
    prevent_initial_call=True
)
def update_omission_analysis(selected_book):
    if not selected_book:
        return html.Div([
            html.P("請選擇書籍以查看主語省略分析。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 獲取書籍資料
    book_data = df_books[df_books['title'] == selected_book]
    
    if book_data.empty:
        return html.Div([
            html.P("找不到對應的書籍資料。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    book_row = book_data.iloc[0]
    
    # 檢查是否有主語省略資料
    if 'omitted_subjects' not in book_row or 'omission_sentences' not in book_row:
        return html.Div([
            html.P("該書籍缺少主語省略分析資料。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    omitted_subjects = book_row['omitted_subjects']
    omission_sentences = book_row['omission_sentences']
    
    # 檢查資料格式
    if not isinstance(omitted_subjects, list) or not isinstance(omission_sentences, list):
        return html.Div([
            html.P("主語省略資料格式錯誤。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    if len(omitted_subjects) == 0:
        return html.Div([
            html.P("該書籍沒有主語省略現象。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 統計主語分佈
    from collections import Counter
    subject_counts = Counter(omitted_subjects)
    
    # 只顯示前10名主語，其餘歸為"其他"
    top_subjects = subject_counts.most_common(10)
    other_count = sum(count for subject, count in subject_counts.items() 
                     if subject not in [s[0] for s in top_subjects])
    
    # 準備餅圖資料
    pie_labels = [subject for subject, count in top_subjects]
    pie_values = [count for subject, count in top_subjects]
    
    if other_count > 0:
        pie_labels.append("其他")
        pie_values.append(other_count)
    
    # 創建餅圖 - 只顯示比例>2%的標籤
    total_pie_count = sum(pie_values)
    pie_text = []
    
    for i, (label, value) in enumerate(zip(pie_labels, pie_values)):
        percentage = (value / total_pie_count) * 100
        if percentage >= 2.0:  # 只顯示占比>=2%的標籤
            pie_text.append(f"{label}<br>{percentage:.1f}%")
        else:
            pie_text.append('')
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.3,
        text=pie_text,
        textinfo='text',
        textposition='outside',
        textfont=dict(size=14),
        marker=dict(
            colors=['#9B59B6', '#3498DB', '#E74C3C', '#F39C12', '#27AE60', 
                   '#E67E22', '#1ABC9C', '#34495E', '#F1C40F', '#95A5A6', '#BDC3C7'][:len(pie_labels)]
        ),
        showlegend=True
    )])
    
    fig_pie.update_layout(
        title=f"《{selected_book}》省略主語分佈（前10名）",
        font=dict(size=14),
        height=450,
        margin=dict(t=80, b=60, l=60, r=120),
        legend=dict(
            orientation="v", 
            yanchor="middle", 
            y=0.5, 
            xanchor="left", 
            x=1.01,
            font=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # 統計摘要
    total_omissions = len(omitted_subjects)
    unique_subjects = len(subject_counts)
    most_common_subject = subject_counts.most_common(1)[0] if subject_counts else ("無", 0)
    
    # 顯示前10名的統計資訊
    top_10_count = sum(count for subject, count in top_subjects)
    coverage_percentage = (top_10_count / total_omissions * 100) if total_omissions > 0 else 0
    
    # 計算省略率（如果有總句子數的話）
    total_sentences = book_row.get('total_sentences', len(book_row.get('text_chunk_smallest', [])))
    omission_rate = (len(omission_sentences) / total_sentences * 100) if total_sentences > 0 else 0
    
    # 創建統計表格
    stats_table = []
    stats_table.append(html.Tr([
        html.Th("統計項目", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}),
        html.Th("數值", style={'border': '1px solid #ddd', 'padding': '12px', 'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("總主語省略次數", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{total_omissions}次", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("含省略的句子數", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{len(omission_sentences)}句", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("省略句子比例", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{omission_rate:.1f}%", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("不同主語類型數", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{unique_subjects}種", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("最常省略的主語", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"「{most_common_subject[0]}」({most_common_subject[1]}次)", 
               style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("前10名覆蓋率", style={'border': '1px solid #ddd', 'padding': '10px'}),
        html.Td(f"{coverage_percentage:.1f}%", style={'border': '1px solid #ddd', 'padding': '10px', 'fontWeight': 'bold'})
    ]))
    
    return html.Div([
        # 餅圖和統計摘要並排 - 左餅右表
        html.Div([
            # 左側：餅圖
            html.Div([
                dcc.Graph(figure=fig_pie)
            ], style={
                'width': '55%', 
                'display': 'inline-block', 
                'verticalAlign': 'top'
            }),
            
            # 右側：統計摘要
            html.Div([
                html.H4("📊 統計摘要", style={
                    'color': '#34495E', 
                    'marginBottom': '15px'
                }),
                html.Table(stats_table, style={
                    'width': '100%', 
                    'borderCollapse': 'collapse'
                })
            ], style={
                'width': '45%', 
                'display': 'inline-block', 
                'verticalAlign': 'top',
                'paddingLeft': '20px'
            })
        ])
    ])

# 動態分析回調函數
@app.callback(
    Output("ethnic-analysis-results", "children"),
    [Input("analyze-button", "n_clicks")],
    [State("book-selector", "value"),
     State("window-size-slider", "value"),
     State({'type': 'big-label-check', 'index': dash.dependencies.ALL}, 'value'),
     State({'type': 'small-label-check', 'index': dash.dependencies.ALL}, 'value')],
    prevent_initial_call=True
)
def update_dynamic_analysis(n_clicks, selected_book, window_size, big_label_values, small_label_values):
    if not n_clicks or not selected_book:
        return html.Div([
            html.P("點擊「執行意象分析」按鈕開始分析。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#7F8C8D',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 收集選中的標籤
    selected_big_labels = []
    for values in big_label_values:
        selected_big_labels.extend(values)
    
    selected_small_labels = []
    for values in small_label_values:
        selected_small_labels.extend(values)
    
    # 檢查是否有選擇標籤
    if not selected_big_labels and not selected_small_labels:
        return html.Div([
            html.P("請先選擇要分析的意象標籤。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 執行動態分析
    analysis_results = analyzer.analyze_imagery_patterns(
        selected_book, selected_big_labels, selected_small_labels, window_size
    )
    
    if not analysis_results:
        return html.Div([
            html.P("該書籍沒有與選中標籤相關的意象數據。", 
                  style={
                      'textAlign': 'center', 
                      'color': '#E74C3C',
                      'fontSize': '16px',
                      'marginTop': '50px'
                  })
        ])
    
    # 獲取數據
    window_analysis = analysis_results['window_analysis']
    sentence_stats = analysis_results['sentence_stats']
    label_words = analysis_results['label_words']
    all_labels = list(label_words.keys())
    
    # 為每個標籤分配顏色（圖表使用深色變體）
    # 使用動態顏色分配確保高對比
    dynamic_colors = get_dynamic_colors_for_selection(selected_big_labels, selected_small_labels)
    
    label_color_mapping = {}
    for label in all_labels:
        if label in selected_big_labels:
            label_color_mapping[label] = dynamic_colors['big_dark'].get(label, 
                                       color_schemes['big_dark'].get(label, '#666666'))
        elif label in selected_small_labels:
            label_color_mapping[label] = dynamic_colors['small_dark'].get(label, 
                                       color_schemes['small_dark'].get(label, '#666666'))
        else:
            # 備用顏色（深色）
            colors = ['#D32F2F', '#388E3C', '#1976D2', '#689F38', '#F57C00', '#7B1FA2']
            label_color_mapping[label] = colors[len(label_color_mapping) % len(colors)]
    
    # 1. 滑動窗口圖表
    fig_window = go.Figure()
    
    for label in all_labels:
        if label_words[label]:  # 只有該書有相關詞彙才顯示
            window_centers = [w['window_center'] for w in window_analysis]
            counts = [w.get(f'{label}_count', 0) for w in window_analysis]
            
            fig_window.add_trace(go.Scatter(
                x=window_centers,
                y=counts,
                mode='lines+markers',
                name=label,
                line=dict(color=label_color_mapping[label], width=3),
                marker=dict(size=6)
            ))
    
    # 動態標題
    analysis_scope = f"大標籤: {', '.join(selected_big_labels) if selected_big_labels else '無'}"
    if selected_small_labels:
        analysis_scope += f" | 小標籤: {', '.join(selected_small_labels)}"
    
    fig_window.update_layout(
        title=f"《{selected_book}》意象重複性分析（窗口大小：{window_size}句）<br><sub>{analysis_scope}</sub>",
        xaxis_title="句子位置",
        yaxis_title="意象出現次數",
        hovermode='x unified',
        height=400,
        plot_bgcolor='white',  # 移除灰色背景
        paper_bgcolor='white'
    )
    
    # 2. 句子散點圖
    fig_scatter = go.Figure()
    
    for i, label in enumerate(all_labels):
        if label_words[label]:
            label_sentences = [s for s in sentence_stats if s['label_counts'].get(label, 0) > 0]
            if label_sentences:
                indices = [s['sentence_idx'] for s in label_sentences]
                
                fig_scatter.add_trace(go.Scatter(
                    x=indices,
                    y=[i] * len(indices),
                    mode='markers',
                    name=f"{label} ({len(indices)}句)",
                    marker=dict(color=label_color_mapping[label], size=8, opacity=0.7)
                ))
    
    fig_scatter.update_layout(
        title=f"《{selected_book}》意象在文本中的分布",
        xaxis_title="句子位置",
        yaxis_title="意象標籤",
        yaxis=dict(tickmode='array', tickvals=list(range(len(all_labels))), ticktext=all_labels),
        height=max(300, len(all_labels) * 60),  # 根據標籤數量調整高度
        plot_bgcolor='white',  # 移除灰色背景
        paper_bgcolor='white'
    )
    
    # 3. 統計摘要
    total_mentions = sum(s['total_count'] for s in sentence_stats)
    label_distribution = {}
    
    for label in all_labels:
        count = sum(s['label_counts'].get(label, 0) for s in sentence_stats)
        label_distribution[label] = count
    
    # 創建統計表格
    stats_table = []
    stats_table.append(html.Tr([
        html.Th("統計項目", style={'border': '1px solid #ddd', 'padding': '8px', 'backgroundColor': '#f2f2f2'}),
        html.Th("數值", style={'border': '1px solid #ddd', 'padding': '8px', 'backgroundColor': '#f2f2f2'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("總句子數", style={'border': '1px solid #ddd', 'padding': '8px'}),
        html.Td(f"{analysis_results['total_sentences']:,}句", style={'border': '1px solid #ddd', 'padding': '8px'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("含相關意象的句子", style={'border': '1px solid #ddd', 'padding': '8px'}),
        html.Td(f"{len(sentence_stats)}句 ({len(sentence_stats)/analysis_results['total_sentences']:.1%})", 
               style={'border': '1px solid #ddd', 'padding': '8px'})
    ]))
    
    stats_table.append(html.Tr([
        html.Td("意象總出現次數", style={'border': '1px solid #ddd', 'padding': '8px'}),
        html.Td(f"{total_mentions}次", style={'border': '1px solid #ddd', 'padding': '8px'})
    ]))
    
    # 各標籤統計
    for label in all_labels:
        if label_distribution[label] > 0:
            percentage = (label_distribution[label] / total_mentions * 100) if total_mentions > 0 else 0
            word_count = len(label_words[label])
            stats_table.append(html.Tr([
                html.Td(f"{label}意象", style={'border': '1px solid #ddd', 'padding': '8px'}),
                html.Td(f"{label_distribution[label]}次 ({percentage:.1f}%) | {word_count}個詞彙", 
                       style={'border': '1px solid #ddd', 'padding': '8px'})
            ]))
    
    # 詞彙展示
    word_display = []
    for label in all_labels:
        if label_words[label]:
            word_display.append(
                html.Div([
                    html.H5(f"{label}相關詞彙：", 
                           style={'color': dynamic_colors['big_dark'].get(label, color_schemes['big_dark'].get(label, '#666666')) if label in selected_big_labels else 
                                           dynamic_colors['small_dark'].get(label, color_schemes['small_dark'].get(label, '#666666')), 
                                  'marginTop': '15px'}),
                    html.P(f"{', '.join(label_words[label][:10])}" + 
                          (f"... (共{len(label_words[label])}個)" if len(label_words[label]) > 10 else ""),
                          style={'fontSize': '14px', 'color': '#34495E', 'marginLeft': '10px'})
                ])
            )
    
    return html.Div([
        # 分析範圍說明
        html.Div([
            html.H4("🎯 分析範圍", style={'color': '#34495E', 'marginBottom': '15px'}),
            html.P(f"書籍：《{selected_book}》", style={'margin': '5px 0', 'fontWeight': 'bold'}),
            html.P(f"大標籤：{', '.join(selected_big_labels) if selected_big_labels else '未選擇'}", 
                  style={'margin': '5px 0'}),
            html.P(f"小標籤：{', '.join(selected_small_labels) if selected_small_labels else '未選擇'}", 
                  style={'margin': '5px 0'}),
            html.P(f"窗口大小：{window_size}句", style={'margin': '5px 0'})
        ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#F8F9FA', 'borderRadius': '5px'}),
        
        # 統計摘要表格
        html.Div([
            html.H4("📊 統計摘要", style={'color': '#34495E', 'marginBottom': '15px'}),
            html.Table(stats_table, style={
                'width': '100%', 
                'borderCollapse': 'collapse',
                'marginBottom': '20px'
            })
        ]),
        
        # 詞彙展示
        html.Div([
            html.H4("📝 詞彙庫", style={'color': '#34495E', 'marginBottom': '15px'}),
            html.Div(word_display)
        ], style={'marginBottom': '30px'}),
        
        # 滑動窗口圖表
        html.Div([
            dcc.Graph(figure=fig_window)
        ], style={'marginBottom': '20px'}),
        
        # 散點分布圖
        html.Div([
            dcc.Graph(figure=fig_scatter)
        ]),
        
        # 分析說明
        html.Div([
            html.H4("📋 分析說明", style={'color': '#34495E', 'marginTop': '30px', 'marginBottom': '15px'}),
            html.Ul([
                html.Li("滑動窗口圖表顯示選中意象在文本中的時序變化模式", style={'marginBottom': '5px'}),
                html.Li("散點圖顯示每個意象標籤在具體句子中的分布位置", style={'marginBottom': '5px'}),
                html.Li("統計表格提供量化的重複性指標", style={'marginBottom': '5px'}),
                html.Li("詞彙庫展示該書中各標籤的具體意象詞彙（基於GPT標注）", style={'marginBottom': '5px'}),
                html.Li("顏色與文本高亮顯示保持一致，便於對照分析", style={'marginBottom': '5px'})
            ], style={'color': '#7F8C8D', 'fontSize': '14px'})
        ])
    ])

# Run app
if __name__ == '__main__':
    app.run(debug=True)