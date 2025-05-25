# repeatness_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. 指定一個支援中文的系統字型（macOS 常見內建字型）
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Heiti TC',     # 黑體
                                   'Songti SC',    # 宋體
                                   'PingFang TC']  # 微軟正黑體／苹方

# 2. 解決負號 '-' 顯示成方塊的問題
mpl.rcParams['axes.unicode_minus'] = False

#─── 1. 讀資料 ───────────────────────────────────────────
def load_data(ima_path: str, books_path: str):
    """
    讀入 imagery_dictionary.pkl 和 df_books.pkl，
    並且把 df_ima 裡的 'idx=_in_text' 改成 'idx_in_text'。
    同時拆解 df_books.book -> author, year, title，再把 book=title。
    返回 (df_ima, df_books)
    """
    df_ima = pd.read_pickle(ima_path)
    df_books = pd.read_pickle(books_path)
    # 處理 df_books
    # 1. 保留原始 book 欄位作為備份
    df_books['orig_book'] = df_books['book'].copy()

    # 2. 去掉 .txt、split 成三段 (作者,年份,書名)
    tmp = (
        df_books['orig_book']
        .str.replace(r'\.txt$', '', regex=True)
        .str.split('-', n=2, expand=True)
    )
    
    
    # 4. 確保有三個部分，否則處理異常情況
    if tmp.shape[1] >= 3:
        df_books['author'] = tmp[0]
        df_books['year'] = pd.to_numeric(tmp[1], errors='coerce').astype('Int64')  # 使用 Int64 處理可能的 NaN
        df_books['title'] = tmp[2]
        
        # 關鍵：將 book 欄位更新為純書名（title）
        df_books['book'] = df_books['title'].copy()
        
    else:
        print("警告：某些書籍名稱格式不符合預期")
        # 提供備用處理方案
        df_books['author'] = 'Unknown'
        df_books['year'] = None
        df_books['book'] = df_books['orig_book']


    book_mapping = dict(zip(df_books['orig_book'], df_books['book']))
    

    # 更新 df_ima 的 book 欄位
    df_ima['book'] = df_ima['book'].map(book_mapping).fillna(df_ima['book'])
 
    df_books = df_books.drop(columns=['orig_book'])
    df_books = df_books.drop(columns=['title'])

    print(f"\n最終結果統計:")
    print(f"df_ima 形狀: {df_ima.shape}")
    print(f"df_books 形狀: {df_books.shape}")
    print(f"df_ima 中的唯一書籍數: {df_ima['book'].nunique()}")
    print(f"df_books 中的唯一書籍數: {df_books['book'].nunique()}")
    
    return df_ima, df_books

#─── 2. Inter-book Heatmap ─────────────────────────────────
def plot_small_label_heatmap(df_ima: pd.DataFrame, books: list = None, figsize=(12,8)):
    mat = (df_ima.groupby(['small_label','book']).size()
             .unstack(fill_value=0))
    if books is not None:
        mat = mat[books]
    plt.figure(figsize=figsize)
    plt.imshow(mat, aspect='auto')
    plt.colorbar(label='count')
    plt.yticks(np.arange(len(mat)), mat.index, fontsize=10)
    plt.xticks(np.arange(len(mat.columns)), mat.columns, rotation=90)
    plt.title('Imagery counts by big_label × book')
    plt.tight_layout()
    plt.show()

#─── 3. Intra-book Timeline ─────────────────────────────────



