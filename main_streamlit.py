import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import requests
from bs4 import BeautifulSoup
from retry import retry
import datetime

# CSSスタイルを追加
st.markdown("""
    <style>
        .main-title { color: #2e86de; font-size: 32px; font-weight: bold; }
        .section-title { color: #16a085; font-size: 24px; font-weight: bold; margin-top: 40px; }
        .property-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
        .property-link { color: #3498db; font-weight: bold; text-decoration: none; }
    </style>
""", unsafe_allow_html=True)

# 日本語フォントの設定
font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'IPAexGothic'

# スクレイピング関数
@retry(tries=3, delay=10, backoff=2)
def get_html(url):
    r = requests.get(url)
    return BeautifulSoup(r.content, "html.parser")

def scrape_suumo_data():
    base_url = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=14&sc=14205&sc=14207&cb=0.0&ct=9999999&et=9999999&cn=9999999&mb=0&mt=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&fw2=&srch_navi=1&page={}"
    max_page = 157
    all_data = []

    for page in range(1, max_page + 1):
        soup = get_html(base_url.format(page))
        items = soup.findAll("div", {"class": "cassetteitem"})
        
        for item in items:
            stations = item.findAll("div", {"class": "cassetteitem_detail-text"})
            
            for station in stations:
                base_data = {
                    "名称": item.find("div", {"class": "cassetteitem_content-title"}).getText().strip(),
                    "カテゴリー": item.find("div", {"class": "cassetteitem_content-label"}).getText().strip(),
                    "アドレス": item.find("li", {"class": "cassetteitem_detail-col1"}).getText().strip(),
                    "アクセス": station.getText().strip(),
                    "築年数": item.find("li", {"class": "cassetteitem_detail-col3"}).findAll("div")[0].getText().strip(),
                    "構造": item.find("li", {"class": "cassetteitem_detail-col3"}).findAll("div")[1].getText().strip(),
                }
                
                tbodys = item.find("table", {"class": "cassetteitem_other"}).findAll("tbody")
                
                for tbody in tbodys:
                    data = base_data.copy()
                    data.update({
                        "階数": tbody.findAll("td")[2].getText().strip(),
                        "家賃": tbody.findAll("td")[3].findAll("li")[0].getText().strip(),
                        "管理費": tbody.findAll("td")[3].findAll("li")[1].getText().strip(),
                        "敷金": tbody.findAll("td")[4].findAll("li")[0].getText().strip(),
                        "礼金": tbody.findAll("td")[4].findAll("li")[1].getText().strip(),
                        "間取り": tbody.findAll("td")[5].findAll("li")[0].getText().strip(),
                        "面積": tbody.findAll("td")[5].findAll("li")[1].getText().strip(),
                        "URL": "https://suumo.jp" + tbody.findAll("td")[8].find("a").get("href"),
                    })
                    all_data.append(data)

    # データフレームに変換
    df = pd.DataFrame(all_data)
    
    # CSVに保存
    today = datetime.datetime.now().strftime("%Y%m%d")
    csv_filename = f"{today}_suumo.csv"
    df.to_csv(csv_filename, index=False)
    st.success(f"データが {csv_filename} に保存されました。")
    return csv_filename

# Streamlitアプリケーション
st.title("茅ヶ崎、藤沢のスーモデータ分析")

# スクレイピングの実行ボタン
if st.button("スクレイピングを実行"):
    scraped_file = scrape_suumo_data()

# 保存済みCSVファイルの選択
csv_files = [file for file in os.listdir() if file.endswith('_suumo.csv')]
if csv_files:
    selected_file = st.selectbox("読み込むCSVファイルを選択してください:", csv_files)
    if selected_file:
        origin_df = pd.read_csv(selected_file)

        # データの前処理
        origin_df["家賃_万円"] = origin_df["家賃"].str.replace("万円", "").str.replace("円", "").str.replace(",", "").astype(float)
        origin_df["面積_m2"] = origin_df["面積"].str.replace("m2", "").astype(float)
        origin_df["築年数"] = origin_df["築年数"].str.extract(r'(\d+)').astype(float)

        # 不動産形態の選択
        st.sidebar.markdown('<h2 class="section-title">絞り込み</h2>', unsafe_allow_html=True)
        keitai_list = st.sidebar.multiselect(
            '不動産の形態を選んでください',
            ['賃貸アパート', '賃貸マンション', '賃貸一戸建て', '賃貸テラス・タウンハウス']
        )

        # 家賃、面積、間取りでのフィルタリング
        min_rent, max_rent = st.sidebar.slider('家賃の範囲（万円）', 0, 100, (0, 50))
        min_area, max_area = st.sidebar.slider('面積の範囲（㎡）', 0, 200, (0, 100))
        layout_options = st.sidebar.multiselect('間取りを選んでください', origin_df['間取り'].unique())

        # フィルタリング
        select_df = origin_df[
            (origin_df['カテゴリー'].isin(keitai_list)) &
            (origin_df['家賃_万円'] >= min_rent) &
            (origin_df['家賃_万円'] <= max_rent) &
            (origin_df['面積_m2'] >= min_area) &
            (origin_df['面積_m2'] <= max_area) &
            (origin_df['間取り'].isin(layout_options) if layout_options else True)
        ]

        st.markdown('<h2 class="section-title">データプレビュー</h2>', unsafe_allow_html=True)

        if select_df.empty:
            st.write("選択された条件に該当する物件がありません。別の条件を選択してください。")
        else:
            st.dataframe(select_df)

            # ソート済み家賃分布のヒストグラム
            st.markdown('<h2 class="section-title">ソート済み家賃の分布</h2>', unsafe_allow_html=True)
            st.bar_chart(select_df['家賃_万円'].value_counts().sort_index())

            # 面積と家賃の関係の散布図（URLリンク追加）
            st.markdown('<h2 class="section-title">面積と家賃の関係</h2>', unsafe_allow_html=True)
            scatter_fig = px.scatter(
                select_df,
                x='面積_m2', y='家賃_万円',
                hover_name='名称',
                hover_data={'URL': True, '面積_m2': ':.2f', '家賃_万円': ':.2f'},
                title="面積と家賃の関係",
            )

            scatter_fig.update_traces(marker=dict(size=10, color='LightSkyBlue'))
            scatter_fig.update_layout(clickmode='event+select')
            st.plotly_chart(scatter_fig)

            # 基本統計量表示
            st.markdown('<h2 class="section-title">基本統計量</h2>', unsafe_allow_html=True)
            st.write(select_df[['家賃_万円', '面積_m2', '築年数']].describe())

            # 機械学習部分
            if st.button("機械学習で割安物件を探す"):
                # 前処理と特徴量の準備
                select_df['築年数'] = select_df['築年数'].fillna(select_df['築年数'].mean())
                select_df['面積_m2'] = select_df['面積_m2'].fillna(select_df['面積_m2'].mean())
                select_df = select_df.dropna(subset=['構造'])

                le_structure = LabelEncoder()
                select_df['構造_encoded'] = le_structure.fit_transform(select_df['構造'])

                # 特徴量とターゲットを定義
                X = select_df[['面積_m2', '築年数', '構造_encoded']]
                y = select_df['家賃_万円']

                # 学習と予測
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                select_df['予測家賃_万円'] = model.predict(X)

                # モデルパフォーマンスの表示
                mse = mean_squared_error(y_test, model.predict(X_test))
                r2 = r2_score(y_test, model.predict(X_test))
                st.markdown('<h2 class="section-title">モデルのパフォーマンス</h2>', unsafe_allow_html=True)
                st.write(f"平均二乗誤差 (MSE): {mse:.2f}")
                st.write(f"決定係数 (R²): {r2:.2f}")

                # SHAPによる特徴量の寄与度解析
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer(X)

                # 割安度の計算
                select_df['割安度'] = select_df['予測家賃_万円'] - select_df['家賃_万円']
                bargain_df = select_df.sort_values(by='割安度', ascending=False).head(10).reset_index(drop=True)

                # 割安物件トップ10表示
                st.markdown('<h2 class="section-title">割安な物件トップ10</h2>', unsafe_allow_html=True)
                for idx, row in bargain_df.iterrows():
                    st.markdown(f"""
                        <div class="property-card">
                            <h3>{row['名称']} ({row['カテゴリー']})</h3>
                            <p><strong>アドレス:</strong> {row['アドレス']}</p>
                            <p><strong>アクセス:</strong> {row['アクセス']}</p>
                            <p><strong>家賃:</strong> {row['家賃']}万円</p>
                            <p><strong>予測家賃:</strong> {round(row['予測家賃_万円'], 2)}万円</p>
                            <p><strong>面積:</strong> {row['面積_m2']} m²</p>
                            <p><strong>築年数:</strong> {row['築年数']}年</p>
                            <p><strong>構造:</strong> {row['構造']}</p>
                            <a class="property-link" href="{row['URL']}" target="_blank">物件リンク</a>
                        </div>
                    """, unsafe_allow_html=True)

                    # SHAP寄与度可視化
                    shap_values_bargain = shap.Explanation(values=shap_values.values[idx], base_values=shap_values.base_values[idx], data=X.iloc[idx])
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.waterfall_plot(shap_values_bargain)
                    st.pyplot(fig)
                    plt.close()
