# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


sns.set()


@st.cache
def set_session_state():
    if 'last_click' not in st.session_state:
        st.session_state.last_click = 0
    if 'is_click_preprocess' not in st.session_state:
        st.session_state.is_click_preprocess = False
    if 'is_click_split' not in st.session_state:
        st.session_state.is_click_split = False
    if 'is_click_train' not in st.session_state:
        st.session_state.is_click_train = False
    if 'log_features' not in st.session_state:
        st.session_state.log_features = []
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = pd.DataFrame()
    if 'x_train' not in st.session_state:
        st.session_state.x_train = []
    if 'x_val' not in st.session_state:
        st.session_state.x_val = []
    if 'y_train' not in st.session_state:
        st.session_state.y_train = []
    if 'y_val' not in st.session_state:
        st.session_state.y_val = []
    if 'y_pred_train' not in st.session_state:
        st.session_state.y_pred_train = []
    if 'y_pred_val' not in st.session_state:
        st.session_state.y_pred_val = []


def trans_screen(process: int):
    if process == 0:
        main_screen()
        st.stop()
    elif process == 1:
        preprocess_data()
        st.session_state.is_click_preprocess = True
    elif process == 2 and st.session_state.is_click_preprocess:
        split_data()
        st.session_state.is_click_split = True
    elif process == 3 and st.session_state.is_click_preprocess and st.session_state.is_click_split:
        train_data()
        st.session_state.is_click_train = True
    elif process == 4 and st.session_state.is_click_preprocess and st.session_state.is_click_split and st.session_state.is_click_train:
        view_results()
    elif process == 5 and st.session_state.is_click_preprocess and st.session_state.is_click_split and st.session_state.is_click_train:
        download_result()
    else:
        view_error_screen()


def main_screen():
    # タイトルの表示
    st.title('カリフォルニアの住宅価格の重回帰分析')
    st.header('本アプリはStreamlit学習用に作成したアプリです')
    st.write('左のサイドバーの1から順番に実行してください')
    st.write('設定を元に戻したい場合は設定のリセットをクリックしてください')


def view_error_screen():
    st.title('エラー画面')
    st.write('誤った順番で実行しようとしていませんか？')
    st.write('サイドバーの順番通りに実行してください')


def preprocess_data():
    st.header('データの確認及び前処理')
    # データの読み込み
    dataset = fetch_california_housing()
    # 説明変数をpandasのDataFrame型で作成
    df = pd.DataFrame(dataset.data)
    # 説明変数名をカラム名も割り当てる
    df.columns = dataset.feature_names
    # 目的変数をカラム名PRICESで作成
    df['PRICES'] = dataset.target

    # チェックボックスがONの時、データセットを表示する
    if st.checkbox('テーブルデータ形式でデータセットを表示'):
        st.dataframe(df)

    # チェック時に目的変数と説明変数の相関を表示
    if st.checkbox('目的変数と説明変数の相関を可視化'):
        # 可視化する説明変数を選択
        checked_variable = st.selectbox(
            '説明変数を1つ選択してください',
            df.drop(columns='PRICES').columns
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(x=df[checked_variable], y=df['PRICES'])
        plt.xlabel(checked_variable)
        plt.ylabel('PRICES')
        st.pyplot(fig)

    # 前処理
    features_not_used = st.multiselect(
        '学習時に使用しない変数を選択してください',
        df.columns
    )
    df = df.drop(columns=features_not_used)

    # 対数変換の実施有無を選択
    left_column, right_column = st.columns(2)
    ans_log = left_column.radio(
        '対数変換を行いますか？',
        ('No', 'Yes')
    )
    df_log, log_features = df.copy(), []
    if ans_log == 'Yes':
        log_features = right_column.multiselect(
            '対数変換を適用する目的変数もしくは説明変数を選択してください',
            df.columns
        )
        df_log[log_features] = np.log(df_log[log_features])
        st.write('対数変換後の目的変数との相関を表示')
        for feature in log_features:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(x=df_log[feature], y=df_log['PRICES'])
            plt.xlabel(feature)
            plt.ylabel('PRICES')
            st.pyplot(fig)
    
    # 標準化の実施有無を選択
    left_column, right_column = st.columns(2)
    ans_std = left_column.radio(
        '標準化を実施しますか？',
        ('No', 'Yes')
    )
    df_std, std_features = df_log.copy(), []
    if ans_std == 'Yes':
        std_features = right_column.multiselect(
            '標準化を適用する変数を選択してください',
            df_log.drop(columns=['PRICES']).columns
        )
        if std_features == []:
            st.stop()
        scaler = preprocessing.StandardScaler()
        scaler.fit(df_std[std_features])
        df_std[std_features] = scaler.transform(df_std[std_features])
        st.write('標準化後の目的変数との相関を表示')
        for feature in std_features:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(x=df_std[feature], y=df_std['PRICES'])
            plt.xlabel(feature)
            plt.ylabel('PRICES')
            st.pyplot(fig)

    st.session_state.preprocessed_data = df_std
    st.session_state.log_features = log_features
    st.write('これでよければ2. データの分割を押してください')


def split_data():
    st.header('データの分割')
    df = st.session_state.preprocessed_data
    # データセットを訓練用と検証用に分割
    left_column, right_column = st.columns(2)
    test_size = left_column.number_input(
        '検証用データの比率',
        min_value = 0.,
        max_value = 1.,
        value = 0.2,
        step = 0.1
    )
    random_seed = right_column.number_input(
        'ランダムシードの設定',
        value = 0,
        step = 1,
        min_value = 0
    )

    # データセットを分割
    X_train, X_val, Y_train, Y_val = train_test_split(
        df.drop(columns=['PRICES']),
        df['PRICES'],
        test_size = test_size,
        random_state = random_seed
    )
    st.session_state.x_train, st.session_state.x_val, st.session_state.y_train, st.session_state.y_val = X_train, X_val, Y_train, Y_val
    st.write('これでよければ3. 学習を押してください')


def train_data():
    st.header('学習')
    # 回帰分析モデルのインスタンスを作成
    regressor = LinearRegression()
    regressor.fit(st.session_state.x_train, st.session_state.y_train)
    Y_pred_train = regressor.predict(st.session_state.x_train)
    Y_pred_val = regressor.predict(st.session_state.x_val)
    if 'PRICES' in st.session_state.log_features:
        Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
        Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)
    st.write('入力データ')
    left_column, right_column = st.columns(2)
    left_column.write('訓練データ')
    right_column.write('検証用データ')
    left_column, right_column = st.columns(2)
    left_column.write(st.session_state.x_train)
    right_column.write(st.session_state.x_val)
    st.write('正解データ')
    left_column, right_column = st.columns(2)
    left_column.write('訓練データ')
    right_column.write('検証用データ')
    left_column, right_column = st.columns(2)
    left_column.write(st.session_state.y_train)
    right_column.write(st.session_state.y_val)
    st.write('予測データ')
    left_column, right_column = st.columns(2)
    left_column.write('訓練データ')
    right_column.write('検証用データ')
    left_column, right_column = st.columns(2)
    left_column.write(Y_pred_train)
    right_column.write(Y_pred_val)
    st.session_state.y_pred_train = Y_pred_train
    st.session_state.y_pred_val = Y_pred_val
    st.write('これでよければ4. 結果の表示を押してください')


def view_results():
    st.header('結果の表示')
    Y_train, Y_pred_train = st.session_state.y_train, st.session_state.y_pred_train
    Y_val, Y_pred_val = st.session_state.y_val, st.session_state.y_pred_val
    # 結果の表示
    # モデル精度
    r2 = r2_score(Y_val, Y_pred_val)
    st.write(f'R2 値 :{r2:.2f}')
    # グラフ描画
    left_column, right_column = st.columns(2)
    show_train = left_column.radio(
        '訓練データの結果を表示',
        ('Yes', 'No')
    )
    show_val = right_column.radio(
        '検証用データの結果を表示',
        ('Yes', 'No')
    )
    y_max_train = max(max(Y_train), max(Y_pred_train))
    y_max_val = max(max(Y_val), max(Y_pred_val))
    y_max = int(max([y_max_train, y_max_val]))
    # 動的に軸範囲を変更可能にする
    left_column, right_column = st.columns(2)
    x_min = left_column.number_input('x軸の最小値 :', value=0, step=1)
    x_max = right_column.number_input('x軸の最大値 :', value=y_max, step=1)
    left_column, right_column = st.columns(2)
    y_min = left_column.number_input('y軸の最小値 :', value=0, step=1)
    y_max = right_column.number_input('y軸の最大値 :', value=y_max, step=1)
    # 結果の表示
    fig = plt.figure(figsize=(3, 3))
    if show_train == 'Yes':
        plt.scatter(Y_train, Y_pred_train, lw=0.1, color='r', label='training data')
    if show_val == 'Yes':
        plt.scatter(Y_val, Y_pred_val, lw=0.1, color='r', label='validation data')
    plt.xlabel('PRICES', fontsize=8)
    plt.ylabel('Prediction', fontsize=8)
    plt.xlim(int(x_min), int(x_max)+5)
    plt.ylim(int(y_min), int(y_max)+5)
    plt.legend(fontsize=6)
    plt.tick_params(labelsize=6)
    st.pyplot(fig)


def download_result():
    st.title('結果をダウンロード')
    csv_output = pd.DataFrame(st.session_state.y_pred_val, columns=['PRICES']).to_csv().encode('utf-8')
    bool_download = st.download_button(
        label='ダウンロード',
        data=csv_output,
        file_name='予測結果.csv',
        mime='text/csv'
    )
    if bool_download:
        st.write('finish!')


def main():
    set_session_state()
    is_preprocess_click = st.sidebar.button('1. 基礎集計')
    is_split_click = st.sidebar.button('2. データの分割')
    is_train_click = st.sidebar.button('3. 学習')
    is_result_click = st.sidebar.button('4. 結果の表示')
    is_download_click = st.sidebar.button('5. 結果のダウンロード')
    if is_preprocess_click:
        st.session_state.last_click = 1
        trans_screen(st.session_state.last_click)
    elif is_split_click:
        st.session_state.last_click = 2
        trans_screen(st.session_state.last_click)
    elif is_train_click:
        st.session_state.last_click = 3
        trans_screen(st.session_state.last_click)
    elif is_result_click:
        st.session_state.last_click = 4
        trans_screen(st.session_state.last_click)
    elif is_download_click:
        st.session_state.last_click = 5
        trans_screen(st.session_state.last_click)
    else:
        trans_screen(st.session_state.last_click)


if __name__ == '__main__':
    main()