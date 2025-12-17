# app.py
# Streamlit: JPYC 可視化・分析ダッシュボード（Etherscan API V2 / 読み取り専用）
#
# 使い方:
# 1) pip install streamlit requests pandas plotly python-dotenv
# 2) .env に ETHERSCAN_API_KEY=xxxxx を入れる（または環境変数に設定）
# 3) streamlit run app.py

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px

# -------------------------
# Config
# -------------------------
load_dotenv()

BASE_URL = "https://api.etherscan.io/v2/api"

# よく使うチェーンの chainid（Etherscan V2 の指定）
CHAINID_MAP = {
    "Polygon Mainnet": 137,
    "Polygon Amoy (testnet)": 80002,
    "Ethereum Mainnet": 1,
    "Ethereum Sepolia (testnet)": 11155111,
    "Gnosis Mainnet": 100,
    "Gnosis Chiado (testnet)": 10200,
    "Avalanche C-Chain Mainnet": 43114,
    "Avalanche Fuji (testnet)": 43113,
    "Astar Mainnet": 592,
}

# JPYC（JPY Coin）Polygon のコントラクト（既知のよく使われるやつ）
DEFAULT_JPYC_POLYGON = "0xe7c3d8c9a439fede00d2600032d5db0be71c3c29"


# -------------------------
# Helpers
# -------------------------
def get_api_key() -> Optional[str]:
    # ローカルだと secrets.toml が無いと例外になることがあるので try/except で守る
    try:
        v = st.secrets.get("ETHERSCAN_API_KEY", None)
        if v:
            return str(v).strip()
    except Exception:
        pass

    v = os.getenv("ETHERSCAN_API_KEY")
    return str(v).strip() if v else None



def is_valid_address(addr: str) -> bool:
    addr = (addr or "").strip()
    return addr.startswith("0x") and len(addr) == 42


def etherscan_v2_get(params: Dict[str, str]) -> Tuple[Dict[str, Any], str]:
    """Etherscan V2 API をGETして (json, final_url) を返す"""
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json(), r.url


def normalize_etherscan_error(data: Dict[str, Any]) -> str:
    """Etherscanのエラーっぽいレスポンスを読みやすく整形"""
    status = data.get("status")
    message = data.get("message")
    result = data.get("result")
    return f"status={status} message={message} result={result}"


def test_tokenbalance(chainid: int, contract: str, address: str, apikey: str) -> Tuple[Dict[str, Any], str]:
    """軽いAPIで疎通確認（tokenbalance）"""
    params = {
        "chainid": str(chainid),
        "module": "account",
        "action": "tokenbalance",
        "contractaddress": contract,
        "address": address,
        "tag": "latest",
        "apikey": apikey,
    }
    data, url = etherscan_v2_get(params)
    return data, url


@st.cache_data(show_spinner=False, ttl=60)
def fetch_tokentx_cached(
    chainid: int,
    contract: str,
    address: str,
    apikey: str,
    offset: int,
    sort: str,
    max_pages: int,
    sleep_s: float,
) -> pd.DataFrame:
    """tokentx をページングして全部集める（キャッシュ付き）"""
    rows: List[Dict[str, Any]] = []
    page = 1

    while True:
        if page > max_pages:
            break

        params = {
            "chainid": str(chainid),
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract,
            "address": address,
            "page": str(page),
            "offset": str(offset),
            "sort": sort,
            "apikey": apikey,
        }

        data, url = etherscan_v2_get(params)
        result = data.get("result")

        # 失敗/不安定系を全部拾う
        if result is None:
            raise RuntimeError(f"Etherscan returned null result. {normalize_etherscan_error(data)} url={url} full={data}")

        if isinstance(result, str):
            # "No transactions found" など
            if "No transactions" in result:
                return pd.DataFrame()
            raise RuntimeError(f"Etherscan API error: {normalize_etherscan_error(data)} url={url}")

        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected API response shape. {normalize_etherscan_error(data)} url={url}")

        if len(result) == 0:
            break

        rows.extend(result)

        if len(result) < offset:
            break

        page += 1
        time.sleep(sleep_s)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 必須カラムチェック
    required = {"timeStamp", "value", "tokenDecimal", "from", "to", "hash"}
    missing = required - set(df.columns)
    if missing:
        sample = df.iloc[0].to_dict() if len(df) else {}
        raise RuntimeError(f"Missing columns {missing}. columns={list(df.columns)} sample_row={sample}")

    # 整形
    df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s", utc=True).dt.tz_convert("Asia/Tokyo")
    df["tokenDecimal"] = pd.to_numeric(df["tokenDecimal"], errors="coerce").fillna(0).astype(int)
    df["value_raw"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).astype("int64")
    df["amount"] = df["value_raw"] / (10 ** df["tokenDecimal"])

    addr = address.lower()
    df["direction"] = df.apply(lambda r: "IN" if str(r["to"]).lower() == addr else "OUT", axis=1)
    df["signed_amount"] = df.apply(lambda r: r["amount"] if r["direction"] == "IN" else -r["amount"], axis=1)

    # 相手先
    df["counterparty"] = df.apply(
        lambda r: (r["from"] if r["direction"] == "IN" else r["to"]),
        axis=1,
    )

    return df


def shorten(addr: str, head: int = 6, tail: int = 4) -> str:
    if not isinstance(addr, str) or len(addr) < head + tail:
        return str(addr)
    return f"{addr[:head]}...{addr[-tail:]}"


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="JPYC 可視化・分析（PoC）", layout="wide")

st.title("JPYC 可視化・分析ダッシュボード（PoC）")
st.caption("Etherscan API V2（読み取り専用）で、特定ウォレットの JPYC Transfer を可視化します。")

apikey = get_api_key()
if not apikey:
    st.error("ETHERSCAN_API_KEY が見つからんかったばい。.env か環境変数、もしくは st.secrets に設定してね。")
    st.stop()

with st.sidebar:
    st.header("設定")

    chain_label = st.selectbox("チェーン", list(CHAINID_MAP.keys()), index=0)
    chainid = CHAINID_MAP[chain_label]

    contract = st.text_input("JPYC コントラクトアドレス", value=DEFAULT_JPYC_POLYGON).strip().lower()

    st.divider()
    st.subheader("取得パラメータ（重い時は軽くしてね）")
    offset = st.slider("offset（1ページ件数）", min_value=10, max_value=1000, value=200, step=10)
    max_pages = st.slider("max_pages（最大ページ数）", min_value=1, max_value=50, value=10, step=1)
    sort = st.selectbox("sort", ["asc", "desc"], index=0)
    sleep_s = st.slider("sleep（ページ間ウェイト秒）", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    debug_mode = st.checkbox("デバッグ情報を表示", value=True)

st.subheader("分析対象")
address = st.text_input("ウォレットアドレス（0x...）", placeholder="0x1234...").strip()

colA, colB = st.columns([1, 1])
with colA:
    do_test = st.button("① まず疎通確認（tokenbalance）", use_container_width=True)
with colB:
    do_fetch = st.button("② 取引履歴を取得して可視化（tokentx）", use_container_width=True)

if address and not is_valid_address(address):
    st.warning("アドレス形式が変たい（0xで始まって42文字か確認してね）")

# -------------------------
# Actions
# -------------------------
if do_test:
    if not is_valid_address(address):
        st.error("まず正しいアドレスを入れてね。")
        st.stop()

    try:
        data, url = test_tokenbalance(chainid, contract, address, apikey)
        st.success("tokenbalance の疎通OK（またはエラー内容が取れた）")
        st.code(url)
        st.json(data)

        # 典型エラーを分かりやすく
        if str(data.get("status")) == "0":
            st.error(f"APIが失敗扱い：{normalize_etherscan_error(data)}")
    except Exception as e:
        st.exception(e)

if do_fetch:
    if not is_valid_address(address):
        st.error("まず正しいアドレスを入れてね。")
        st.stop()

    with st.spinner("取引履歴を取得中..."):
        try:
            df = fetch_tokentx_cached(
                chainid=chainid,
                contract=contract,
                address=address,
                apikey=apikey,
                offset=offset,
                sort=sort,
                max_pages=max_pages,
                sleep_s=sleep_s,
            )
        except Exception as e:
            st.error("取得に失敗したばい。下のエラーとデバッグ情報を見てね。")
            st.exception(e)
            st.stop()

    if df.empty:
        st.warning("取引データが見つからんかったばい（アドレス/チェーン/コントラクトが合ってるか確認してね）")
        st.stop()

    # デバッグ表示
    if debug_mode:
        st.write("取得件数:", len(df))
        st.write("期間:", df["timeStamp"].min(), "〜", df["timeStamp"].max())
        st.write("columns:", list(df.columns))

    # KPI
    st.subheader("KPI")
    total_in = df.loc[df["direction"] == "IN", "amount"].sum()
    total_out = df.loc[df["direction"] == "OUT", "amount"].sum()
    net = total_in - total_out

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("総受取（IN）JPYC", f"{total_in:,.2f}")
    k2.metric("総送付（OUT）JPYC", f"{total_out:,.2f}")
    k3.metric("ネット（IN-OUT）JPYC", f"{net:,.2f}")
    k4.metric("取引件数", f"{len(df):,}")

    # 期間フィルタ
    st.subheader("期間フィルタ")
    min_dt = df["timeStamp"].min().to_pydatetime()
    max_dt = df["timeStamp"].max().to_pydatetime()
    start_dt, end_dt = st.date_input(
        "表示期間（日付）",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    df_f = df[(df["timeStamp"].dt.date >= start_dt) & (df["timeStamp"].dt.date <= end_dt)].copy()
    if df_f.empty:
        st.warning("この期間にはデータが無いばい。")
        st.stop()

    # 日次ネットフロー
    st.subheader("日次ネットフロー（signed）")
    daily = (
        df_f.set_index("timeStamp")["signed_amount"]
        .resample("D")
        .sum()
        .reset_index()
    )
    fig1 = px.line(daily, x="timeStamp", y="signed_amount")
    st.plotly_chart(fig1, use_container_width=True)

    # IN/OUTの内訳（週次でも見たい時）
    st.subheader("日次 IN/OUT")
    daily_inout = (
        df_f.assign(date=df_f["timeStamp"].dt.floor("D"))
        .pivot_table(index="date", columns="direction", values="amount", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    # 欠けてても列を揃える
    if "IN" not in daily_inout.columns:
        daily_inout["IN"] = 0.0
    if "OUT" not in daily_inout.columns:
        daily_inout["OUT"] = 0.0

    daily_long = daily_inout.melt(id_vars=["date"], value_vars=["IN", "OUT"], var_name="direction", value_name="amount")
    fig2 = px.bar(daily_long, x="date", y="amount", color="direction")
    st.plotly_chart(fig2, use_container_width=True)

    # 相手先ランキング
    st.subheader("相手先ランキング（上位20）")
    top = (
        df_f.groupby("counterparty")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    top["counterparty_short"] = top["counterparty"].apply(shorten)
    fig3 = px.bar(top, x="counterparty_short", y="amount")
    st.plotly_chart(fig3, use_container_width=True)

    # 生データ
    st.subheader("取引一覧（最新順）")
    view_cols = ["timeStamp", "direction", "amount", "from", "to", "hash"]
    show = df_f.sort_values("timeStamp", ascending=False)[view_cols].copy()
    show["from"] = show["from"].apply(shorten)
    show["to"] = show["to"].apply(shorten)
    show["hash"] = show["hash"].apply(lambda x: shorten(str(x), 10, 6))
    st.dataframe(show, use_container_width=True)

    # CSVダウンロード
    st.download_button(
        "CSVでダウンロード",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="jpyc_tokentx.csv",
        mime="text/csv",
        use_container_width=True,
    )
