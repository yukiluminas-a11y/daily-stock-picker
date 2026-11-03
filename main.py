import datetime
import pandas as pd
import yfinance as yf

# ==========================================
# 🌍 グローバル監視銘柄リスト（日本株・外国株）
# ==========================================
GLOBAL_TICKERS = [
    # --- 米国メガテック & AI銘柄 ---
    {"symbol": "NVDA",  "name": "NVIDIA", "market": "米国"},
    {"symbol": "AAPL",  "name": "Apple", "market": "米国"},
    {"symbol": "MSFT",  "name": "Microsoft", "market": "米国"},
    {"symbol": "AMZN",  "name": "Amazon", "market": "米国"},
    {"symbol": "GOOGL", "name": "Alphabet (Google)", "market": "米国"},
    {"symbol": "TSLA",  "name": "Tesla", "market": "米国"},
    {"symbol": "AVGO",  "name": "Broadcom", "market": "米国"},
    
    # --- 米国高配当 & セクターリーダー ---
    {"symbol": "LLY",   "name": "Eli Lilly (製薬)", "market": "米国"},
    {"symbol": "JPM",   "name": "JPMorgan Chase (金融)", "market": "米国"},
    {"symbol": "WMT",   "name": "Walmart (消費財)", "market": "米国"},

    # --- 日本主要株 (東証) ---
    {"symbol": "7203.T", "name": "トヨタ自動車", "market": "日本"},
    {"symbol": "6758.T", "name": "ソニーグループ", "market": "日本"},
    {"symbol": "9984.T", "name": "ソフトバンクグループ", "market": "日本"},
    {"symbol": "8306.T", "name": "三菱UFJ FG", "market": "日本"},
    {"symbol": "6861.T", "name": "キーエンス", "market": "日本"},
    {"symbol": "8035.T", "name": "東京エレクトロン", "market": "日本"},

    # --- 欧州 & グローバル ADR ---
    {"symbol": "ASML",   "name": "ASML Holding (半導体露光/オランダ)", "market": "欧州(ADR)"},
    {"symbol": "TSM",    "name": "TSMC (半導体ファウンドリ/台湾)", "market": "アジア(ADR)"},
    {"symbol": "NVO",    "name": "Novo Nordisk (製薬/デンマーク)", "market": "欧州(ADR)"},
]

def analyze_stock(ticker_info):
    symbol = ticker_info["symbol"]
    name = ticker_info["name"]
    market = ticker_info["market"]

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        if len(df) < 100:
            return None

        currency = "￥" if market == "日本" else "$"
        latest_price = df['Close'].iloc[-1]
        
        # --- テクニカル指標の計算 ---
        # 1. 移動平均線 (MA)
        ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = df['Close'].rolling(window=200).mean().iloc[-1]
        
        # 2. RSI (14日)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # 3. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_prev = signal.iloc[-2]

        # 4. ボリンジャーバンド (20日, 2σ)
        std20 = df['Close'].rolling(window=20).std().iloc[-1]
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        
        # 5. 出来高変化率
        vol_latest = df['Volume'].iloc[-1]
        vol_avg5 = df['Volume'].rolling(window=5).mean().iloc[-1]
        vol_ratio = (vol_latest / vol_avg5) if vol_avg5 > 0 else 1.0

        # ==========================================
        # 🧠 プロの売買判断アルゴリズム (スコアリング)
        # ==========================================
        score = 0
        reasons = []

        # 【トレンド判定】
        if latest_price > ma50 and ma20 > ma50:
            score += 25
            reasons.append("中期上昇トレンド維持（株価 > 50日線 & 20日線 > 50日線）")
        elif latest_price < ma50 and ma20 < ma50:
            score -= 25
            reasons.append("下降トレンド継続中（50日線を下回る強気リスクあり）")

        if latest_price > ma200:
            score += 15
            reasons.append("長期強気相場（200日移動平均線の上で推移）")
        else:
            score -= 15
            reasons.append("長期弱気相場（200日移動平均線を下回る）")

        # 【オシレーター判定 (RSI)】
        if rsi < 30:
            score += 35
            reasons.append(f"RSI({rsi:.1f}) が強い売られすぎ領域。売りのクライマックス（押し目買場）")
        elif rsi < 42:
            score += 15
            reasons.append(f"RSI({rsi:.1f}) 割安ゾーン。エントリー好機")
        elif rsi > 75:
            score -= 30
            reasons.append(f"RSI({rsi:.1f}) が買われすぎ過熱ゾーン。利益確定売りを推奨")
        elif rsi > 65:
            score -= 10
            reasons.append(f"RSI({rsi:.1f}) やや買われすぎ傾向")

        # 【MACD転換シグナル】
        if macd_prev < signal_prev and macd_val > signal_val:
            score += 25
            reasons.append("MACDがゴールデンクロス（買い転換サイン初動）")
        elif macd_prev > signal_prev and macd_val < signal_val:
            score -= 25
            reasons.append("MACDがデッドクロス（売り転換サイン初動）")

        # 【ボリンジャーバンド逆張り・ブレイク判定】
        if latest_price <= lower_band:
            score += 20
            reasons.append("ボリンジャーバンド-2σ到達。売られすぎからの自律反発期待")
        elif latest_price >= upper_band:
            score -= 20
            reasons.append("ボリンジャーバンド+2σ到達。短期的な天井感・調整警戒")

        # 【出来高】
        if vol_ratio >= 1.5:
            if latest_price > df['Close'].iloc[-2]:
                score += 15
                reasons.append(f"大口資金流入（出来高が直近5日平均の{vol_ratio:.1f}倍に急増し上昇）")
            else:
                score -= 15
                reasons.append(f"機関投資家の売り抜け懸念（出来高{vol_ratio:.1f}倍で下落）")

        # ==========================================
        # 🎯 売買予測アクション（売買助言判定）
        # ==========================================
        if score >= 50:
            action = "STRONG BUY"
            action_label = "🔥 強気買い（今すぐ注目）"
            action_color = "#22c55e" # Green
        elif score >= 20:
            action = "BUY"
            action_label = "🟢 買い（押し目狙い）"
            action_color = "#10b981"
        elif score <= -40:
            action = "STRONG SELL"
            action_label = "🚨 強烈な売り / 手仕舞い"
            action_color = "#ef4444" # Red
        elif score <= -15:
            action = "SELL"
            action_label = "🟠 売り推奨 / 利益確定"
            action_color = "#f97316"
        else:
            action = "HOLD"
            action_label = "⚪️ 中立 / 静観・様子見"
            action_color = "#94a3b8"

        return {
            "symbol": symbol,
            "name": name,
            "market": market,
            "currency": currency,
            "price": f"{latest_price:,.2f}" if market != "日本" else f"{int(latest_price):,}",
            "rsi": round(rsi, 1),
            "score": score,
            "action": action,
            "action_label": action_label,
            "action_color": action_color,
            "reasons": reasons
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def main():
    results = []
    for item in GLOBAL_TICKERS:
        res = analyze_stock(item)
        if res:
            results.append(res)
    
    # スコアの高い順にソート
    results.sort(key=lambda x: x['score'], reverse=True)
    
    today_str = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")

    # HTMLの自動生成（より洗練されたダッシュボードデザイン）
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global Stock AI Advisor | 日本株・外国株 売買予測</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151c2c;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border-color: #1e293b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 25px;
        }}
        h1 {{
            margin: 0;
            font-size: 1.8rem;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-sub);
            font-size: 0.9rem;
            margin-top: 8px;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .stock-title {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .market-badge {{
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 4px;
            background: #1e293b;
            color: #38bdf8;
            font-weight: bold;
        }}
        .symbol {{
            font-size: 1.3rem;
            font-weight: bold;
        }}
        .name {{
            color: var(--text-sub);
            font-size: 0.95rem;
        }}
        .action-badge {{
            font-size: 0.95rem;
            font-weight: bold;
            padding: 6px 14px;
            border-radius: 20px;
            color: #fff;
        }}
        .stock-info {{
            display: flex;
            gap: 25px;
            margin: 15px 0 10px 0;
            font-size: 0.9rem;
            color: var(--text-sub);
        }}
        .stock-info strong {{
            color: var(--text-main);
        }}
        .reasons-list {{
            background-color: #0b0f19;
            border-radius: 8px;
            padding: 12px 15px;
            margin-top: 10px;
            font-size: 0.88rem;
        }}
        .reasons-list ul {{
            margin: 5px 0 0 0;
            padding-left: 20px;
            color: #cbd5e1;
        }}
        .reasons-list li {{
            margin-bottom: 4px;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            color: #64748b;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌐 Global AI Stock Trading Advisor</h1>
            <div class="subtitle">日本株・米国株・欧州ADR リアルタイム分析 & 売買予測シグナル | 更新日時 (JST): {today_str}</div>
        </header>
"""

    for item in results:
        reasons_html = "".join([f"<li>{r}</li>" for r in item['reasons']])
        html_content += f"""
        <div class="card">
            <div class="card-header">
                <div class="stock-title">
                    <span class="market-badge">{item['market']}</span>
                    <span class="symbol">{item['symbol']}</span>
                    <span class="name">{item['name']}</span>
                </div>
                <div class="action-badge" style="background-color: {item['action_color']};">
                    {item['action_label']}
                </div>
            </div>
            <div class="stock-info">
                <div>株価: <strong>{item['currency']}{item['price']}</strong></div>
                <div>RSI (14日): <strong>{item['rsi']}</strong></div>
                <div>総合スコア: <strong>{item['score']} pts</strong></div>
            </div>
            <div class="reasons-list">
                📊 <strong>売買判断の根拠とテクニカル分析:</strong>
                <ul>
                    {reasons_html}
                </ul>
            </div>
        </div>
"""

    html_content += """
        <footer>
            ⚠️ 【免責事項】当Webページが提示する売買判断・分析結果は、アルゴリズムによる自動計算に基づく情報提供のみを目的としており、特定の銘柄の売買を勧誘・保証するものではありません。実際の投資判断はご自身の責任で行ってください。
        </footer>
    </div>
</body>
</html>
"""

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
