import os, json, time, asyncio, logging, threading
from datetime import datetime
from flask import Flask, request
import requests, yfinance as yf, pandas as pd, numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ========== إعدادات ==========
TELEGRAM_TOKEN = "8888596362:AAF4_V3rWfV5hd6I81kUMrFuaC2n-DaZrAI"
CHAT_ID = 6230253013
SYMBOL = "EURUSD=X"
TIMEFRAME = "5m"
ANALYSIS_INTERVAL = 300
LOT_SIZE = 1.0
TRADE_LOG_FILE = "/tmp/trade_log.json"  # Render يسمح بالكتابة في /tmp فقط
MODEL_FILE = "/tmp/model_weights.json"

# ---------- باقي الدوال (fetch_data, calculate_indicators, detect_whale_trap, liquidity_zones, machine_learning_filter, generate_recommendation, load_trades, save_trades) ----------
def fetch_data():
    try:
        data = yf.download(SYMBOL, period="5d", interval=TIMEFRAME, progress=False)
        if data.empty: return None
        return data
    except Exception as e:
        logging.error(f"خطأ في جلب البيانات: {e}")
        return None

def calculate_indicators(df):
    df = df.copy()
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband(); df['bb_lower'] = bb.bollinger_lband(); df['bb_mid'] = bb.bollinger_mavg()
    df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['vwap'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()
    return df

def detect_whale_trap(df):
    if len(df) < 3: return False, "none"
    latest, prev = df.iloc[-1], df.iloc[-2]
    if (latest['Close'] >= latest['Open'] and prev['Close'] < prev['Open']):
        if (latest['Open'] <= prev['Close'] and latest['Close'] >= prev['Open']):
            if prev['Close'] <= df['bb_upper'].iloc[-2] and latest['Close'] > df['bb_upper'].iloc[-1]:
                if latest['Volume'] > prev['Volume'] * 1.5: return True, "bullish_trap"
    if (latest['Close'] < latest['Open'] and prev['Close'] > prev['Open']):
        if (latest['Open'] >= prev['Close'] and latest['Close'] <= prev['Open']):
            if prev['Close'] >= df['bb_lower'].iloc[-2] and latest['Close'] < df['bb_lower'].iloc[-1]:
                if latest['Volume'] > prev['Volume'] * 1.5: return True, "bearish_trap"
    return False, "none"

def liquidity_zones(df):
    levels = []
    for i in range(2, len(df)-2):
        if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1]:
            levels.append(('resistance', df['High'].iloc[i]))
        if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1]:
            levels.append(('support', df['Low'].iloc[i]))
    if not levels: return None, None
    current_price = df['Close'].iloc[-1]
    supports = [l for t, l in levels if t == 'support' and l < current_price]
    resistances = [l for t, l in levels if t == 'resistance' and l > current_price]
    return (max(supports) if supports else None, min(resistances) if resistances else None)

def machine_learning_filter(features):
    try:
        with open(MODEL_FILE, 'r') as f: weights = json.load(f)
    except FileNotFoundError:
        weights = {"rsi_weight": -0.5, "bb_weight": 1.0, "vol_weight": 0.3, "bias": 0.0}
    score = (features['rsi'] * weights.get('rsi_weight', 0) +
             features['bb_position'] * weights.get('bb_weight', 0) +
             features['volume_ratio'] * weights.get('vol_weight', 0) + weights.get('bias', 0))
    return score > 0.5

def generate_recommendation():
    df = fetch_data()
    if df is None: return None
    df = calculate_indicators(df)
    last = df.iloc[-1]
    bb_lower, bb_upper, rsi, price = last['bb_lower'], last['bb_upper'], last['rsi'], last['Close']
    signal = None
    if price <= bb_lower and rsi < 40:
        signal = "BUY"; entry = price; sl = entry - (entry * 0.0015); tp = entry + (entry * 0.003)
    elif price >= bb_upper and rsi > 60:
        signal = "SELL"; entry = price; sl = entry + (entry * 0.0015); tp = entry - (entry * 0.003)
    if signal is None: return None
    trap, trap_type = detect_whale_trap(df)
    if trap and ((signal=="BUY" and trap_type=="bullish_trap") or (signal=="SELL" and trap_type=="bearish_trap")): return None
    sup, res = liquidity_zones(df)
    if signal=="BUY" and sup: sl = max(sl, sup * 0.998)
    elif signal=="SELL" and res: sl = min(sl, res * 1.002)
    features = {
        'rsi': rsi,
        'bb_position': (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5,
        'volume_ratio': last['Volume'] / df['Volume'].rolling(20).mean().iloc[-1] if df['Volume'].rolling(20).mean().iloc[-1] != 0 else 1.0
    }
    if not machine_learning_filter(features): return None
    trades = load_trades()
    win_rate = 65
    if len(trades) >= 5:
        recent = trades[-20:]
        wins = sum(1 for t in recent if t['profit'] > 0)
        win_rate = int((wins / len(recent)) * 100)
    return {
        'signal': signal, 'entry': round(entry, 5), 'sl': round(sl, 5),
        'tp': round(tp, 5), 'win_rate': win_rate, 'volume': LOT_SIZE,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def load_trades():
    try:
        with open(TRADE_LOG_FILE, 'r') as f: return json.load(f)
    except FileNotFoundError: return []

def save_trades(trades):
    with open(TRADE_LOG_FILE, 'w') as f: json.dump(trades, f, indent=2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("جاهز لإرسال التوصيات 🚀\nاستخدم /report لعرض الأداء.")

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = load_trades()
    if not trades:
        await update.message.reply_text("لا توجد صفقات مكتملة بعد."); return
    total = len(trades); wins = sum(1 for t in trades if t['profit'] > 0); total_profit = sum(t['profit'] for t in trades)
    msg = f"📋 الأداء:\nالصفقات: {total}\nالناجحة: {wins}\nالخاسرة: {total-wins}\nصافي الربح: {total_profit:.2f} نقطة"
    await update.message.reply_text(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text.startswith("ربح") or text.startswith("كسب"):
        try: profit = float(text.split()[-1])
        except: await update.message.reply_text("استخدم: ربح 30.5"); return
        trades = load_trades()
        if trades and 'entry' in trades[-1] and 'exit' not in trades[-1]:
            trades[-1]['exit'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); trades[-1]['profit'] = profit
            save_trades(trades); await update.message.reply_text(f"تم تسجيل ربح {profit} نقطة. شكراً!"); update_model()
        else: await update.message.reply_text("لا توجد صفقة مفتوحة حالياً.")
    elif text.startswith("خسارة") or text.startswith("خسر"):
        try: loss = abs(float(text.split()[-1]))
        except: await update.message.reply_text("استخدم: خسارة 15"); return
        trades = load_trades()
        if trades and 'entry' in trades[-1] and 'exit' not in trades[-1]:
            trades[-1]['exit'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); trades[-1]['profit'] = -loss
            save_trades(trades); await update.message.reply_text(f"تم تسجيل خسارة {loss} نقطة."); update_model()
        else: await update.message.reply_text("لا توجد صفقة مفتوحة.")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    if data == "ignore":
        await query.edit_message_text(text=query.message.text + "\n\n⛔ تم التجاهل."); return
    if data.startswith("exec_"):
        parts = data.split("_"); signal = parts[1]; entry = float(parts[2])
        trades = load_trades()
        new_trade = {'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'signal': signal, 'entry': entry, 'volume': LOT_SIZE, 'status': 'open'}
        trades.append(new_trade); save_trades(trades)
        await query.edit_message_text(text=query.message.text + "\n\n✅ تم تسجيل الدخول. أبلغني بالنتيجة لاحقاً (ربح/خسارة + المبلغ)")

async def send_recommendation(context: ContextTypes.DEFAULT_TYPE):
    rec = generate_recommendation()
    if rec is None: return
    keyboard = [[InlineKeyboardButton("✅ نفذت الصفقة", callback_data=f"exec_{rec['signal']}_{rec['entry']}"),
                 InlineKeyboardButton("❌ تجاهل", callback_data="ignore")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message = (f"📊 *توصية تداول*\nالزوج: EURUSD\nالنوع: {'شراء' if rec['signal']=='BUY' else 'بيع'}\n"
               f"سعر الدخول: {rec['entry']}\nوقف الخسارة: {rec['sl']}\nهدف الربح: {rec['tp']}\n"
               f"الحجم: {rec['volume']} لوت\nنسبة النجاح المتوقعة: {rec['win_rate']}%\nالوقت: {rec['timestamp']}")
    await context.bot.send_message(chat_id=CHAT_ID, text=message, reply_markup=reply_markup, parse_mode='Markdown')

def update_model():
    trades = load_trades()
    if len(trades) < 5: return
    recent = trades[-20:]
    wins = sum(1 for t in recent if t['profit'] > 0); win_rate = wins / len(recent)
    with open(MODEL_FILE, 'w') as f:
        json.dump({"rsi_weight": -0.5 if win_rate < 0.5 else -0.3,
                   "bb_weight": 1.0 if win_rate < 0.6 else 1.5,
                   "vol_weight": 0.3, "bias": 0.1 if win_rate > 0.6 else -0.1}, f)

# ---------- إعداد البوت ----------
application = Application.builder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("report", report))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
application.add_handler(CallbackQueryHandler(button_handler))
application.job_queue.run_repeating(send_recommendation, interval=ANALYSIS_INTERVAL, first=5)

# ---------- مسارات Flask ----------
@app.route('/')
def home():
    return "AlphendAI Bot is running."

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    if data:
        asyncio.run(application.process_update(Update.de_json(data, application.bot)))
    return "OK"

# ---------- منع النوم ----------
def keep_alive():
    while True:
        time.sleep(240)
        try: requests.get("https://your-app.onrender.com/")  # سنغيره لاحقاً
        except: pass

threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
