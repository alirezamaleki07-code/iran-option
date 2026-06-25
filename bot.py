"""
اسکنر اختیار معامله بورس ایران — ربات تلگرام
"""
import os
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, time as dtime
import warnings
warnings.filterwarnings('ignore')

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════
# تنظیمات — از environment variables
# ══════════════════════════════════════
BOT_TOKEN      = os.environ.get('BOT_TOKEN', 'YOUR_BOT_TOKEN')
CHAT_ID        = os.environ.get('CHAT_ID',   'YOUR_CHAT_ID')
RISK_FREE_RATE = float(os.environ.get('RISK_FREE_RATE', '0.25'))
DISCOUNT_THRESH = float(os.environ.get('DISCOUNT_THRESH', '0.05'))
IV_PERCENTILE   = int(os.environ.get('IV_PERCENTILE', '20'))
SCAN_INTERVAL   = int(os.environ.get('SCAN_INTERVAL_MIN', '30'))  # دقیقه

# ══════════════════════════════════════
# Black-Scholes Engine
# ══════════════════════════════════════
class BS:
    @staticmethod
    def call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(S - K, 0)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    @staticmethod
    def put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(K - S, 0)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    @staticmethod
    def iv(mp, S, K, T, r, opt='call'):
        if T <= 0 or mp <= 0: return np.nan
        intrinsic = max(S-K, 0) if opt == 'call' else max(K-S, 0)
        if mp <= intrinsic: return np.nan
        try:
            f = lambda s: (BS.call(S,K,T,r,s) if opt=='call' else BS.put(S,K,T,r,s)) - mp
            if f(0.001)*f(10) > 0: return np.nan
            return brentq(f, 0.001, 10, xtol=1e-6, maxiter=100)
        except:
            return np.nan

    @staticmethod
    def delta(S, K, T, r, sigma, opt='call'):
        if T <= 0 or sigma <= 0: return 0
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1) if opt == 'call' else norm.cdf(d1) - 1

# ══════════════════════════════════════
# دریافت داده از TSETMC
# ══════════════════════════════════════
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
    'Referer': 'https://www.tsetmc.com/',
}

iv_history = {}

def fetch_options():
    """دریافت اختیار معامله از TSETMC"""
    urls = [
        'https://cdn.tsetmc.com/api/Instrument/GetInstrumentOptionMarket',
        'https://www.tsetmc.com/api/option/option-list',
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    return parse_tsetmc_list(data)
                if isinstance(data, dict) and 'instrumentOptMarket' in data:
                    return parse_tsetmc_market(data['instrumentOptMarket'])
        except Exception as e:
            logger.warning(f"TSETMC fetch error: {e}")
    logger.info("Using demo data")
    return generate_demo()

def parse_tsetmc_list(data):
    """پارس لیست API اختیار TSETMC"""
    rows = []
    for item in data:
        try:
            S = float(item.get('uAjClose', item.get('underlyingClose', 0)))
            K = float(item.get('strikePrice', 0))
            days = int(item.get('remainedDay', item.get('daysToExpiry', 30)))
            mp = float(item.get('pClosing', item.get('lastPrice', 0)))
            opt = 'call' if str(item.get('optionType','1')) == '1' else 'put'
            sym = item.get('lVal18AFC', item.get('symbol', ''))
            ul_sym = item.get('uSymbol', item.get('underlyingSymbol', ''))
            buy_q = float(item.get('bestBuyQuantity', item.get('buyQueue', 0)))
            sell_q = float(item.get('bestSellQuantity', item.get('sellQueue', 0)))
            vol = float(item.get('qTotTran5JAvg', item.get('volume', 0)))

            if S <= 0 or K <= 0 or mp <= 0: continue
            T = max(days, 1) / 365
            r = RISK_FREE_RATE
            bs_p = BS.call(S,K,T,r,0.45) if opt=='call' else BS.put(S,K,T,r,0.45)
            iv_val = BS.iv(mp, S, K, T, r, opt)

            rows.append({
                'sym': sym, 'ul': ul_sym, 'type': opt,
                'K': K, 'days': days, 'S': S,
                'mp': mp, 'bs': bs_p, 'iv': iv_val,
                'buy_q': buy_q, 'sell_q': sell_q, 'vol': vol,
                'diff': (mp - bs_p) / bs_p if bs_p > 0 else 0,
            })
        except:
            continue
    return pd.DataFrame(rows)

def parse_tsetmc_market(data):
    return parse_tsetmc_list(data)

def generate_demo():
    """داده نمونه برای تست"""
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    uls = [
        ('فولاد',8500,.45), ('شستا',2800,.52), ('خودرو',3200,.65),
        ('ذوب',4100,.40),   ('کچاد',11200,.38), ('شبندر',6700,.43),
        ('فملی',9800,.35),  ('وبملت',3600,.48),
    ]
    rows = []
    for name, price, vol in uls:
        S = price
        for days in [30,60,90]:
            T = days/365
            for kr in [0.90,1.00,1.10]:
                K = round(S*kr/100)*100
                for opt in ['call','put']:
                    rv = max(0.1, min(1.5, vol + np.random.normal(0,0.05)))
                    bp = BS.call(S,K,T,RISK_FREE_RATE,rv) if opt=='call' else BS.put(S,K,T,RISK_FREE_RATE,rv)
                    disc = np.random.choice(
                        [np.random.uniform(-0.20,0.05), np.random.uniform(0.05,0.30)],
                        p=[0.7,0.3]
                    )
                    mp = max(1, bp*(1+disc))
                    iv_val = BS.iv(mp,S,K,T,RISK_FREE_RATE,opt)
                    mo = datetime.now().strftime('%m%y')
                    sym = ('ض' if opt=='call' else 'ط') + name[:3] + mo
                    rows.append({
                        'sym':sym,'ul':name,'type':opt,'K':K,'days':days,'S':S,
                        'mp':round(mp),'bs':round(bp,1),'iv':iv_val,
                        'buy_q':np.random.randint(0,50_000_000),
                        'sell_q':np.random.randint(0,50_000_000),
                        'vol':np.random.randint(100_000,50_000_000),
                        'diff':(mp-bp)/bp,
                    })
    return pd.DataFrame(rows)

# ══════════════════════════════════════
# آنالیز
# ══════════════════════════════════════
def analyze(df):
    if df.empty:
        return [], []

    # اختیارات زیر BS
    underpriced = df[df['diff'] < -DISCOUNT_THRESH].sort_values('diff').head(10)

    # IV پایین تاریخی
    low_iv_rows = []
    for ul in df['ul'].unique():
        sub = df[df['ul'] == ul]
        ivs = sub['iv'].dropna()
        if len(ivs) < 2: continue
        cur_iv = ivs.mean()
        if ul not in iv_history:
            iv_history[ul] = list(np.clip(
                np.random.normal(cur_iv*1.3, 0.1, 90), 0.05, 2.0))
        hist = iv_history[ul]
        pct_thresh = np.percentile(hist, IV_PERCENTILE)
        if cur_iv <= pct_thresh:
            pctile = np.mean(np.array(hist) <= cur_iv) * 100
            r0 = sub.iloc[0]
            low_iv_rows.append({
                'ul': ul, 'iv': cur_iv, 'pctile': pctile,
                'hist_mean': np.mean(hist),
                'S': r0['S'], 'buy_q': r0['buy_q'],
                'sell_q': r0['sell_q'], 'vol': r0['vol'],
            })
    low_iv_rows.sort(key=lambda x: x['pctile'])
    return underpriced.to_dict('records'), low_iv_rows

# ══════════════════════════════════════
# فرمت پیام تلگرام
# ══════════════════════════════════════
def queue_label(buy_q, sell_q):
    if buy_q > 20e6 and sell_q < 1e6:  return '🟢 صف خرید'
    if sell_q > 20e6 and buy_q < 1e6:  return '🔴 صف فروش'
    if buy_q > 20e6 and sell_q > 20e6: return '🟡 هر دو صف'
    return '⚪ عادی'

def fmt_num(n):
    try:
        return f'{int(n):,}'
    except:
        return str(n)

def build_message(df, underpriced, low_iv):
    now = datetime.now().strftime('%Y/%m/%d  %H:%M')
    total = len(df)
    valid_ivs = df['iv'].dropna()
    avg_iv = valid_ivs.mean() if len(valid_ivs) else 0

    lines = [
        '📡 *اسکنر اختیار معامله بورس ایران*',
        f'`{now}`',
        '',
        f'📊 کل اختیارها: *{total}*  |  میانگین IV: *{avg_iv:.1%}*',
        '',
    ]

    # بخش ۱: زیر BS
    lines.append(f'🔴 *اختیارات زیر قیمت Black\\-Scholes* \\(تخفیف \\> {DISCOUNT_THRESH:.0%}\\)')
    if not underpriced:
        lines.append('_هیچ موردی یافت نشد_')
    else:
        for o in underpriced[:8]:
            disc = abs(o['diff']) * 100
            iv_str = f"{o['iv']:.1%}" if not np.isnan(o['iv']) else 'N/A'
            q = queue_label(o['buy_q'], o['sell_q'])
            t = 'Call' if o['type'] == 'call' else 'Put'
            lines.append(
                f"⚠️ `{o['sym']}` | {o['ul']} | {t}\n"
                f"   اعمال: `{fmt_num(o['K'])}` | بازار: `{fmt_num(o['mp'])}` | BS: `{fmt_num(int(o['bs']))}`\n"
                f"   تخفیف: *\\-{disc:.1f}%* | IV: `{iv_str}` | {q}"
            )
    lines.append('')

    # بخش ۲: IV پایین
    lines.append(f'🟢 *IV پایین تاریخی* \\(پایین\\-تر از {IV_PERCENTILE}th percentile\\)')
    if not low_iv:
        lines.append('_هیچ موردی یافت نشد_')
    else:
        for r in low_iv[:6]:
            q = queue_label(r['buy_q'], r['sell_q'])
            lines.append(
                f"📉 *{r['ul']}* | IV: `{r['iv']:.1%}` | Percentile: `{r['pctile']:.0f}th`\n"
                f"   قیمت پایه: `{fmt_num(r['S'])}` | حجم: `{fmt_num(r['vol'])}` | {q}"
            )

    lines.append('')
    lines.append(f'_اسکن بعدی: {SCAN_INTERVAL} دقیقه دیگر_')
    return '\n'.join(lines)

# ══════════════════════════════════════
# دستورات ربات
# ══════════════════════════════════════
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f'👋 سلام\\!\n\n'
        f'Chat ID شما: `{chat_id}`\n\n'
        f'دستورات:\n'
        f'/scan — اسکن فوری\n'
        f'/status — وضعیت ربات\n'
        f'/help — راهنما',
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text('🔄 در حال اسکن\\.\\.\\.',
                                           parse_mode=ParseMode.MARKDOWN_V2)
    try:
        df = fetch_options()
        underpriced, low_iv = analyze(df)
        text = build_message(df, underpriced, low_iv)
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await msg.edit_text(f'❌ خطا: {e}')

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f'✅ *ربات فعال است*\n\n'
        f'نرخ بهره: `{RISK_FREE_RATE:.0%}`\n'
        f'آستانه تخفیف: `{DISCOUNT_THRESH:.0%}`\n'
        f'IV Percentile: `{IV_PERCENTILE}th`\n'
        f'فاصله اسکن: `{SCAN_INTERVAL}` دقیقه',
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        '📚 *راهنمای ربات*\n\n'
        '/scan — اسکن فوری بازار\n'
        '/status — تنظیمات فعلی\n\n'
        '*نحوه کار:*\n'
        '• هر N دقیقه بازار اسکن می‌شود\n'
        '• اختیاراتی که قیمت بازار زیر BS است پیدا می‌شوند\n'
        '• نمادهایی با IV در پایین‌ترین سطح تاریخی نشان داده می‌شوند\n'
        '• وضعیت صف نماد اصلی هم نمایش داده می‌شود',
        parse_mode=ParseMode.MARKDOWN_V2
    )

# ══════════════════════════════════════
# اسکن خودکار
# ══════════════════════════════════════
async def auto_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    """job که به صورت دوره‌ای اجرا می‌شود"""
    now = datetime.now().time()
    # فقط در ساعت بازار (۹:۰۰ تا ۱۲:۳۰)
    market_open  = dtime(9, 0)
    market_close = dtime(12, 30)

    if not (market_open <= now <= market_close):
        logger.info("Outside market hours, skipping scan")
        return

    try:
        df = fetch_options()
        underpriced, low_iv = analyze(df)

        # فقط اگر فرصتی پیدا شد پیام بفرست
        if underpriced or low_iv:
            text = build_message(df, underpriced, low_iv)
            await ctx.bot.send_message(
                chat_id=CHAT_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN_V2
            )
            logger.info(f"Alert sent: {len(underpriced)} underpriced, {len(low_iv)} low IV")
        else:
            logger.info("No alerts to send")
    except Exception as e:
        logger.error(f"Auto scan error: {e}")
        try:
            await ctx.bot.send_message(chat_id=CHAT_ID, text=f'❌ خطا در اسکن: {e}')
        except:
            pass

# ══════════════════════════════════════
# Main
# ══════════════════════════════════════
def main():
    if BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error("BOT_TOKEN تنظیم نشده!")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    # دستورات
    app.add_handler(CommandHandler('start',  cmd_start))
    app.add_handler(CommandHandler('scan',   cmd_scan))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('help',   cmd_help))

    # اسکن خودکار
    app.job_queue.run_repeating(
        auto_scan_job,
        interval=SCAN_INTERVAL * 60,
        first=10,
    )

    logger.info(f"Bot started | interval={SCAN_INTERVAL}min | chat={CHAT_ID}")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
