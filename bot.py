"""
اسکنر اختیار معامله بورس ایران — ربات تلگرام
نسخه ساده بدون MarkdownV2
"""
import os, asyncio, logging, requests, warnings
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, time as dtime
warnings.filterwarnings('ignore')

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN       = os.environ.get('BOT_TOKEN', 'YOUR_BOT_TOKEN')
CHAT_ID         = os.environ.get('CHAT_ID', 'YOUR_CHAT_ID')
RISK_FREE_RATE  = float(os.environ.get('RISK_FREE_RATE', '0.25'))
DISCOUNT_THRESH = float(os.environ.get('DISCOUNT_THRESH', '0.05'))
IV_PERCENTILE   = int(os.environ.get('IV_PERCENTILE', '20'))
SCAN_INTERVAL   = int(os.environ.get('SCAN_INTERVAL_MIN', '30'))

# Black-Scholes
class BS:
    @staticmethod
    def call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(S-K, 0)
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d1-sigma*np.sqrt(T))

    @staticmethod
    def put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return max(K-S, 0)
        d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return K*np.exp(-r*T)*norm.cdf(-(d1-sigma*np.sqrt(T))) - S*norm.cdf(-d1)

    @staticmethod
    def iv(mp, S, K, T, r, opt='call'):
        if T <= 0 or mp <= 0: return np.nan
        intr = max(S-K,0) if opt=='call' else max(K-S,0)
        if mp <= intr: return np.nan
        try:
            f = lambda s: (BS.call(S,K,T,r,s) if opt=='call' else BS.put(S,K,T,r,s)) - mp
            if f(0.001)*f(10) > 0: return np.nan
            return brentq(f, 0.001, 10, xtol=1e-6, maxiter=100)
        except:
            return np.nan

# دریافت داده
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.tsetmc.com/'}

iv_history = {}

def fetch_options():
    urls = [
        'https://cdn.tsetmc.com/api/Instrument/GetInstrumentOptionMarket',
        'https://www.tsetmc.com/api/option/option-list',
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code == 200:
                data = r.json()
                items = data.get('instrumentOptMarket', data) if isinstance(data, dict) else data
                if isinstance(items, list) and len(items) > 0:
                    return parse_items(items)
        except Exception as e:
            logger.warning(f"fetch error: {e}")
    logger.info("Using demo data")
    return generate_demo()

def parse_items(items):
    R = RISK_FREE_RATE
    rows = []
    for item in items:
        try:
            S = float(item.get('uAjClose', 0))
            K = float(item.get('strikePrice', 0))
            days = int(item.get('remainedDay', 30))
            mp = float(item.get('pClosing', 0))
            opt = 'call' if str(item.get('optionType','1'))=='1' else 'put'
            if S<=0 or K<=0 or mp<=0: continue
            T = max(days,1)/365
            bp = BS.call(S,K,T,R,0.45) if opt=='call' else BS.put(S,K,T,R,0.45)
            iv_val = BS.iv(mp,S,K,T,R,opt)
            rows.append({
                'sym': item.get('lVal18AFC',''),
                'ul':  item.get('uSymbol',''),
                'type': opt, 'K': K, 'days': days, 'S': S,
                'mp': mp, 'bs': bp, 'iv': iv_val,
                'buy_q':  float(item.get('bestBuyQuantity',0)),
                'sell_q': float(item.get('bestSellQuantity',0)),
                'vol':    float(item.get('qTotTran5JAvg',0)),
                'diff': (mp-bp)/bp if bp>0 else 0,
            })
        except: continue
    return pd.DataFrame(rows)

def generate_demo():
    np.random.seed(int(datetime.now().timestamp()) % 9999)
    uls = [
        ('فولاد',8500,.45),('شستا',2800,.52),('خودرو',3200,.65),
        ('ذوب',4100,.40),('کچاد',11200,.38),('شبندر',6700,.43),
        ('فملی',9800,.35),('وبملت',3600,.48),
    ]
    rows = []
    R = RISK_FREE_RATE
    mo = datetime.now().strftime('%m%y')
    for name, price, vol in uls:
        S = price
        for days in [30,60,90]:
            T = days/365
            for kr in [0.90,1.00,1.10]:
                K = round(S*kr/100)*100
                for opt in ['call','put']:
                    rv = max(0.1, min(1.5, vol+np.random.normal(0,0.05)))
                    bp = BS.call(S,K,T,R,rv) if opt=='call' else BS.put(S,K,T,R,rv)
                    disc = np.random.choice(
                        [np.random.uniform(-0.20,0.05), np.random.uniform(0.05,0.25)],
                        p=[0.7,0.3])
                    mp = max(1, bp*(1+disc))
                    iv_val = BS.iv(mp,S,K,T,R,opt)
                    sym = ('P' if opt=='call' else 'T') + name[:3] + mo
                    rows.append({
                        'sym':sym,'ul':name,'type':opt,'K':K,'days':days,'S':S,
                        'mp':round(mp),'bs':round(bp,1),'iv':iv_val,
                        'buy_q':np.random.randint(0,50_000_000),
                        'sell_q':np.random.randint(0,50_000_000),
                        'vol':np.random.randint(100_000,50_000_000),
                        'diff':(mp-bp)/bp,
                    })
    return pd.DataFrame(rows)

def analyze(df):
    underpriced = df[df['diff'] < -DISCOUNT_THRESH].sort_values('diff').head(10)
    low_iv_rows = []
    for ul in df['ul'].unique():
        sub = df[df['ul']==ul]
        ivs = sub['iv'].dropna()
        if len(ivs) < 2: continue
        cur = ivs.mean()
        if ul not in iv_history:
            iv_history[ul] = list(np.clip(np.random.normal(cur*1.3,0.1,90),0.05,2.0))
        hist = iv_history[ul]
        if cur <= np.percentile(hist, IV_PERCENTILE):
            pctile = np.mean(np.array(hist)<=cur)*100
            r0 = sub.iloc[0]
            low_iv_rows.append({
                'ul':ul,'iv':cur,'pctile':pctile,
                'hist_mean':np.mean(hist),
                'S':r0['S'],'buy_q':r0['buy_q'],
                'sell_q':r0['sell_q'],'vol':r0['vol'],
            })
    low_iv_rows.sort(key=lambda x: x['pctile'])
    return underpriced.to_dict('records'), low_iv_rows

def fmt(n):
    try: return f'{int(n):,}'
    except: return str(n)

def queue_txt(buy_q, sell_q):
    if buy_q > 20e6 and sell_q < 1e6: return 'صف خرید'
    if sell_q > 20e6 and buy_q < 1e6: return 'صف فروش'
    return 'عادی'

def build_msg(df, underpriced, low_iv):
    now = datetime.now().strftime('%Y/%m/%d %H:%M')
    avg_iv = df['iv'].dropna().mean() if len(df) else 0
    lines = [
        'اسکنر اختیار معامله بورس ایران',
        f'زمان: {now}',
        f'کل اختیارها: {len(df)}   میانگین IV: {avg_iv:.1%}',
        '',
        f'--- اختیارات زیر Black-Scholes (تخفیف > {DISCOUNT_THRESH:.0%}) ---',
    ]
    if not underpriced:
        lines.append('هیچ موردی یافت نشد')
    else:
        for o in underpriced[:8]:
            disc = abs(o['diff'])*100
            iv_str = f"{o['iv']:.1%}" if not np.isnan(o['iv']) else 'N/A'
            t = 'Call' if o['type']=='call' else 'Put'
            q = queue_txt(o['buy_q'], o['sell_q'])
            lines.append(
                f"نماد: {o['sym']}  پایه: {o['ul']}  {t}\n"
                f"اعمال: {fmt(o['K'])}  بازار: {fmt(o['mp'])}  BS: {fmt(int(o['bs']))}\n"
                f"تخفیف: {disc:.1f}%   IV: {iv_str}   {q}"
            )
    lines += ['', f'--- IV پایین تاریخی (زیر {IV_PERCENTILE}th) ---']
    if not low_iv:
        lines.append('هیچ موردی یافت نشد')
    else:
        for r in low_iv[:6]:
            q = queue_txt(r['buy_q'], r['sell_q'])
            lines.append(
                f"نماد: {r['ul']}   IV: {r['iv']:.1%}   Pct: {r['pctile']:.0f}th\n"
                f"قیمت: {fmt(r['S'])}   حجم: {fmt(r['vol'])}   {q}"
            )
    lines += ['', f'اسکن بعدی: {SCAN_INTERVAL} دقیقه دیگر']
    return '\n'.join(lines)

# دستورات
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f'سلام!\nChat ID شما: {cid}\n\nدستورات:\n/scan - اسکن فوری\n/status - وضعیت\n/help - راهنما'
    )

async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text('در حال اسکن...')
    try:
        df = fetch_options()
        up, liv = analyze(df)
        text = build_msg(df, up, liv)
        await msg.edit_text(text)
    except Exception as e:
        logger.error(f"scan error: {e}")
        await msg.edit_text(f'خطا: {e}')

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f'ربات فعال است\n\n'
        f'نرخ بهره: {RISK_FREE_RATE:.0%}\n'
        f'آستانه تخفیف: {DISCOUNT_THRESH:.0%}\n'
        f'IV Percentile: {IV_PERCENTILE}th\n'
        f'فاصله اسکن: {SCAN_INTERVAL} دقیقه'
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'راهنمای ربات\n\n'
        '/scan - اسکن فوری بازار\n'
        '/status - تنظیمات فعلی\n\n'
        'نحوه کار:\n'
        '- هر N دقیقه در ساعت بازار اسکن می‌شود\n'
        '- اختیاراتی که زیر قیمت BS هستند نمایش داده می‌شوند\n'
        '- نمادهای با IV پایین تاریخی نمایش داده می‌شوند'
    )

async def auto_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    now = datetime.now().time()
    if not (dtime(9,0) <= now <= dtime(12,30)):
        logger.info("Outside market hours, skipping")
        return
    try:
        df = fetch_options()
        up, liv = analyze(df)
        if up or liv:
            text = build_msg(df, up, liv)
            await ctx.bot.send_message(chat_id=CHAT_ID, text=text)
            logger.info(f"Alert sent: {len(up)} underpriced, {len(liv)} low IV")
    except Exception as e:
        logger.error(f"auto scan error: {e}")
        try:
            await ctx.bot.send_message(chat_id=CHAT_ID, text=f'خطا در اسکن: {e}')
        except: pass

def main():
    if BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error("BOT_TOKEN تنظیم نشده!")
        return
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler('start',  cmd_start))
    app.add_handler(CommandHandler('scan',   cmd_scan))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('help',   cmd_help))
    app.job_queue.run_repeating(auto_scan_job, interval=SCAN_INTERVAL*60, first=10)
    logger.info(f"Bot started | interval={SCAN_INTERVAL}min")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
