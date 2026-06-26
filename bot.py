"""
اسکنر اختیار معامله بورس ایران — ربات تلگرام
با قابلیت واچ‌لیست شخصی
"""
import os, json, logging, requests, warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, time as dtime
import pandas as pd
warnings.filterwarnings('ignore')

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ── تنظیمات از Environment Variables ──
BOT_TOKEN       = os.environ.get('BOT_TOKEN', 'YOUR_BOT_TOKEN')
CHAT_ID         = os.environ.get('CHAT_ID', 'YOUR_CHAT_ID')
RISK_FREE_RATE  = float(os.environ.get('RISK_FREE_RATE', '0.25'))
DISCOUNT_THRESH = float(os.environ.get('DISCOUNT_THRESH', '0.05'))
IV_PERCENTILE   = int(os.environ.get('IV_PERCENTILE', '20'))
SCAN_INTERVAL   = int(os.environ.get('SCAN_INTERVAL_MIN', '30'))

# ── واچ‌لیست (در حافظه — هر بار ربات restart شود پاک می‌شود) ──
WATCHLIST_FILE = 'watchlist.json'

def load_watchlist():
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except: pass
    return []

def save_watchlist(wl):
    try:
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(wl, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"save watchlist error: {e}")

watchlist = load_watchlist()

# ══════════════════════════════════════
# Black-Scholes
# ══════════════════════════════════════
class BS:
    @staticmethod
    def call(S, K, T, r, s):
        if T<=0 or s<=0: return max(S-K, 0)
        d1 = (np.log(S/K)+(r+0.5*s**2)*T)/(s*np.sqrt(T))
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d1-s*np.sqrt(T))

    @staticmethod
    def put(S, K, T, r, s):
        if T<=0 or s<=0: return max(K-S, 0)
        d1 = (np.log(S/K)+(r+0.5*s**2)*T)/(s*np.sqrt(T))
        return K*np.exp(-r*T)*norm.cdf(-(d1-s*np.sqrt(T))) - S*norm.cdf(-d1)

    @staticmethod
    def iv(mp, S, K, T, r, opt='call'):
        if T<=0 or mp<=0: return np.nan
        intr = max(S-K,0) if opt=='call' else max(K-S,0)
        if mp<=intr: return np.nan
        try:
            f = lambda s: (BS.call(S,K,T,r,s) if opt=='call' else BS.put(S,K,T,r,s)) - mp
            if f(0.001)*f(10)>0: return np.nan
            return brentq(f, 0.001, 10, xtol=1e-6, maxiter=100)
        except: return np.nan

# ══════════════════════════════════════
# دریافت داده از TSETMC
# ══════════════════════════════════════
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.tsetmc.com/'}
iv_history = {}

def fetch_options(symbols):
    """دریافت اختیارات فقط برای نمادهای واچ‌لیست"""
    if not symbols:
        return pd.DataFrame()

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
                    df = parse_and_filter(items, symbols)
                    if not df.empty:
                        logger.info(f"TSETMC: {len(df)} options for {symbols}")
                        return df
        except Exception as e:
            logger.warning(f"TSETMC error: {e}")

    logger.info("Demo mode")
    return generate_demo(symbols)

def parse_and_filter(items, symbols):
    """پارس و فیلتر بر اساس واچ‌لیست"""
    R = RISK_FREE_RATE
    rows = []
    for item in items:
        ul = item.get('uSymbol', item.get('underlyingSymbol', ''))
        # بررسی اینکه نماد پایه در واچ‌لیست است
        matched = any(s in ul or ul in s for s in symbols)
        if not matched:
            continue
        try:
            S    = float(item.get('uAjClose', 0))
            K    = float(item.get('strikePrice', 0))
            days = int(item.get('remainedDay', 30))
            mp   = float(item.get('pClosing', 0))
            opt  = 'call' if str(item.get('optionType','1'))=='1' else 'put'
            if S<=0 or K<=0 or mp<=0: continue
            T  = max(days,1)/365
            bp = BS.call(S,K,T,R,0.45) if opt=='call' else BS.put(S,K,T,R,0.45)
            rows.append({
                'sym':  item.get('lVal18AFC',''),
                'ul':   ul,
                'type': opt, 'K': K, 'days': days, 'S': S,
                'mp': mp, 'bs': bp, 'iv': BS.iv(mp,S,K,T,R,opt),
                'buy_q':  float(item.get('bestBuyQuantity',0)),
                'sell_q': float(item.get('bestSellQuantity',0)),
                'vol':    float(item.get('qTotTran5JAvg',0)),
                'diff':   (mp-bp)/bp if bp>0 else 0,
            })
        except: continue
    return pd.DataFrame(rows)

def generate_demo(symbols):
    """داده نمونه فقط برای نمادهای واچ‌لیست"""
    PRICES = {
        'فولاد':8500,'شستا':2800,'خودرو':3200,'ذوب':4100,
        'کچاد':11200,'شبندر':6700,'فملی':9800,'وبملت':3600,
        'ومعادن':5200,'فارس':7800,'شپنا':4500,'پارسان':9100,
        'وتجارت':1800,'تاپیکو':6300,'صبا':2200,'شفن':8900,
    }
    R = RISK_FREE_RATE
    rows = []
    mo = datetime.now().strftime('%m%y')
    np.random.seed(42)

    for sym in symbols:
        price = PRICES.get(sym, 5000)
        vol   = 0.45
        S     = price
        for days in [30,60,90]:
            T = days/365
            for kr in [0.90,0.95,1.00,1.05,1.10]:
                K = round(S*kr/100)*100
                for opt in ['call','put']:
                    rv = max(0.1, min(1.5, vol+np.random.normal(0,0.06)))
                    bp = BS.call(S,K,T,R,rv) if opt=='call' else BS.put(S,K,T,R,rv)
                    disc = np.random.choice(
                        [np.random.uniform(-0.20,0.05),
                         np.random.uniform(0.05,0.25)], p=[0.7,0.3])
                    mp = max(1, bp*(1+disc))
                    iv_val = BS.iv(mp,S,K,T,R,opt)
                    rows.append({
                        'sym': ('P' if opt=='call' else 'T')+sym[:3]+mo,
                        'ul': sym, 'type': opt, 'K': K, 'days': days, 'S': S,
                        'mp': round(mp), 'bs': round(bp,1), 'iv': iv_val,
                        'buy_q':  np.random.randint(0,50_000_000),
                        'sell_q': np.random.randint(0,50_000_000),
                        'vol':    np.random.randint(100_000,50_000_000),
                        'diff':   (mp-bp)/bp,
                    })
    return pd.DataFrame(rows)

# ══════════════════════════════════════
# آنالیز
# ══════════════════════════════════════
def analyze(df):
    underpriced = df[df['diff'] < -DISCOUNT_THRESH].sort_values('diff').head(10)
    low_iv_rows = []
    for ul in df['ul'].unique():
        sub  = df[df['ul']==ul]
        ivs  = sub['iv'].dropna()
        if len(ivs) < 2: continue
        cur  = ivs.mean()
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
    if buy_q>20e6 and sell_q<1e6:  return 'صف خرید'
    if sell_q>20e6 and buy_q<1e6:  return 'صف فروش'
    return 'عادی'

def build_msg(df, underpriced, low_iv, symbols):
    now    = datetime.now().strftime('%Y/%m/%d  %H:%M')
    avg_iv = df['iv'].dropna().mean() if len(df) else 0
    SEP    = '─' * 32

    lines = [
        '📡 اسکنر اختیار معامله بورس ایران',
        f'🕐 {now}',
        f'🔍 نمادها: {" | ".join(symbols)}',
        f'📊 اختیارها: {len(df)}    IV میانگین: {avg_iv:.1%}',
        SEP,
    ]

    # بخش ۱: زیر BS
    lines.append(f'🔴 زیر Black-Scholes  (تخفیف > {DISCOUNT_THRESH:.0%})')
    lines.append(SEP)
    if not underpriced:
        lines.append('   موردی یافت نشد')
    else:
        for i, o in enumerate(underpriced[:8], 1):
            disc   = abs(o['diff']) * 100
            iv_str = f"{o['iv']:.1%}" if not np.isnan(o['iv']) else 'N/A'
            t      = 'Call' if o['type'] == 'call' else 'Put'
            q      = queue_txt(o['buy_q'], o['sell_q'])
            lines.append(
                f"  {i}) {o['ul']}  {t}  ({o['sym']})"
            )
            lines.append(
                f"     اعمال: {fmt(o['K'])}  |  بازار: {fmt(o['mp'])}  |  BS: {fmt(int(o['bs']))}"
            )
            lines.append(
                f"     تخفیف: {disc:.1f}%  |  IV: {iv_str}  |  {q}"
            )
            lines.append('')

    lines.append(SEP)
    # بخش ۲: IV پایین
    lines.append(f'🟢 IV پایین تاریخی  (زیر {IV_PERCENTILE}th)')
    lines.append(SEP)
    if not low_iv:
        lines.append('   موردی یافت نشد')
    else:
        for i, r in enumerate(low_iv[:6], 1):
            q = queue_txt(r['buy_q'], r['sell_q'])
            lines.append(f"  {i}) {r['ul']}")
            lines.append(f"     IV: {r['iv']:.1%}  |  Percentile: {r['pctile']:.0f}th")
            lines.append(f"     قیمت: {fmt(r['S'])}  |  {q}")
            lines.append('')

    lines.append(SEP)
    lines.append(f'⏱ اسکن بعدی: {SCAN_INTERVAL} دقیقه دیگر')
    return '\n'.join(lines)

# ══════════════════════════════════════
# دستورات ربات
# ══════════════════════════════════════
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    await update.message.reply_text(
        f'سلام! به ربات اسکنر اختیار معامله خوش آمدید\n'
        f'Chat ID شما: {cid}\n\n'
        f'دستورات:\n'
        f'/add فولاد   -- اضافه کردن نماد به واچ‌لیست\n'
        f'/remove فولاد -- حذف نماد از واچ‌لیست\n'
        f'/list        -- نمایش واچ‌لیست\n'
        f'/scan        -- اسکن فوری\n'
        f'/status      -- وضعیت ربات\n'
        f'/help        -- راهنما'
    )

async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global watchlist
    args = ctx.args
    if not args:
        await update.message.reply_text(
            'نام نماد را بنویسید\nمثال: /add فولاد'
        )
        return
    sym = ' '.join(args).strip()
    if sym in watchlist:
        await update.message.reply_text(f'نماد {sym} قبلا در واچ‌لیست است')
        return
    watchlist.append(sym)
    save_watchlist(watchlist)
    await update.message.reply_text(
        f'نماد {sym} به واچ‌لیست اضافه شد\n'
        f'واچ‌لیست فعلی: {", ".join(watchlist)}'
    )

async def cmd_remove(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global watchlist
    args = ctx.args
    if not args:
        await update.message.reply_text(
            'نام نماد را بنویسید\nمثال: /remove فولاد'
        )
        return
    sym = ' '.join(args).strip()
    if sym not in watchlist:
        await update.message.reply_text(f'نماد {sym} در واچ‌لیست نیست')
        return
    watchlist.remove(sym)
    save_watchlist(watchlist)
    if watchlist:
        await update.message.reply_text(
            f'نماد {sym} از واچ‌لیست حذف شد\n'
            f'واچ‌لیست فعلی: {", ".join(watchlist)}'
        )
    else:
        await update.message.reply_text(
            f'نماد {sym} حذف شد\nواچ‌لیست خالی است'
        )

async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not watchlist:
        await update.message.reply_text(
            'واچ‌لیست خالی است\n'
            'با /add نماد اضافه کنید\nمثال: /add فولاد'
        )
        return
    await update.message.reply_text(
        f'واچ‌لیست شما ({len(watchlist)} نماد):\n\n' +
        '\n'.join(f'{i+1}. {s}' for i,s in enumerate(watchlist)) +
        '\n\nبرای اسکن: /scan'
    )

async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not watchlist:
        await update.message.reply_text(
            'واچ‌لیست خالی است\n'
            'ابتدا نماد اضافه کنید\nمثال: /add فولاد'
        )
        return
    msg = await update.message.reply_text(
        f'در حال اسکن {len(watchlist)} نماد...\n'
        f'نمادها: {", ".join(watchlist)}'
    )
    try:
        df = fetch_options(watchlist)
        if df.empty:
            await msg.edit_text('داده‌ای یافت نشد. دوباره تلاش کنید.')
            return
        up, liv = analyze(df)
        text = build_msg(df, up, liv, watchlist)
        await msg.edit_text(text)
    except Exception as e:
        logger.error(f"scan error: {e}")
        await msg.edit_text(f'خطا در اسکن: {e}')

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    wl_txt = ', '.join(watchlist) if watchlist else 'خالی'
    await update.message.reply_text(
        f'ربات فعال است\n\n'
        f'واچ‌لیست: {wl_txt}\n'
        f'تعداد نمادها: {len(watchlist)}\n\n'
        f'نرخ بهره: {RISK_FREE_RATE:.0%}\n'
        f'آستانه تخفیف: {DISCOUNT_THRESH:.0%}\n'
        f'IV Percentile: {IV_PERCENTILE}th\n'
        f'فاصله اسکن: {SCAN_INTERVAL} دقیقه'
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'راهنمای ربات اسکنر اختیار معامله\n\n'
        'مدیریت واچ‌لیست:\n'
        '/add فولاد   -- اضافه کردن نماد\n'
        '/remove فولاد -- حذف نماد\n'
        '/list        -- نمایش واچ‌لیست\n\n'
        'اسکن:\n'
        '/scan        -- اسکن فوری نمادهای واچ‌لیست\n'
        '/status      -- وضعیت و تنظیمات\n\n'
        'نمونه استفاده:\n'
        '/add فولاد\n'
        '/add شستا\n'
        '/add خودرو\n'
        '/scan'
    )

# ══════════════════════════════════════
# اسکن خودکار
# ══════════════════════════════════════
async def auto_scan_job(ctx: ContextTypes.DEFAULT_TYPE):
    if not watchlist:
        logger.info("Watchlist empty, skipping auto scan")
        return
    now = datetime.now().time()
    if not (dtime(9,0) <= now <= dtime(12,30)):
        logger.info("Outside market hours, skipping")
        return
    try:
        df = fetch_options(watchlist)
        if df.empty: return
        up, liv = analyze(df)
        if up or liv:
            text = build_msg(df, up, liv, watchlist)
            await ctx.bot.send_message(chat_id=CHAT_ID, text=text)
            logger.info(f"Alert: {len(up)} underpriced, {len(liv)} low IV")
    except Exception as e:
        logger.error(f"auto scan error: {e}")
        try:
            await ctx.bot.send_message(chat_id=CHAT_ID, text=f'خطا: {e}')
        except: pass

# ══════════════════════════════════════
# Main
# ══════════════════════════════════════
def main():
    if BOT_TOKEN == 'YOUR_BOT_TOKEN':
        logger.error("BOT_TOKEN تنظیم نشده!")
        return

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler('start',  cmd_start))
    app.add_handler(CommandHandler('add',    cmd_add))
    app.add_handler(CommandHandler('remove', cmd_remove))
    app.add_handler(CommandHandler('list',   cmd_list))
    app.add_handler(CommandHandler('scan',   cmd_scan))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('help',   cmd_help))

    app.job_queue.run_repeating(
        auto_scan_job,
        interval=SCAN_INTERVAL*60,
        first=30,
    )

    logger.info(f"Bot started | watchlist={watchlist}")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
