import logging, time
from concurrent.futures import ThreadPoolExecutor
from tradinglab.config import load_config
from tradinglab.data.store import MarketData
logging.basicConfig(level=logging.INFO)
cfg = load_config()
md = MarketData(cfg.paths.data_dir)
def one(s):
    t0 = time.time(); k = md.klines(s, "5m", "2022-07-01", market="spot")
    return s, len(k), str(k.index[0]), str(k.index[-1]), round(time.time() - t0)
with ThreadPoolExecutor(3) as pool:
    for r in pool.map(one, cfg.intraday.symbols):
        print(r, flush=True)
