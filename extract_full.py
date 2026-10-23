"""Stream the Polymarket Full Market Dataset JSON (5-7 GB) ONCE and write one compact row per resolved binary market.

Source: https://huggingface.co/datasets/od2961/polymarket-full-market-dataset (2026-04-29 snapshot)
Usage:  python extract_full.py polymarket_full_market_dataset_2026_04_29_214954.json full_markets.parquet
Needs:  pandas, pyarrow, ijson   (all in requirements.txt)

Per market it keeps metadata, the real event/series ids, the resolution, and the YES price at fixed horizons
before two anchors: the EVENT anchor (earlier of market_end_time and resolution_time) and the resolution time.
A candle counts as a real price only if its close is non-null and its raw_meta does not mark it synthetic/empty.
Every resolved binary market is written, with has_real_candles, so coverage selection can be measured later."""
import sys, json
from datetime import timedelta
import pandas as pd

HORIZONS_DAYS = (30, 7, 1)
# raw_meta keys whose TRUE value marks a candle that carries no trade of its own (values are checked, not key names)
NOT_REAL_FLAGS = ('empty', 'is_empty', 'synthetic', 'is_synthetic', 'carry_forward', 'carried_forward', 'placeholder', 'forward_filled')
META_SEEN = {}          # tally of (empty flag, source) combinations, printed at the end as a check


def ts(x):
    if x in (None, ''): return None
    try:
        t = pd.Timestamp(x)
        return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')
    except Exception:
        return None


def iter_rows(path):
    """Yield market rows from (a) a top-level JSON array, (b) a top-level JSON object whose first array-valued
    key holds the rows, or (c) a JSON-lines file.  Streams; never loads the whole file."""
    import ijson
    with open(path, 'rb') as f:
        head = f.read(64).lstrip()
    if head.startswith(b'['):
        with open(path, 'rb') as f:
            yield from ijson.items(f, 'item')
        return
    if head.startswith(b'{'):
        with open(path, 'rb') as f:                      # JSON lines?  then the first line is one complete row
            first = f.readline(20_000_000)
        if first.endswith(b'\n'):
            try:
                if isinstance(json.loads(first), dict):
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line: yield json.loads(line)
                    return
            except json.JSONDecodeError:
                pass
        key = None
        with open(path, 'rb') as f:                      # scan only until the first top-level array begins
            for prefix, event, value in ijson.parse(f):
                if event == 'start_array' and prefix and '.' not in prefix:
                    key = prefix; break
        if key is None: raise ValueError('top-level object contains no array of rows')
        print(f'  streaming rows from top-level key "{key}" (ijson backend: {ijson.backend})', flush=True)
        with open(path, 'rb') as f:
            yield from ijson.items(f, f'{key}.item')
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)


def is_real(c):
    """A candle is a real price only if it has a close and no raw_meta flag says it is empty or synthetic."""
    meta = c.get('raw_meta') or {}
    if not isinstance(meta, dict):
        try: meta = json.loads(meta)
        except Exception: meta = {}
    key = (bool(meta.get('empty')), str(meta.get('source'))[:40], str(meta.get('empty_reason'))[:30])
    META_SEEN[key] = META_SEEN.get(key, 0) + 1
    if c.get('close') is None: return False
    return not any(bool(meta.get(k)) for k in NOT_REAL_FLAGS)


def yes_candles(row):
    ids = row.get('ids') or {}; toks = ids.get('clob_token_ids') or []
    yes_tok = str(toks[0]) if toks else None; yes_label = (row.get('outcomes') or {}).get('yes_label')
    out = []
    for c in row.get('candles') or []:
        if (c.get('period_interval_minutes') or 1440) != 1440: continue
        if str(c.get('token_id')) == yes_tok or (yes_label and c.get('outcome') == yes_label):
            t = ts(c.get('end_period_ts') or c.get('start_period_ts'))
            if t is not None: out.append((t, c))
    out.sort(key=lambda x: x[0]); return out


def price_before(cands, target):
    """Last real close at or before target: (price, staleness in days), or (None, None)."""
    best = None
    for t, c in cands:
        if t <= target and is_real(c): best = (t, c)
    if best is None: return None, None
    return float(best[1]['close']), (target - best[0]).total_seconds() / 86400


def extract(row):
    outcomes = row.get('outcomes') or {}; res = row.get('resolution') or {}
    if not outcomes.get('is_binary') or res.get('winner_yes') is None: return None
    ids, mkt, ev = row.get('ids') or {}, row.get('market') or {}, row.get('event') or {}
    mcr, mm, raw = row.get('market_closing_rules') or {}, row.get('market_memory') or {}, (row.get('market') or {}).get('raw_meta') or {}
    t_res = ts(res.get('resolution_time')) or ts(res.get('closed_time')) or ts(mcr.get('close_time'))
    t_end = ts(mcr.get('market_end_time')) or ts(mcr.get('event_end_time')) or ts(mcr.get('end_date_raw'))
    t_start = ts(mcr.get('market_start_time')) or ts(mcr.get('event_start_time')) or ts(raw.get('created_at')) or ts(raw.get('createdAt'))
    anchor_event = min([t for t in (t_end, t_res) if t is not None], default=None)
    cands = yes_candles(row); real = [(t, c) for t, c in cands if is_real(c)]
    out = dict(market_id=ids.get('market_id'), event_id=ids.get('event_id'), condition_id=ids.get('condition_id'),
               question=row.get('question'), event_title=ev.get('title'), event_slug=ev.get('slug'),
               category=ev.get('category') or mm.get('category'), series_id=ev.get('series_id'),
               tags='|'.join(str(t.get('label')) for t in (ev.get('tags') or []) if isinstance(t, dict)),
               negative_risk=mkt.get('negative_risk'), volume=None if mkt.get('volume') is None else float(mkt.get('volume')),
               t_start=t_start, t_end=t_end, t_resolution=t_res, anchor_event=anchor_event,
               resolution_method=res.get('resolution_method'), uma_status=res.get('uma_resolution_status'),
               y=int(bool(res.get('winner_yes'))),
               n_candles=len(cands), n_real_candles=len(real), has_real_candles=len(real) > 0,
               q_open=float(real[0][1]['close']) if real else None, t_open=real[0][0] if real else None,
               q_last=float(real[-1][1]['close']) if real else None, t_last=real[-1][0] if real else None)
    for name, anchor in (('event', anchor_event), ('res', t_res)):
        for h in HORIZONS_DAYS:
            p, stale = price_before(real, anchor - timedelta(days=h)) if anchor is not None else (None, None)
            out[f'q_{h}d_{name}'] = p; out[f'stale_{h}d_{name}'] = stale
    return out


def main(src, dst):
    rows, n_seen, n_binary_resolved = [], 0, 0
    for row in iter_rows(src):
        n_seen += 1
        r = extract(row)
        if r is not None: rows.append(r); n_binary_resolved += 1
        if n_seen % 50_000 == 0: print(f'  {n_seen:,} rows scanned, {n_binary_resolved:,} resolved binary kept', flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(dst, index=False)
    print('candle raw_meta combinations seen (empty flag, source, empty_reason): count')
    for k, v in sorted(META_SEEN.items(), key=lambda kv: -kv[1])[:12]: print(f'   {k}: {v:,}')
    print(f'scanned {n_seen:,} markets; wrote {len(df):,} resolved binary markets to {dst}; '
          f'{int(df.has_real_candles.sum()):,} have at least one real price; '
          f'{int(df.q_1d_event.notna().sum()):,} have a price 1 day before the event anchor')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
