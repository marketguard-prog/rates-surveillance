# %% [markdown]
# # Detecting Spoofing in Fixed Income Futures: A Layered Approach
#
# *Companion notebook to Issue 1 of the newsletter.*
#
# A walkthrough of three signals — order-to-trade ratio (OTR) with side decomposition,
# cancel-to-fill latency, and a composite score — applied to a synthetic population
# of Treasury-futures traders, two of whom are systematically spoofing.
#
# All data is synthetic. The detection logic is illustrative, not production.
# What's missing — book-state reconstruction, real-time alerting, false-positive
# reduction layer, multi-instrument correlation, episode clustering — is intentional.
# This is the demo. The production versions are the paid notebooks.
#
# MIT licensed. Reply to the newsletter with feedback or pushback.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

N_HONEST = 16
N_SPOOFERS = 2
N_DIRECTIONAL = 2
DAY_LENGTH_MS = 8 * 60 * 60 * 1000  # 8 hours

# %% [markdown]
# ## 1. Synthetic data generation
#
# We generate order events for 20 traders over an 8-hour session. Each event has
# a millisecond timestamp, a trader ID, an action (place / cancel / fill), a side
# (buy / sell), a size, and an order ID.
#
# - **16 honest traders** place orders with realistic order-to-trade ratios (5–30:1),
#   roughly symmetric on each side, with cancels uncorrelated to their fills.
# - **2 spoofers** place 2–3 large phantom orders on one side, then a smaller genuine
#   order on the opposite side, then cancel the phantoms shortly after the genuine
#   order fills.
# - **2 directional traders** have low OTR and one-sided flow (long-only or short-only).
#
# The point is to generate data where the spoofers' statistical signature is *latent*.
# You can't pick them out from the raw event stream by eye. You need the detectors.

# %%
def generate_honest_trader(trader_id, n_orders=200):
    events = []
    for i in range(n_orders):
        t_place = np.random.randint(0, DAY_LENGTH_MS)
        side = np.random.choice(['buy', 'sell'])
        size = int(np.random.choice([50, 100, 200, 300], p=[0.4, 0.3, 0.2, 0.1]))
        order_id = f"{trader_id}_o{i}"
        events.append((t_place, trader_id, 'place', side, size, order_id))
        outcome = np.random.choice(['fill', 'cancel', 'expire'], p=[0.35, 0.55, 0.10])
        if outcome == 'fill':
            t_end = t_place + np.random.randint(500, 30000)
            events.append((min(t_end, DAY_LENGTH_MS), trader_id, 'fill', side, size, order_id))
        elif outcome == 'cancel':
            t_end = t_place + np.random.randint(100, 60000)
            events.append((min(t_end, DAY_LENGTH_MS), trader_id, 'cancel', side, 0, order_id))
    return events


def generate_directional_trader(trader_id, side, n_orders=100):
    events = []
    for i in range(n_orders):
        t_place = np.random.randint(0, DAY_LENGTH_MS)
        size = int(np.random.choice([100, 200, 500], p=[0.5, 0.3, 0.2]))
        order_id = f"{trader_id}_o{i}"
        events.append((t_place, trader_id, 'place', side, size, order_id))
        outcome = np.random.choice(['fill', 'cancel'], p=[0.85, 0.15])
        t_end = t_place + np.random.randint(200, 5000) if outcome == 'fill' \
            else t_place + np.random.randint(100, 20000)
        events.append((min(t_end, DAY_LENGTH_MS), trader_id, outcome, side,
                       size if outcome == 'fill' else 0, order_id))
    return events


def generate_spoofer(trader_id, n_episodes=80, bias=0.85):
    """Spoofer is directionally biased — most episodes trade the same genuine side.
    Realistic: real spoofing cases usually show a directional bias over the day."""
    events = []
    biased_side = np.random.choice(['buy', 'sell'])
    for ep in range(n_episodes):
        # `bias` fraction of episodes go in the biased direction
        if np.random.random() < bias:
            genuine_side = biased_side
        else:
            genuine_side = 'sell' if biased_side == 'buy' else 'buy'
        phantom_side = 'sell' if genuine_side == 'buy' else 'buy'
        t_start = np.random.randint(0, DAY_LENGTH_MS - 5000)

        # 2–3 phantom orders, large size
        n_phantoms = np.random.randint(2, 4)
        phantom_orders = []
        for i in range(n_phantoms):
            t = t_start + int(np.random.randint(0, 200))
            size = int(np.random.choice([300, 400, 500, 600, 700]))
            oid = f"{trader_id}_p{ep}_{i}"
            events.append((t, trader_id, 'place', phantom_side, size, oid))
            phantom_orders.append((oid, t))

        # Genuine order (smaller, opposite side)
        t_genuine = t_start + int(np.random.randint(150, 500))
        genuine_size = int(np.random.choice([100, 200, 300, 400]))
        gid = f"{trader_id}_g{ep}"
        events.append((t_genuine, trader_id, 'place', genuine_side, genuine_size, gid))

        # Genuine fills quickly
        t_fill = t_genuine + int(np.random.randint(50, 250))
        events.append((t_fill, trader_id, 'fill', genuine_side, genuine_size, gid))

        # Cancel phantoms shortly after fill — the signature
        for oid, _ in phantom_orders:
            t_cancel = t_fill + int(np.random.randint(30, 400))
            events.append((t_cancel, trader_id, 'cancel', phantom_side, 0, oid))
    return events


all_events = []
for i in range(N_HONEST):
    all_events.extend(generate_honest_trader(f"H{i:02d}"))
for i in range(N_SPOOFERS):
    all_events.extend(generate_spoofer(f"S{i:02d}"))
for i in range(N_DIRECTIONAL):
    side = 'buy' if i == 0 else 'sell'
    all_events.extend(generate_directional_trader(f"D{i:02d}", side))

df = pd.DataFrame(all_events,
                  columns=['ts', 'trader_id', 'action', 'side', 'size', 'order_id'])
df = df.sort_values('ts').reset_index(drop=True)
print(f"Generated {len(df):,} events across {df['trader_id'].nunique()} traders")
df.head(10)

# %% [markdown]
# ## 2. Quick look — can you spot the spoofers by eye?
#
# Volume per trader, per side. Spoofers will look like high-volume two-sided
# market makers in aggregate. They are not.

# %%
volume = df.groupby(['trader_id', 'action', 'side']).size().unstack(fill_value=0)
print(volume.head(20))

# %% [markdown]
# ## 3. Detector 1 — Order-to-Trade Ratio, with side decomposition
#
# Honest market makers have OTR around 5–30:1 and roughly symmetric across sides.
# Spoofers have OTR that is high on one side (the phantom side) and normal on
# the other. The *asymmetry between sides* is the cleanest single number.

# %%
def compute_otr(df):
    rows = []
    for trader_id, group in df.groupby('trader_id'):
        for side in ['buy', 'sell']:
            sub = group[group['side'] == side]
            placed = int((sub['action'] == 'place').sum())
            filled = int((sub['action'] == 'fill').sum())
            rows.append({'trader_id': trader_id, 'side': side,
                         'placed': placed, 'filled': filled,
                         'otr': placed / max(filled, 1)})
    return pd.DataFrame(rows)


otr_df = compute_otr(df)
otr_pivot = otr_df.pivot(index='trader_id', columns='side', values='otr').reset_index()
otr_pivot['asymmetry'] = (otr_pivot['buy'] - otr_pivot['sell']).abs() / \
    otr_pivot[['buy', 'sell']].max(axis=1).replace(0, 1)
otr_pivot['type'] = otr_pivot['trader_id'].apply(
    lambda x: 'spoofer' if x.startswith('S') else ('directional' if x.startswith('D') else 'honest'))
print(otr_pivot.sort_values('asymmetry', ascending=False).head(8))

# Plot: buy-side OTR vs sell-side OTR
colors = {'honest': '#4682B4', 'spoofer': '#DC143C', 'directional': '#DAA520'}
fig, ax = plt.subplots(figsize=(9, 7))
for t in ['honest', 'directional', 'spoofer']:
    sub = otr_pivot[otr_pivot['type'] == t]
    ax.scatter(sub['buy'], sub['sell'], c=colors[t], label=t, s=140,
               edgecolor='black', linewidth=0.7, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(r['trader_id'], (r['buy'], r['sell']), fontsize=7,
                    ha='center', va='center', zorder=4)
mx = otr_pivot[['buy', 'sell']].values.max() * 1.1
ax.plot([0, mx], [0, mx], 'k--', alpha=0.3, label='symmetric')
ax.set_xlabel('Buy-side OTR')
ax.set_ylabel('Sell-side OTR')
ax.set_title('OTR by side — spoofers sit far from the diagonal')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('otr_by_side.png', dpi=120)
plt.show()

# %% [markdown]
# ## 4. Detector 2 — Cancel-to-Fill Latency
#
# For each fill, find the next opposite-side cancel by the same trader within 2
# seconds, and record the time delta. Honest traders cancel for reasons
# uncorrelated to their fills — the latencies look uniform over many seconds.
# Spoofers cancel because they just filled — latencies cluster tightly near zero.

# %%
def compute_cancel_to_fill(df, window_ms=2000):
    fills = df[df['action'] == 'fill'][['ts', 'trader_id', 'side']].reset_index(drop=True)
    cancels = df[df['action'] == 'cancel'][['ts', 'trader_id', 'side']].sort_values('ts')
    rows = []
    for _, f in fills.iterrows():
        opp = 'sell' if f['side'] == 'buy' else 'buy'
        cand = cancels[
            (cancels['trader_id'] == f['trader_id']) &
            (cancels['side'] == opp) &
            (cancels['ts'] > f['ts']) &
            (cancels['ts'] < f['ts'] + window_ms)
        ]
        if len(cand) > 0:
            rows.append({'trader_id': f['trader_id'],
                         'latency_ms': int(cand['ts'].min() - f['ts'])})
    return pd.DataFrame(rows)


ctf = compute_cancel_to_fill(df)
ctf['type'] = ctf['trader_id'].apply(
    lambda x: 'spoofer' if x.startswith('S') else ('directional' if x.startswith('D') else 'honest'))

ctf_stats = ctf.groupby('trader_id').agg(
    n_pairs=('latency_ms', 'count'),
    median_latency=('latency_ms', 'median'),
    p10_latency=('latency_ms', lambda x: x.quantile(0.1)),
).reset_index()
ctf_stats['type'] = ctf_stats['trader_id'].apply(
    lambda x: 'spoofer' if x.startswith('S') else ('directional' if x.startswith('D') else 'honest'))
print(ctf_stats.sort_values('median_latency').head(10))

# Plot
fig, ax = plt.subplots(figsize=(11, 7))
order = ctf_stats.sort_values('median_latency')['trader_id'].tolist()
data = [ctf[ctf['trader_id'] == tid]['latency_ms'].values for tid in order]
bp = ax.boxplot(data, positions=np.arange(len(order)), vert=False, widths=0.6,
                patch_artist=True, showfliers=False)
for patch, tid in zip(bp['boxes'], order):
    t = 'spoofer' if tid.startswith('S') else 'directional' if tid.startswith('D') else 'honest'
    patch.set_facecolor(colors[t])
    patch.set_alpha(0.8)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel('Latency from fill to next opposite-side cancel (ms)')
ax.set_title('Cancel-to-fill latency per trader — tight clusters near zero are the signal')
ax.set_xlim(0, 1500)
ax.grid(alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('cancel_to_fill_latency.png', dpi=120)
plt.show()

# %% [markdown]
# ## 5. Composite score
#
# Each signal in isolation produces false positives. The composite ranks traders
# by combining OTR asymmetry and cancel-to-fill latency. Real production scoring
# would normalize across longer windows, weight signals empirically, and pass
# through a triage layer. Here we just z-score and sum.

# %%
# Require minimum two-sided activity to compute meaningful OTR asymmetry —
# directional traders (one-sided flow) shouldn't be confused with spoofers.
MIN_PLACES_PER_SIDE = 30
MIN_LATENCY_PAIRS = 10

two_sided = otr_df.pivot(index='trader_id', columns='side', values='placed').reset_index()
two_sided['has_two_sides'] = (two_sided['buy'] >= MIN_PLACES_PER_SIDE) & \
                              (two_sided['sell'] >= MIN_PLACES_PER_SIDE)

score = otr_pivot[['trader_id', 'asymmetry', 'type']].merge(
    ctf_stats[['trader_id', 'n_pairs', 'median_latency']], on='trader_id', how='left'
).merge(two_sided[['trader_id', 'has_two_sides']], on='trader_id')

# OTR asymmetry is only meaningful for two-sided participants.
score['asym_eff'] = score['asymmetry'].where(score['has_two_sides'], 0)
# Latency signal requires enough paired observations to trust.
score['n_pairs'] = score['n_pairs'].fillna(0).astype(int)
score['lat_eff'] = score['median_latency'].where(score['n_pairs'] >= MIN_LATENCY_PAIRS, 2000)

# z-score each signal
score['asym_z'] = (score['asym_eff'] - score['asym_eff'].mean()) / score['asym_eff'].std()
score['lat_z'] = -(score['lat_eff'] - score['lat_eff'].mean()) / score['lat_eff'].std()
score['composite'] = score['asym_z'] + score['lat_z']
print(score.sort_values('composite', ascending=False)[
    ['trader_id', 'type', 'asymmetry', 'n_pairs', 'median_latency', 'composite']].head(10))

# %% [markdown]
# ## 6. What's deliberately missing
#
# This notebook teaches the *shape* of the problem. The gap between this and a
# production detector is large, and exists on purpose:
#
# - **Book-state reconstruction.** Real spoofing detection requires reconstructing
#   the L2 order book at the moment of each genuine fill so you can measure depth
#   contribution — what fraction of visible bid- or offer-side depth was this
#   trader's, in the moments before they traded the opposite side. Without that,
#   you can't tell whether the phantom orders actually moved anything.
#
# - **False-positive reduction.** The composite will flag legitimate edge cases:
#   directional traders who happen to cancel a small opposite-side order just
#   after a fill, market makers withdrawing one side around news events. Production
#   systems need a triage layer that consumes context (news, vol regime,
#   trader's normal pattern) before raisi