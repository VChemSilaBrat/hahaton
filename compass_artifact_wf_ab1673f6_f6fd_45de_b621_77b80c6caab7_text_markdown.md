# Winning the Sberbank gender-from-transactions Kaggle competition

The Kaggle competition **"python-and-analyze-data-final-project"** is an InClass-style task tied to a Sberbank Python+data-analysis course, asking participants to predict client gender (0 = female, 1 = male) from card-transaction history using **ROC-AUC** as the metric. The dataset is the well-known Sberbank Data Science Journey 2016 "Task A" gender dataset (mirrored on GitHub at `demidovakatya/competitions/sberbank/data.md`), redistributed for the course. Live Kaggle pages are gated by Cloudflare and could not be scraped, so leaderboard scores and individual public-notebook AUCs for this specific InClass competition are not directly retrievable. However, the dataset, task, and "what works" are heavily documented in peer-reviewed papers (CoLES, LATTE, Universal Representations) and Russian competition writeups, which give a clear, ranked, actionable recipe. **Expect a tuned LightGBM on hand-crafted MCC/amount aggregates to reach roughly 0.86–0.88 AUC**, and **CoLES sequence embeddings stacked on top to push toward 0.90–0.92 AUC** — the published industrial SOTA on this Sber gender data is 0.902 (LATTE, 2025) and 0.920 (CoLES on Sber's full internal dataset).

## Dataset, metric, and what the leaderboard likely looks like

The data has five files. `transactions.csv` holds rows of `customer_id, tr_datetime, mcc_code, tr_type, amount, term_id`, where **`tr_datetime` is a synthetic `"DD HH:MM:SS"` string** (DD is days from the start, not a real calendar date — a critical gotcha). `amount` is signed: negative is debit/spend, positive is credit/income. `customers_gender_train.csv` provides labels for a subset of customers; everyone in `transactions.csv` who is *not* in that file is the test set. `tr_mcc_codes.csv` and `tr_types.csv` are semicolon-delimited dictionaries. Roughly **~10–15k labeled customers, ~150–185 unique MCC codes, ~80–150 distinct tr_type codes, tens of millions of transactions**.

The `tr_type` taxonomy is documented and worth exploiting directly: **cash withdrawal codes 2000/2010/2020/2100/2200/2900; transfer-to-card 2330/2331/2340/2370/2440/2456; purchases 1000/1010/1100/1210/1310/1410/1510; salary deposit 7000; ATM deposits 7010-series**. Engineering binary flags from these groupings is one of the cheapest gains available.

For benchmark context across the Sber gender problem family: a naive KNN/SVM on raw MCC counts is essentially random (~51% accuracy on the rstudio-pubs student writeup); a well-tuned LightGBM on MCC pivots + amount aggregates lands **0.85–0.88 AUC**; CoLES embeddings concatenated with hand-crafted features push to **0.92** on Sber's retail-scoring set; **LATTE (2025) holds SOTA at 0.902** on the public Sber gender set; CoLES alone reaches **0.920** on Sber's industrial gender data. The Kaggle InClass leaderboard for this course is almost certainly **topped by 0.87–0.89**, with course baselines around 0.82–0.84.

## Ranked feature-engineering recipes by AUC impact

The single most important takeaway from the Sberbank Data Science Day post-mortem (Habr 318160) and the CoLES paper is blunt: **"MCC codes are more important than tr_type. Models on raw transactions don't work — only on aggregates. Shallow trees on aggregates win."** Build features at the `customer_id` level, not the transaction level.

**Tier 1 — required to break 0.83 AUC.** The workhorse is the **MCC pivot table**: `tx.groupby(['customer_id','mcc_code'])['amount'].agg(['sum','count','mean']).unstack(fill_value=0)`, plus a normalized share version (`mcc_sum.div(mcc_sum.sum(axis=1), axis=0)`). Add a **MCC × tr_type cross-pivot** of summed absolute amount — this encodes "how much did this customer cash-withdraw at this MCC" vs "how much did they purchase". On its own this family delivers ~0.82–0.85 AUC.

**Tier 2 — adds 1–3 points.** Customer-level aggregations of `amount`: **sum, mean, std, median, min, max, q05, q95, skew, kurt**, and the same on `abs(amount)` and `log1p(abs(amount))`. Split by sign: `debit_sum, debit_count, debit_mean` and `credit_sum, credit_count, credit_mean`. Add **MCC-distribution descriptors**: `n_unique_mcc`, `mcc_entropy = -Σp·log p`, `favorite_mcc`, share of top-10 MCCs.

**Tier 3 — fine-tuning, +0.005 to +0.01 each.** Time features: parse `tr_datetime` into `day = int(part1)`, `hour = part2.hour`, `dow = day % 7`, `is_weekend = dow >= 5`; build 24-bin hour-of-day and 7-bin day-of-week normalized histograms; compute `n_active_days`, `tx_per_day`, span_days, and inter-transaction-gap stats (mean, std, max). **Out-of-fold target encoding** of `mcc_code`, `tr_type`, and `(mcc_code, is_weekend)` with smoothing ~20. **Hand-crafted gender priors** as features (see MCC list below).

**Tier 4 — the game-changer beyond GBM tabular.** **CoLES contrastive sequence embeddings** via Sberbank's `pytorch-lifestream` library. Train a GRU encoder on (mcc_code, tr_type, amount, time) tuples with contrastive loss on random subsequences; the resulting 800–1024-dim embedding, **concatenated with the Tier 1–3 features and fed into LightGBM/CatBoost, is the published recipe that took Sber's retail set from 0.88 → 0.92 AUC**. CoLES paper hyperparameters for the Sber age-group dataset (which has the same schema): embedding dim 800, lr 0.001, batch 64 clients × 5 sub-sequences, 150 epochs, min_seq=25, max_seq=200, GRU encoder, classical contrastive margin loss with hard-negative mining. The Data Fusion 2024 winner extended this with a **uniform-time-grid CoLES variant** that captures transaction frequency — the single biggest gender signal.

**Tier 5 — usually subsumed.** TF-IDF on MCC sequences (`TfidfVectorizer(ngram_range=(1,2), min_df=10, sublinear_tf=True)` on space-joined MCC strings) + Logistic Regression as a stacking input. The SDSJ "MCC2VEC" team built sentences split by ≥12 h gaps and trained Word2Vec on them; CoLES ablations show this is on par with hand-crafted features but inferior to CoLES embeddings.

## Concrete pandas recipe (copy-paste ready)

```python
import pandas as pd, numpy as np
tx['day']  = tx['tr_datetime'].str.split().str[0].astype(int)
tx['hour'] = pd.to_datetime(tx['tr_datetime'].str.split().str[1]).dt.hour
tx['dow']  = tx['day'] % 7
tx['is_weekend'] = (tx['dow'] >= 5).astype(int)
tx['amount_abs'] = tx['amount'].abs()
tx['log_amount'] = np.log1p(tx['amount_abs'])

# Tier 1: MCC pivots
mcc_cnt = tx.groupby(['customer_id','mcc_code']).size().unstack(fill_value=0).add_prefix('mcc_cnt_')
mcc_sum = tx.groupby(['customer_id','mcc_code'])['amount_abs'].sum().unstack(fill_value=0).add_prefix('mcc_sum_')
mcc_share = mcc_sum.div(mcc_sum.sum(axis=1).replace(0,1), axis=0).add_prefix('share_')

# Tier 2: amount stats + debit/credit splits
amt = tx.groupby('customer_id')['amount'].agg(['sum','mean','std','median','min','max','count','skew',
        ('q05', lambda s: s.quantile(.05)), ('q95', lambda s: s.quantile(.95))])
deb = tx[tx.amount<0].groupby('customer_id')['amount_abs'].agg(['sum','mean','count']).add_prefix('deb_')
cre = tx[tx.amount>0].groupby('customer_id')['amount'].agg(['sum','mean','count']).add_prefix('cre_')

# Tier 3: time + entropy + gender priors
def entropy(s):
    p = s.value_counts(normalize=True).values; return -(p*np.log(p+1e-12)).sum()
mcc_stats = tx.groupby('customer_id').agg(n_unique_mcc=('mcc_code','nunique'),
                                          mcc_entropy=('mcc_code', entropy))
hour_h = pd.crosstab(tx.customer_id, tx.hour, normalize='index').add_prefix('hr_')
dow_h  = pd.crosstab(tx.customer_id, tx.dow,  normalize='index').add_prefix('dow_')

FEMALE_MCC = [5977,5631,5621,5651,5641,5661,5691,5697,5944,5948,5992,7230,7297,7298]
MALE_MCC   = [5541,5542,5532,5533,5571,5734,5813,5921,5945,5993,7538,7941,7995,7997]
tx['fem'] = tx.mcc_code.isin(FEMALE_MCC); tx['mal'] = tx.mcc_code.isin(MALE_MCC)
prior = tx.groupby('customer_id').agg(fem_share=('fem','mean'), mal_share=('mal','mean'))
prior['fem_minus_mal'] = prior['fem_share'] - prior['mal_share']

X = (mcc_cnt.join(mcc_sum).join(mcc_share).join(amt).join(deb).join(cre)
     .join(mcc_stats).join(hour_h).join(dow_h).join(prior).fillna(0))
```

## MCC codes that actually separate male and female

The Mastercard MCC catalog (cross-checked against `mcc-codes.ru`) gives the high-signal categories that consistently appear in feature-importance lists for Russian-bank gender models:

**Strongly female-skewed:** **5977** Cosmetic Stores, **7230** Beauty & Barber Shops, **5621** Women's Ready-to-Wear, **5631** Women's Accessories, **5641** Children's & Infants' Wear, **7297** Massage Parlors, **7298** Health & Beauty Spas, **5944** Jewelry, **5992** Florists, **5948** Luggage/Leather. Behaviorally: more transactions overall, higher grocery/pharmacy frequency, smaller mean amount, more diversified MCC distribution.

**Strongly male-skewed:** **5541/5542** Service Stations / Fuel, **5532/5533** Auto Tires/Parts, **5571** Motorcycle, **7538** Auto Service, **5813** Bars, **5921** Liquor, **5945** Hobby/Toy/Game, **5734** Computer Software, **7995** Betting/Gambling, **7997** Country/Sports Clubs, **5993** Cigar Stores. Behaviorally: higher mean amount per transaction, more cash-withdrawal volume, weekend-evening peaks at bars, higher concentration in fuel/auto.

The empirically dominant signal in the Sber data is **frequency × amount per MCC**, not raw MCC counts — the rstudio-pubs naive student baseline using only counts achieved ~51% accuracy, while amount-weighted features move performance into the useful range.

## Models, hyperparameters, and ensembling

**The competition-winning hierarchy is well-established**: LightGBM/CatBoost on engineered features ≤ supervised RNN with target ≤ CoLES-embedding + CatBoost ≤ blend of all three. For an InClass competition with limited compute, **LightGBM is almost always the winner**, and CatBoost is competitive thanks to native categorical handling of `mcc_code` and `tr_type`.

| Model | Recommended config |
|---|---|
| **LightGBM** (primary) | `objective='binary', metric='auc', learning_rate=0.02, num_leaves=63–127, max_depth=-1, min_child_samples=20–50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0, n_estimators=3000, early_stopping_rounds=200`, 5-fold stratified CV |
| **CatBoost** (secondary) | `depth=6–8, learning_rate=0.03, iterations=4000, l2_leaf_reg=5, border_count=128, cat_features=['mcc_code','tr_type'], od_type='Iter', od_wait=200`, native handling of categoricals on long-format aggregations |
| **Supervised GRU** (Data Fusion 2023 baseline) | bidirectional GRU hidden=128, embedding 8–32 per categorical (mcc, tr_type, hour, dow), spatial dropout 0.1, lr 5e-3, batch 256, ~30 epochs, signed log-amount as numerical input with batch-norm |
| **CoLES** (best lift) | GRU encoder hidden 800–1024, contrastive margin 0.5, hard-negative pair selector, lr 0.001, batch 64 clients × 5 random slices, slice length 25–200, 60–150 epochs |

**Ensembles that have demonstrably worked.** AlfaBattle 2.0 Top-6 used a flat **0.6·LGBM + 0.4·RNN blend** of probabilities (private 0.776 → 0.7812). Data Fusion 2024 winner stacked **TF-IDF-extended LightGBM + WTTE-RNN + CoLES + supervised NN with two heads (target + time-to-event)**. The simplest reliable stack for this Kaggle InClass: train 5-fold LightGBM with 5–10 different seeds, average OOF, then blend with a 5-fold logistic regression on TF-IDF MCC bigrams using weights tuned on OOF (~0.85·LGBM + 0.15·LR).

## A two-week plan to maximize AUC

Week 1 — establish the strong tabular baseline. Build the Tier 1+2 feature matrix (~500–2000 columns: MCC pivots in counts/sums/shares, MCC × tr_type cross-pivot, amount aggregates, debit/credit splits, hour/dow histograms, gender-prior shares). Train LightGBM with 5-fold stratified CV; **target 0.86–0.88 AUC**. Add OOF target encoding of `mcc_code` and `(mcc_code, is_weekend)`; expect +0.003. Verify the day index is treated correctly — `tr_datetime` is days-from-start, not a calendar date, and `dow = day % 7` is the correct day-of-week extraction.

Week 2 — add CoLES and ensemble. Install `pytorch-lifestream`, build a `MemoryMapDataset` over `(customer_id, event_time, mcc_code, tr_type, amount)`, train a GRU CoLES encoder for ~60 epochs, export the 1024-dim embedding, and concatenate it to your tabular features for a second LightGBM. Blend the two models on OOF predictions. **Realistic target: 0.89–0.91 AUC**, which would likely top the InClass leaderboard. If short on time, skip CoLES and instead run a TF-IDF + Logistic Regression model on space-joined MCC sequences and ensemble with LightGBM — this alone typically adds +0.005 AUC for almost no engineering cost.

## Conclusion

The path to a top score is sharply asymmetric: **80% of the AUC comes from a disciplined, ratio-aware MCC × amount pivot table fed to LightGBM with proper 5-fold CV**, and the remaining ~3 points come from CoLES sequence embeddings or a supervised RNN stacked on top. The most common failure mode in student notebooks is treating raw transactions as features instead of customer-level aggregates — Sberbank's own post-mortem flagged this explicitly nine years ago and it still holds. The two highest-leverage non-obvious moves are **normalized MCC-share ratios (not raw counts) combined with hand-crafted female-vs-male MCC priors**, and **CoLES embeddings via `pytorch-lifestream`** when compute allows. Everything else — target encoding, time histograms, tr_type splits — is incremental polish on a solid baseline.