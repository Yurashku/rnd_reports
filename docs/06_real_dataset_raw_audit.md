# R&D-6: Real dataset raw audit

- **Version:** v1.0
- **Related:** `docs/06_safe_intime_cupac_context.md`, `docs/06_datasets_setup.md`,
  `src/rnd_reports/datasets/inspect.py`, `tools/audit_datasets.py`
- **Scope:** honest raw inspection of public uplift / treatment–control datasets to decide
  what level of validation each supports for **Safe in-time CUPAC** (feature classes
  A/B/C/D/E/F + `unsafe_demo`, see context doc §3).

This is a raw audit, not a documentation review. Public datasets were downloaded locally into
the **gitignored** path `data/06_safe_intime_cupac/<dataset>/` and inspected with
`schema_summary` (dtypes, missingness, cardinality, example values, datetime/id/binary/
treatment-target flags). **No raw data is committed.** Datasets that could not be loaded are
explicitly marked as such — uninspected schemas are never presented as locally confirmed.

> **Feature-class reminder.** A = pre-treatment; B = expert-safe in-time; C = balance-gated
> in-time; D = DAG-required; E = mediator-risk; F = leakage/outcome-derived; `unsafe_demo` =
> demonstration-only. Real **B/C** require *named, semantically interpretable in-time columns
> with defensible timing evidence* (we know they are observed during the experiment and that
> treatment cannot influence them, or they pass a balance gate). Anonymized `f*`/`PC*` columns
> and pre-period aggregates do **not** qualify as B/C.

---

## Executive summary

### Verdict table

| dataset | loaded_locally | source | randomized / T–C | treatment_col | target_col | feature_timing_evidence | A_candidates | B_candidates | C_candidates | D/E/F/unsafe candidates | validation_level | recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Orange Belgium churn | **yes** (11,896×180) | Verhelst 2025; GitHub Dropbox CSV; OpenML 45580 | yes (RCT; high-risk subset randomized) | `t` | `y` (churn) | none — `PC1..PC160` are PCA components, `FACTOR1..18` anonymized codes; no dates | PCA/anon features (A-ish) | none | none | none | **A_only** | A-only; even A is PCA-transformed (low interpretability) |
| Lenta | **yes** (687,029×195) | scikit-uplift `fetch_lenta` | needs verification (treated as A/B; RCT not clearly proven in docs) | `group` (test/control) | `response_att` | named **pre-campaign** aggregates over 15d/1m/3m/6m/12m windows | many named history aggregates | none | none (15d window is pre-period, not in-time) | none clearly labelled | **A_only** | best large named-feature A-only dataset |
| Criteo Uplift v2.1 | **no** (sklift S3 → HTTP 403) | arXiv 2111.10106; scikit-uplift `fetch_criteo` | yes (RCT / incrementality) | `treatment` | `visit`, `conversion` | none — `f0..f11` anonymized; `exposure` is post-treatment | `f0..f11` (weak, anon) | none | none | **`exposure` → E/mediator + unsafe_demo** | **A_only** *(docs_only_not_locally_confirmed)* | weak A-only; use `exposure` only as E/unsafe demonstration |
| X5 RetailHero | **yes** (train 200,039; clients 400,162; purchases 45.8M) | ods.ai competition | needs verification (treatment_flg ≈ 50/50) | `treatment_flg` | `target` | **real `transaction_datetime`** in multi-table purchase log; **campaign date not published** | engineered pre-communication purchase history | none | none defensible (cannot split pre/in-time without campaign date) | post-communication purchases → E/F if misused | **A_only** | strongest real multi-table+timestamp set; A-only today, true C only if campaign window obtained |
| MegaFon | **yes** (600,000×52) | scikit-uplift `fetch_megafon` | treatment_group present (generated) | `treatment_group` | `conversion` | none — `X_1..X_50` anonymized synthetic | anon synthetic | none | none | none | **not_suitable** (synthetic/generated) | sanity / scale-robustness benchmark only, not real validation |
| Lazada / DESCN | **yes** (test 181,669; train 926,669) | GitHub `kailiang-zhong/DESCN`; Dropbox zip | test = randomized; **train = biased by prior policy** | `is_treat` | `label` (conversion) | none — `f0..f82` anonymized | anon `f0..f82` | none | none | biased train → D/propensity demo | **A_only** (on randomized test) | A-only on randomized test; useful for biased-train/propensity demos |
| CUVET-policy | **partial** (HF viewer schema; 13.7 GB not fully pulled) | HuggingFace `anonadata/CUVET-policy` | yes (2-week A/B, 5 continuous arms) | `policy` | `value`, `cost` | features collected at bidding moment but `f0..f4` anonymized | anon `f0..f4` | none | none | continuous treatment / cost = D-ish policy problem | **A_only** *(schema from HF viewer)* | A-only; large continuous-treatment benchmark; CC-BY-NC-SA (non-commercial) |
| MT-LIFT | **no** (Google/Baidu drive) | GitHub `MTDJDSP/MT-LIFT` | yes (RCT, multi-treatment 0–4) | `treatment` | `click`, `conversion` | none — `f0..f98` anonymized snapshot | anon `f0..f98` | none | none | **funnel click→conversion → click is E/mediator** | **access_blocked** (would be A_only + E demo) | request access; funnel = mediator/E demonstration, not B/C |
| Dunnhumby Complete Journey | **no** (registration form) | dunnhumby.com source-files | **no — observational** (coupon not randomized) | coupon exposure | redemption / spend | **rich**: 2-yr household transactions + campaigns + direct-marketing contact history with dates | engineered pre-coupon history | possible *construction* (not RCT) | possible *construction* (not RCT) | exposure/contact → D/E; post-coupon behavior → E/F | **access_blocked + not_suitable as RCT** | best multi-table+timing candidate for *constructing* C/D, but observational → demonstration only |
| Kaggle "Orange Telecom Churn" (mnassrib) | n/a | kaggle.com `mnassrib/telecom-churn` | **no treatment/control** | — | churn | — | — | — | — | — | **rejected** (ordinary churn prediction) | reject; not an uplift dataset (spreadsheet conflated it with Verhelst Orange Belgium) |
| Criteo-ITE | n/a | GitHub `criteo-research/large-scale-ITE` | simulated potential outcomes | simulated | synthetic CATE | — | — | none | none | semi-synthetic | **not_suitable** (semi-synthetic) | CATE/ITE benchmark, not real treatment |
| IHDP / Twins / ACIC 2016–2022 / Lalonde-NSW | n/a (docs) | CEVAE / causallib / Synapse / DoWhy | simulated DGP (Lalonde-NSW: real RCT subset) | simulated `t` | simulated outcome | none in-time | real covariates | none | none | **D / causal-validation references** | **not_suitable for B/C** | reserve for a future DAG/causal (D) validation stage, not for B/C uplift |
| Hillstrom | yes (implemented) | MineThatData | yes (RCT) | `segment` | `spend` | none in-time | named pre-treatment | none | none | `visit`/`conversion` → E | **A_only** | already implemented; pipeline/A-only smoke |

### Direct answers to the audit questions

- **Is full A+B+C validation possible on public datasets?** **No — not confirmed.** No public
  uplift/RCT dataset exposes real *safe in-time* B/C columns. Features are either anonymized
  (`f*`, `PC*`, `X_*`), pre-period aggregates (A), or post-treatment behavior (E/F). The only
  datasets with genuine in-time timestamps (X5, Dunnhumby) either hide the treatment/campaign
  date (X5) or are not randomized (Dunnhumby).
- **Which datasets are good for A-only CUPAC validation?** Lenta (large, named features),
  X5 RetailHero (engineered purchase history), Orange Belgium (PCA features), Lazada/DESCN
  (randomized test set), Hillstrom (already wired), CUVET-policy (anon, large), and Criteo
  (weak, anon — once a working mirror is available).
- **Which datasets provide real B/C candidates?** **None with defensible RCT timing.** The
  closest *construction* opportunities are X5 (`transaction_datetime` exists, but no published
  campaign date) and Dunnhumby (timestamps + contact history, but observational and
  access-gated). Neither yields a clean, randomized B/C today.
- **Which datasets are useful only for unsafe/demo or scale testing?** MegaFon (synthetic →
  scale/robustness), Criteo `exposure` and MT-LIFT `click` (mediator/E demonstration),
  CUVET-policy (continuous-treatment scale), Dunnhumby (post-coupon behavior as E/F demo).
- **Which datasets are not suitable?** Kaggle mnassrib telecom-churn (no treatment/control —
  ordinary prediction), Criteo-ITE and the CATE benchmarks IHDP/Twins/ACIC (semi-synthetic,
  simulated treatment — reserved as future **D** references, not B/C).

### Five-way classification (as requested)

1. **Classic uplift, mostly A-only:** Lenta, X5 RetailHero, Orange Belgium, Lazada/DESCN,
   Hillstrom, CUVET-policy, Criteo (weak/anon).
2. **Possible C candidates (construction only, with caveats):** X5 (timestamps, campaign date
   hidden), Dunnhumby (timestamps + contacts, observational). **No clean RCT C exists.**
3. **Possible D/E/F/unsafe candidates:** Criteo `exposure` (E), MT-LIFT `click` funnel (E),
   X5/Dunnhumby post-contact behavior (E/F), biased Lazada train (D/propensity),
   CUVET continuous policy (D-ish).
4. **Rejected ordinary prediction (no treatment/control):** Kaggle mnassrib telecom-churn;
   generic churn/marketing datasets without an experimental treatment.
5. **Access-blocked / semi-synthetic-only:** Criteo (sklift S3 403 at audit time),
   MT-LIFT (Google/Baidu drive), Dunnhumby (registration form), Kaggle (auth), ACIC 2018
   (Synapse registration); semi-synthetic: Criteo-ITE, IHDP, Twins, ACIC, Lalonde-NSW.

### Conclusion

> **Public uplift/RCT datasets mostly support A-only CUPAC validation. Full real-data A+B+C
> validation is not confirmed because real safe in-time B/C columns are absent, anonymized, or
> lack timing evidence.**

The synthetic POC (see `rnd/06_safe_intime_cupac/report.md`) remains the only place where the
full `A → A+B → A+B+C` chain is validated end-to-end, because only there are the B/C/D/E/F
labels known by construction.

---

## Dataset details

### 1. Orange Belgium churn uplift

- **Source:** Verhelst et al., 2025 — *A Churn Prediction Dataset from the Telecom Sector*
  (arXiv 2312.07206); GitHub `TheoVerhelst/Churn-Uplift-Dataset-Paper`; OpenML id 45580.
- **Access method / fetch:** direct CSV from the GitHub-linked Dropbox
  (`churn_uplift_anonymized.csv`, ~37 MB) → `data/06_safe_intime_cupac/orange_belgium/`.
  (`tools/audit_datasets.py` also attempts `fetch_openml(data_id=45580)`.)
- **Loaded locally:** **yes** — 11,896 rows × 180 columns.
- **Treatment / target:** `t` (binary 0/1), `y` (binary churn). Randomized retention offer; the
  paper randomizes a high-risk subset → genuine T–C.
- **Feature families:** `PC1..PC160` (float **principal components**) + `FACTOR1..FACTOR18`
  (anonymized categorical `V*` codes). **All features anonymized / PCA-transformed.**
- **Named vs anonymized:** fully anonymized.
- **Timing evidence:** none — no dates, no in-time columns; PCA destroys per-feature semantics.
- **A:** PCA components (usable as A). **B:** none. **C:** none. **D/E/F/unsafe:** none.
- **Full A+B+C possible:** no.
- **Limitations:** PCA features cannot be reasoned about for safety; only A-style adjustment.
- **Recommendation:** A-only candidate; lowest interpretability of the A-only group.

### 2. Lenta

- **Source:** <https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html>.
- **Access method / fetch:** `sklift.datasets.fetch_lenta(data_home=…)`.
- **Loaded locally:** **yes** — 687,029 rows × 195 columns.
- **Treatment / target:** `group` (`test`/`control`), `response_att` (binary).
- **Feature families:** named customer/purchase-history aggregates over **15d / 1m / 3m / 6m /
  12m** windows (`cheque_count_*`, `sale_sum_*`, `food_share_15d`, `k_var_cheque_15d`,
  `stdev_days_between_visits_15d`, …), plus `age`, `gender`, `children`.
- **Named vs anonymized:** named (semantically interpretable) — rare and valuable.
- **Timing evidence:** windows are **pre-campaign aggregates** (history ending before the
  campaign). The 15d window is the shortest but is still pre-period — **not** in-time.
- **A:** the full named aggregate set. **B:** none (no external/context in-time variables).
  **C:** none (no in-time columns; short windows are pre-period). **D/E/F:** none clearly.
- **Full A+B+C possible:** no.
- **Limitations:** randomization not strongly documented → marked needs-verification.
  Note: `schema_summary` flags `stdev_days_between_visits_15d` as target/treatment-like (the
  substring "visit" matches the heuristic) — a known false positive, it is a pre-period feature.
- **Recommendation:** best **A-only** dataset with named features; ideal first real adapter.

### 3. Criteo Uplift v2.1

- **Source:** arXiv 2111.10106 (Diemert et al.); scikit-uplift `fetch_criteo`.
- **Access method / fetch:** `sklift.datasets.fetch_criteo(percent10=True, data_home=…)`.
- **Loaded locally:** **no.** At audit time the sklift download returned
  `HTTPError: 403 Forbidden` for `…/criteo10.csv.gz` (S3 bucket access changed).
- **Treatment / target:** `treatment`; targets `visit`, `conversion` (per docs).
- **Feature families:** `f0..f11` anonymized floats; plus `exposure` (actual ad shown / RTB win).
- **Named vs anonymized:** anonymized.
- **Timing evidence:** none for `f*`. `exposure` is realized **after** assignment →
  post-treatment / mediator.
- **A:** `f0..f11` (weak, anon). **B:** none. **C:** none. **D/E/F/unsafe:** `exposure` is a
  classic **E/mediator** and an `unsafe_demo` candidate (conditioning on it biases ATE).
- **Full A+B+C possible:** no.
- **Limitations:** could not be locally confirmed → **docs_only_not_locally_confirmed**.
- **Recommendation:** weak A-only once a working mirror exists; use `exposure` strictly as an
  E/`unsafe_demo` illustration, never as safe B/C.

### 4. X5 RetailHero

- **Source:** <https://ods.ai/competitions/x5-retailhero-uplift-modeling>; scikit-uplift `fetch_x5`.
- **Access method / fetch:** `sklift.datasets.fetch_x5(data_home=…)` (multi-table Bunch).
- **Loaded locally:** **yes** — `uplift_train` 200,039 rows (`treatment_flg` ≈ 50/50,
  target≈62% positive), `clients` 400,162 rows, `purchases` **45,786,568** rows.
- **Treatment / target:** `treatment_flg`, `target` (in `uplift_train`, keyed by `client_id`).
- **Feature families:** `clients` (`first_issue_date`, `first_redeem_date`, `age`, `gender`);
  `purchases` (real **`transaction_datetime`**, points, `purchase_sum`, `store_id`,
  `product_id`, quantities). Multi-table, requires ETL/aggregation.
- **Named vs anonymized:** named, with genuine timestamps.
- **Timing evidence:** **real datetimes in the purchase log** — the only fully-downloadable set
  with true event timing. **But the campaign/communication date is not published**, so you
  cannot split purchases into pre- vs in-treatment defensibly.
- **A:** engineered pre-communication purchase-history aggregates. **B:** none. **C:** **none
  defensible** today (no published treatment timestamp). **D/E/F:** purchases dated after the
  (unknown) communication would be E/F if accidentally included.
- **Full A+B+C possible:** no (today). Conditional: with a published campaign window one could
  attempt balance-gated C — out of scope here and not currently supported by the public data.
- **Limitations:** randomization needs verification; ETL-heavy; campaign date hidden.
- **Recommendation:** strongest real multi-table+timestamp dataset → **A-only** adapter now;
  flag as the single best future-C candidate *if* the treatment window becomes available.

### 5. MegaFon

- **Source:** <https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html>.
- **Access method / fetch:** `sklift.datasets.fetch_megafon(data_home=…)`.
- **Loaded locally:** **yes** — 600,000 rows × 52 columns.
- **Treatment / target:** `treatment_group` (`control`/`treatment`), `conversion` (binary).
- **Feature families:** `X_1..X_50` anonymized floats (continuous, no missingness).
- **Named vs anonymized:** anonymized **and synthetic/generated** from telecom logs.
- **Timing evidence:** none.
- **A/B/C/D/E/F:** anon synthetic features only; no safety-class signal.
- **Full A+B+C possible:** no.
- **Limitations:** synthetic/generated → cannot validate causal correctness on real behavior.
- **Recommendation:** **not_suitable** for real validation; keep only as a scale/robustness
  sanity benchmark. Catalog `kind` set to `synthetic`.

### 6. Lazada / DESCN (discovery)

- **Source:** GitHub `kailiang-zhong/DESCN` (DESCN paper, KDD 2022); Dropbox `lzd_data_public.zip`.
- **Access method / fetch:** direct Dropbox zip → `data/06_safe_intime_cupac/lazada_descn/`.
- **Loaded locally:** **yes** — `full_testset.csv` 181,669 rows (**randomized**),
  `full_trainset.csv` 926,669 rows (**biased by prior policy**); 86 cols.
- **Treatment / target:** `is_treat` (voucher), `label` (conversion); `data_id` is an id.
- **Feature families:** `f0..f82` anonymized.
- **Timing evidence:** none.
- **A:** anon `f0..f82`. **B/C:** none. **D:** the biased train set is a propensity/D demo.
- **Full A+B+C possible:** no.
- **Recommendation:** **A-only** on the randomized test set; nice extra real dataset and a
  good biased-train/propensity demonstration. License: see repo (research/non-commercial).

### 7. CUVET-policy (discovery)

- **Source:** HuggingFace `anonadata/CUVET-policy` (CUVET paper, Criteo).
- **Access method / fetch:** public parquet on HF (Dataset Viewer / `datasets`); 13.7 GB total.
- **Loaded locally:** **partial** — schema read from the HF Dataset Viewer; full ~86.7M-row /
  13.7 GB parquet **not** fully downloaded (size). Marked **schema from HF viewer**.
- **Treatment / target:** `policy` (continuous, 5 arms {0.8,0.9,1,1.1,1.2}); `value`, `cost`.
- **Feature families:** `f0..f4` anonymized (user features + noise + standardization), plus
  `is_val` split flag. Randomized 2-week A/B test.
- **Timing evidence:** features said to be collected at the bidding moment, but anonymization
  removes semantics → no usable in-time signal.
- **A:** anon `f0..f4`. **B/C:** none. **D:** continuous-treatment policy / cost constraint.
- **Full A+B+C possible:** no.
- **Recommendation:** **A-only**; valuable as a large continuous-treatment / policy benchmark.
  License **CC-BY-NC-SA 4.0** (non-commercial) — note before any reuse.

### 8. MT-LIFT (discovery)

- **Source:** GitHub `MTDJDSP/MT-LIFT` (*Entire Chain Uplift Modeling*, 2024).
- **Access method:** data hosted on **Google Drive / Baidu Drive** (not in repo).
- **Loaded locally:** **no** — access not automatable here; access request pending externally.
- **Treatment / target:** `treatment ∈ [0,4]` (multi-treatment coupons, RCT); `click`,
  `conversion` (funnel).
- **Feature families:** `f0..f98` anonymized/desensitized; ~5.54M rows; snapshot (no timestamps).
- **Timing evidence:** none in features. The **click→conversion funnel** is the interesting part.
- **A:** anon `f0..f98`. **B/C:** none. **E:** `click` is a textbook **mediator** between
  treatment and `conversion`.
- **Full A+B+C possible:** no (and access-blocked).
- **Recommendation:** **access_blocked**; if obtained, useful as a multi-treatment A-only set
  and a clean **mediator/E** demonstration — not B/C.

### 9. Dunnhumby — The Complete Journey (discovery)

- **Source:** <https://www.dunnhumby.com/source-files/> (also community R/Kaggle mirrors).
- **Access method:** **registration form** required on the official site.
- **Loaded locally:** **no** — form-gated.
- **Treatment / target:** coupon exposure / redemption; **observational — coupon assignment is
  not randomized** (no clean RCT).
- **Feature families:** multi-table — 2 years of household transactions, campaign tables, and
  **direct-marketing contact history**, all with dates.
- **Timing evidence:** **the richest of any candidate** — real timestamps for transactions,
  campaigns, and contacts allow constructing pre- and in-experiment features.
- **A/C/D/E/F:** timestamps enable C/D *construction*; coupon/contact columns are D/E; post-coupon
  behavior is E/F. But without randomization none of this validates causal correctness.
- **Full A+B+C possible:** no — not an RCT, and access-gated.
- **Recommendation:** best dataset for *constructing* in-time C/D features, but **observational →
  demonstration/semi only**, and access-blocked for automated download. Do not treat as RCT.

### 10. Rejected and semi-synthetic references

- **Kaggle "Orange Telecom Churn" (`mnassrib/telecom-churn`):** ordinary churn-prediction
  dataset with **no treatment/control** → **rejected**. (The spreadsheet conflated it with the
  Verhelst Orange Belgium uplift set in §1; only the latter is a real uplift dataset.)
- **Criteo-ITE (`criteo-research/large-scale-ITE`):** semi-synthetic potential outcomes on top
  of real Criteo features → CATE/ITE benchmark, **not** real treatment → not_suitable for B/C.
- **IHDP, Twins, ACIC 2016/2018/2019/2022:** semi-synthetic causal benchmarks with simulated
  treatment/DGP and real covariates. No in-time B/C. **Reserve for a future DAG/causal (D)
  validation stage**, not for B/C uplift. ACIC 2018 is Synapse-registration-gated.
- **Lalonde / NSW:** classic; the experimental NSW subset is a real RCT but tiny, earnings
  outcome, no in-time covariates → not useful for B/C.

### 6′. Hillstrom (already implemented — short note)

Hillstrom (MineThatData e-mail) is a randomized e-mail campaign already wired via
`HillstromAdapter` (see `docs/06_datasets_setup.md`). It has named pre-treatment customer
features (A) and post-treatment `visit`/`conversion` (E); **no in-time B/C**. It validates the
**pipeline / A-only** path, not B/C.

---

## Reproduction

```bash
# scikit-uplift is a LOCAL audit-only tool, never a project dependency:
.venv/bin/pip install scikit-uplift
.venv/bin/python tools/audit_datasets.py                      # all sklift datasets
.venv/bin/python tools/audit_datasets.py lenta megafon        # selected
# Orange Belgium / Lazada-DESCN are pulled in audit_datasets.py via direct CSV/zip URLs.
```

All downloads land in the **gitignored** `data/06_safe_intime_cupac/<dataset>/`. Schema text
summaries (`schema_summary.txt`) are written there only; no raw data enters git.
