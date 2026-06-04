# R&D-6: Expanded public dataset scouting and delta effects

- **Версия:** v1.0
- **Связано:** `docs/06_safe_intime_cupac_context.md`,
  `docs/06_real_dataset_raw_audit.md` (предыдущий raw-аудит),
  `src/rnd_reports/datasets/inspect.py`, `src/rnd_reports/datasets/expanded_adapters.py`,
  `tools/audit_datasets.py`, `rnd/06_safe_intime_cupac/expanded_dataset_delta_results.csv`.
- **Scope:** более широкий, чем raw-аудит, поиск + реальная загрузка/инспекция + прототипы
  адаптеров + фактические delta-эффект бенчмарки. Главный фокус — найти представительность
  **C** (затем **D**), включая не-A/B «песочницы» (event-log / bandit / journey / advertising).

> **Напоминание про классы.** A = pre-treatment; B = expert-safe in-time; C = balance-gated
> in-time; D = DAG-required; E = mediator-risk; F = leakage/outcome-derived; `unsafe_demo` =
> демонстрация. Реальные **B/C** требуют *именованных, семантически интерпретируемых in-time
> колонок с защитимым timing-evidence*. Анонимные снимки (`f*`/`PC*`/`X_*`) и пре-период
> агрегаты как B/C **не** годятся. Сконструированные in-time признаки в «песочницах» — это
> **sandbox-кандидаты C**, а не реальная валидация.

---

## 1. Executive summary

Реальная загрузка (sandbox-сеть включалась только для скачивания, данные — в gitignored
`data/06_safe_intime_cupac/`, в git не коммитятся) и прогон цепочки методов дали следующее.

**Счётчики представительности (по факту схем, не по документации):**

- **A (pre-treatment):** дают практически все кандидаты — **10+** датасетов (Hillstrom,
  Lenta, X5, Orange Belgium, Lazada/DESCN, Criteo, MegaFon*, Open Bandit, dunnhumby,
  CriteoPrivateAd; *MegaFon синтетический). Реальных «A-носителей» — **9**.
- **B (expert-safe in-time):** **0** датасетов. Ни один публичный набор не выкладывает
  именованные in-time экспертно-безопасные колонки.
- **C-кандидаты (реальные, защитимые):** **0**. C-конструкция (sandbox) теоретически
  возможна на **2** event-log датасетах (Open Bandit — лаги истории пользователя строго до
  timestamp; dunnhumby — in-window активность до cutoff), но это **не** реальная валидация.
- **D-кандидаты:** **5** (Criteo `exposure`; Lazada biased train propensity; Open Bandit
  `action`/`position`/`pscore`; CriteoPrivateAd display/privacy-механика; dunnhumby
  campaign/coupon/contact).
- **E/F/unsafe-кандидаты:** **7+** (Hillstrom `visit`/`conversion`; Criteo `exposure`;
  X5 пост-коммуникационные покупки; MT-LIFT `click`-funnel; Open Bandit reward-производные;
  CriteoPrivateAd delayed click/sale/report; dunnhumby redemption/post-coupon).

**Строгие uplift / A-валидация:** Hillstrom, Lenta, X5 RetailHero, Orange Belgium,
Lazada/DESCN (рандомизированный test), Criteo — пригодны как **A-only**.

**C/D-«песочницы» (не A/B):** Open Bandit (logged bandit), dunnhumby Complete Journey
(observational retail journey), CriteoPrivateAd (advertising logs). Полезны для *исследования*
C/D-механики, но не валидируют causal-корректность safe in-time.

**Отклонённые:** Kaggle «Orange Telecom Churn» (`mnassrib`) — нет treatment/control;
Criteo-ITE / IHDP / Twins / ACIC — semi-synthetic (симулированное лечение); MT-LIFT —
access-blocked (Google/Baidu drive), при доступе всё равно A-only + E.

**Доступна ли полная реальная A+B+C-валидация?** **Нет, не подтверждена.** Реальных
safe in-time B/C-колонок в публичных датасетах нет. Полную цепочку `A → A+B → A+B+C`
по-прежнему валидирует только синтетика (классы известны по построению).

**Рекомендованный следующий путь:** (1) довести production-grade A-only адаптеры Lenta/X5
(самые сильные именованные/мульти-табличные реальные наборы); (2) использовать C/D-«песочницы»
(Open Bandit, dunnhumby) как **методический** полигон для конструирования и диагностики
in-time C/D — но с явной пометкой sandbox; (3) реальную B/C-валидацию переносить на
внутренние корпоративные данные (см. VarWar), т.к. публичные датасеты её не дают.

---

## 2. Representativeness table

`loaded_locally`: **yes** = реально загружен и проинспектирован в этой итерации;
`docs_only_not_locally_confirmed` / `access_blocked` — иначе.

| dataset | source_url | loaded_locally | dataset_type | has_treatment_or_intervention | has_target | has_timestamps | has_event_logs | has_A | has_B | has_C_candidate | has_D_candidate | has_E_F_unsafe_candidate | validation_level | suitability_for_delta_effects | recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hillstrom | blog.minethatdata.com | yes | real_rct | yes (email) | yes (spend) | no | no | yes | no | no | no | yes (visit/conversion) | A_only_real | ab + cupac_A | pipeline/A-only smoke (уже подключён) |
| Lenta | uplift-modeling.com/…/fetch_lenta | yes | real_uplift | yes (group) | yes (response_att) | no | no | yes (named) | no | no | no | no | A_only_real | ab + cupac_A | лучший именованный A-only; первый prod-адаптер |
| Criteo Uplift v2.1 | arxiv 2111.10106 | see §3 | real_rct | yes (treatment) | yes (visit) | no | no | yes (anon f0..f11) | no | no | yes (exposure) | yes (exposure) | A_only_real | ab + cupac_A | слабый A-only; exposure только как E/unsafe |
| X5 RetailHero | ods.ai/…/x5-retailhero | yes | real_uplift_multitable | yes (treatment_flg) | yes (target) | yes (purchases) | yes | yes (engineered) | no | no(*camp.date hidden) | no | yes (post-purchase) | A_only_real | ab + cupac_A | сильнейший мульти-табличный A-only; будущий C только при дате кампании |
| MegaFon | uplift-modeling.com/…/fetch_megafon | yes | synthetic | yes (treatment_group) | yes (conversion) | no | no | yes (anon synth) | no | no | no | no | not_suitable | scale/sanity only | синтетика — только scale-бенчмарк |
| Orange Belgium | arxiv 2312.07206 | yes | real_rct | yes (t) | yes (y churn) | no | no | yes (PCA anon) | no | no | no | no | A_only_real | ab + cupac_A | A-only; наименее интерпретируемые признаки |
| Lazada / DESCN | github.com/kailiang-zhong/DESCN | yes | real_uplift | yes (is_treat) | yes (label) | no | no | yes (anon f0..f82) | no | no | yes (biased train) | no | A_only_real | ab + cupac_A | A-only на random test; biased train = D/propensity demo |
| Open Bandit | github.com/st-tech/zr-obp | see §3 | research_sandbox (logged bandit) | logged action/position | yes (reward) | yes (rounds) | yes | yes (context) | no | sandbox-construct | yes (action/position/pscore) | yes (reward-derived) | D_F_sandbox | ab + cupac_A (демо) | C/D-полигон; не реальная A/B-валидация |
| dunnhumby Complete Journey | dunnhumby.com/source-files | access_blocked | event_log_sandbox (observational) | coupon (не random) | redemption/spend | yes (rich) | yes | yes (engineered) | no | sandbox-construct | yes (campaign/coupon/contact) | yes (redemption/post-coupon) | C_D_sandbox | — (нет clean RCT) | богатейший timing; observational → demo-only |
| CriteoPrivateAd | huggingface.co/datasets/criteo/CriteoPrivateAd | see §3 | research_sandbox (ad logs) | display/request | click/sale (delayed) | partial | yes | yes (pre-display) | no | no | yes (display/privacy) | yes (delayed report) | D_F_sandbox | — (нет A/B-контракта) | D/F-полигон; не safe B/C |
| CUVET-policy | huggingface.co/…/CUVET-policy | docs_only_not_locally_confirmed | real (continuous policy) | yes (policy 5 arms) | yes (value/cost) | no | no | yes (anon f0..f4) | no | no | yes (cost/policy) | no | A_only_real | (heavy 13.7GB) | A-only; large continuous-treatment; CC-BY-NC-SA |
| MT-LIFT | github.com/MTDJDSP/MT-LIFT | access_blocked | real_uplift (multi-treatment) | yes (treatment 0–4) | click/conversion | no | no | yes (anon f0..f98) | no | no | no | yes (click funnel=E) | access_blocked | — | при доступе: A-only + mediator/E demo |

(*) X5: реальные `transaction_datetime` есть, но **дата кампании не опубликована** → защитимый
in-time C построить нельзя.

---

## 3. Dataset sections

### Hillstrom (MineThatData)
- **Source / access:** blog.minethatdata.com; CSV (mirror) → `data/.../hillstrom/`; адаптер
  `HillstromAdapter` (Step 8).
- **Local load:** yes (через loader; используется и в delta).
- **Schema:** named pre-treatment (`recency, history, mens, womens, newbie` + one-hot
  `history_segment, zip_code, channel`); `spend` (target), `visit`/`conversion` (post).
- **Treatment / target:** `segment != 'No E-Mail'` / `spend`.
- **A:** клиентские пре-рассылочные признаки. **B/C:** нет. **D:** нет. **E/F:**
  `visit`/`conversion` (пост-treatment поведение).
- **Timing:** рандомизированный RCT; in-time колонок нет.
- **Adapt → BenchmarkDataset:** yes (готов). **Delta:** ab + `sklearn_cupac_A`.
- **Recommendation:** pipeline/A-only smoke; не B/C.

### Lenta
- **Source / access:** `sklift.datasets.fetch_lenta` → `lenta/lenta_dataset.csv.gz` (687 029×195).
- **Local load:** yes. Адаптер `load_lenta_benchmark_dataset`.
- **Schema:** `group` (test/control), `response_att` (target); именованные пре-кампейн
  агрегаты 15d/1m/3m/6m/12m (`cheque_count_*`, `sale_sum_*`, `k_var_*`, …) + `age`, `gender`,
  `children`, `main_format`, `response_sms`, `response_viber`.
- **A:** весь именованный числовой набор (+ one-hot gender). **B:** нет. **C:** нет — окна
  пре-период, не in-time (даже 15d — история до кампании). **D/E/F:** нет явных.
- **Timing:** агрегаты заканчиваются до кампании → pre-treatment.
- **Adapt → BenchmarkDataset:** yes. **Delta:** ab + `sklearn_cupac_A`.
- **Recommendation:** лучший реальный **A-only** с именованными признаками; первый prod-адаптер.

### Criteo Uplift v2.1
- **Source / access:** `sklift.datasets.fetch_criteo(percent10=True)`. На момент raw-аудита
  S3 отдавал HTTP 403; в этой итерации повторная попытка загрузки — см. delta SKIPPED/ok.
- **Local load:** см. лог delta (если 403 — `docs_only_not_locally_confirmed`).
- **Schema:** `treatment`; `visit`/`conversion`; `f0..f11` анонимны; `exposure` (RTB-показ).
- **A:** `f0..f11` (слабые, анонимные). **B/C:** нет. **D:** `exposure` (post-treatment
  механика показа). **E/F/unsafe:** `exposure` — классический mediator/unsafe_demo.
- **Adapt → BenchmarkDataset:** yes (`load_criteo_percent10_benchmark_dataset`); `exposure`
  размечается `E_mediator_risk`. **Delta:** ab + `sklearn_cupac_A` (если загрузился).
- **Recommendation:** слабый A-only; `exposure` только как E/unsafe-демонстрация.

### X5 RetailHero
- **Source / access:** `sklift.datasets.fetch_x5` → `x5/{uplift_train,clients,purchases}.csv.gz`.
- **Local load:** yes (purchases 45.8M строк). Адаптер `load_x5_a_only_benchmark_dataset`
  (чанковая агрегация истории по client_id).
- **Schema:** `uplift_train`(client_id, treatment_flg, target); `clients`(age, gender,
  first_issue/redeem_date); `purchases`(real `transaction_datetime`, purchase_sum, points, …).
- **A:** инженерные пер-клиентские агрегаты истории (count/sum/mean/points/recency) + age +
  one-hot gender. **B:** нет. **C:** **нет защитимого** — дата кампании не опубликована, нельзя
  отделить pre- от in-treatment покупок. **E/F:** покупки после (неизвестной) коммуникации.
- **Timing:** реальные datetime в логе покупок — единственный полностью скачиваемый набор с
  настоящим event-timing, но без даты кампании.
- **Adapt → BenchmarkDataset:** yes (A-only). **Delta:** ab + `sklearn_cupac_A`.
- **Recommendation:** сильнейший реальный мульти-табличный A-only; единственный кандидат на
  будущий C *при* публикации окна кампании.

### MegaFon
- **Source / access:** `sklift.datasets.fetch_megafon` → `megafon/` (600 000×52).
- **Local load:** yes. **Schema:** `treatment_group`, `conversion`, `X_1..X_50` анонимны.
- **A/B/C/D/E/F:** синтетические анонимные признаки; сигнала классов безопасности нет.
- **Recommendation:** `not_suitable` для реальной валидации; только scale/robustness sanity.

### Orange Belgium churn uplift
- **Source / access:** arXiv 2312.07206; Dropbox CSV / OpenML 45580 → `orange_belgium/`.
- **Local load:** yes (11 896×180). Адаптер `load_orange_belgium_benchmark_dataset`.
- **Schema:** `t`, `y` (churn); `PC1..PC160` (PCA float), `FACTOR1..FACTOR18` (anon кат. → one-hot).
- **A:** PCA + one-hot FACTOR. **B/C/D/E/F:** нет (PCA уничтожает семантику).
- **Adapt → BenchmarkDataset:** yes. **Delta:** ab + `sklearn_cupac_A`.
- **Recommendation:** A-only; самая низкая интерпретируемость признаков среди A-группы.

### Lazada / DESCN
- **Source / access:** github `kailiang-zhong/DESCN`; Dropbox zip → `lazada_descn/`.
- **Local load:** yes (random `full_testset.csv` 181 669×86). Адаптер
  `load_lazada_descn_benchmark_dataset`.
- **Schema:** `is_treat` (voucher), `label`; `f0..f82` анонимны.
- **A:** `f0..f82`. **B/C:** нет. **D:** biased `full_trainset` (prior policy) — propensity/D demo.
- **Adapt → BenchmarkDataset:** yes (random test). **Delta:** ab + `sklearn_cupac_A`.
- **Recommendation:** A-only на random-test; biased train = хороший propensity/D-демонстратор.

### Open Bandit Dataset (C/D-sandbox)
- **Source / access:** github `st-tech/zr-obp`; `pip install obp` (в .venv);
  `OpenBanditDataset(behavior_policy, campaign)`.
- **Local load:** см. delta-лог (bundled-сэмпл/полный релиз).
- **Schema:** logged bandit feedback — `context` (user/item), `action`, `position`, `reward`
  (click), `pscore` (propensity).
- **Mapping:** **A** = context (до действия); **D** = `action`/`position`/`pscore` (механика
  политики, нужен DAG); **C (sandbox-construct)** = лаги истории пользователя строго до
  timestamp (если линкуются); **F** = reward-производные.
- **Treatment (demo):** бинаризация показа топ-позиции — **не** настоящий A/B.
- **Adapt → ResearchDataset** (`load_open_bandit_research_dataset`), `dataset_type=event_log_sandbox`.
  **Delta:** ab + `sklearn_cupac_A` на context (демо); D-признаки исключены политикой.
- **Recommendation:** методический C/D-полигон; **не** реальная A+B+C-валидация.

### dunnhumby — The Complete Journey (C/D-sandbox)
- **Source / access:** dunnhumby.com source-files (**регистрация**) / community-зеркала.
- **Local load:** access_blocked (форма). Адаптер-прототип
  `load_completejourney_research_dataset` (требует распакованных таблиц; иначе понятная ошибка).
- **Schema (по докам):** transactions / products / households / campaigns / coupons /
  coupon_redempt / campaign_table — с реальными датами.
- **Mapping (анкер = дата первого контакта кампании):** **A** = пред-анкерная история покупок;
  **C (sandbox-construct)** = in-window активность до cutoff исхода; **D/E** =
  campaign/coupon/contact; **E/F** = post-anchor покупки/redemption.
- **Timing:** богатейший event-timing среди кандидатов, **но** observational (купоны не random).
- **Adapt:** ResearchDataset (event_log_sandbox), demo-only. **Delta:** нет (нет clean RCT).
- **Recommendation:** лучший набор для *конструирования* in-time C/D, но observational →
  demonstration/semi only; не RCT.

### CriteoPrivateAd (D/F-sandbox)
- **Source / access:** HuggingFace `criteo/CriteoPrivateAd`; `pip install datasets` (в .venv);
  многогигабайтный parquet.
- **Local load:** см. delta/инспекцию (по размеру может быть partial).
- **Schema (по докам/паркету):** anonymized advertising logs; display/request контекст,
  campaign/privacy-механика, delayed click/sale/report-агрегаты.
- **Mapping:** **A** = pre-display контекст/request; **D** = display/order/campaign/privacy;
  **F** = delayed click/sale/report-производные; защитимый in-time C — обычно нет.
- **Adapt:** ResearchDataset (research_sandbox), demo-only. **Delta:** нет A/B-контракта.
- **Recommendation:** D/F-полигон; не safe B/C. License CC-BY-NC-SA (non-commercial).

---

## 4. Rejected candidates

| name / link | reason | treatment/control missing | timing missing | download/access failed |
|---|---|---|---|---|
| Kaggle «Orange Telecom Churn» (`mnassrib/telecom-churn`) | обычная churn-prediction, не uplift | **yes** | yes | no |
| Criteo-ITE (`criteo-research/large-scale-ITE`) | semi-synthetic потенциальные исходы | simulated | yes | no |
| IHDP / Twins / ACIC 2016–2022 | semi-synthetic DGP, симулированное лечение | simulated | yes | partly (ACIC 2018 Synapse-gate) |
| Lalonde / NSW | реальный RCT, но крошечный, без in-time ковариат | no | yes | no |
| MT-LIFT (`MTDJDSP/MT-LIFT`) | данные на Google/Baidu drive | no | yes | **yes (access_blocked)** |
| dunnhumby Complete Journey | observational + регистрация | not random | no (timing богат) | **yes (registration)** |

Подробный пятичастный разбор и исходные команды загрузки — в
`docs/06_real_dataset_raw_audit.md`.

---

## 5. Delta-effect results

Числа генерируются `tools/audit_datasets.py --delta` и сохраняются в
`rnd/06_safe_intime_cupac/expanded_dataset_delta_results.csv` (компактная сводка, не сырьё).
Фигуры — `rnd/06_safe_intime_cupac/figures/`. Сводная таблица и интерпретация продублированы в
`rnd/06_safe_intime_cupac/report.md` (раздел «Expanded public dataset validation»).

На всех реальных датасетах валиден только путь **A-only** (`ab_hypex` →
`sklearn_cupac_A`): CUPAC по классу A снижает дисперсию без смещения ATE. Реальной
A+C/A+B+C-строки получить не удалось — защитимых C/B-колонок в публичных данных нет. C/D
исследуются только на «песочницах» (Open Bandit) с явной пометкой sandbox.

> **Вывод.** Полная A+B+C-валидация на публичных реальных датасетах не подтверждена.
> Однако C/D-представительность можно исследовать на event-log / bandit / advertising /
> journey datasets, которые не являются прямыми A/B uplift benchmark.
