"""Графика single-table сравнения методов (RnD-8): эффекты с ДИ и p-values.

Витрина поверх таблицы :class:`~rnd_reports.multiple_testing.pipeline.ComparisonResult`:

- левая панель — forest-плот эффекта по каждой метрике с доверительным интервалом
  (``ci_low``/``ci_high``, если посчитаны), точки окрашены по решению reference-метода;
- правая панель — точечный график raw и adjusted p-value по всем методам в лог-шкале
  с порогом ``alpha``: видно, как коррекция «толкает» p-value вправо к незначимости.

Matplotlib импортируется лениво внутри функции — пакет не тянет графику при обычном
импорте (и не требует дисплея).
"""

from __future__ import annotations

import numpy as np

from .methods import METHOD_NAMES, p_adj_columns, reject_columns

# Палитра методов — единообразно с порядком METHOD_NAMES в таблицах/графиках.
_METHOD_COLORS = {
    "bonferroni": "#1f77b4",
    "holm": "#2ca02c",
    "westfall_young": "#9467bd",
    "romano_wolf": "#8c564b",
    "benjamini_hochberg": "#ff7f0e",
}


def plot_pvalue_comparison(result, alpha: float = 0.05, reference_method: str = "holm"):
    """Двухпанельная визуализация single-table сравнения; возвращает matplotlib Figure.

    Параметры:
        result: :class:`ComparisonResult` из :func:`run_comparison`;
        alpha: порог значимости (вертикальная линия на правой панели);
        reference_method: метод, по решению которого красятся точки эффектов слева.
    """
    import matplotlib.pyplot as plt

    table = result.table
    order = table["effect"].sort_values(ascending=True).index  # снизу вверх по величине эффекта
    table = table.loc[order].reset_index(drop=True)
    targets = table["target"].tolist()
    y = np.arange(len(targets))

    has_ci = {"ci_low", "ci_high"}.issubset(table.columns)
    reject_col = reject_columns().get(reference_method)
    rejected = (table[reject_col].to_numpy(dtype=bool)
                if reject_col in table.columns else np.zeros(len(table), dtype=bool))

    height = max(3.0, 0.42 * len(targets) + 1.5)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, height), sharey=True)

    # --- Левая панель: эффект ± ДИ -----------------------------------------------
    if has_ci:
        xerr = np.vstack([table["effect"] - table["ci_low"],
                          table["ci_high"] - table["effect"]])
        ax_l.errorbar(table["effect"], y, xerr=xerr, fmt="none", ecolor="0.6",
                      elinewidth=1.3, capsize=2, zorder=1)
    for sig, color, label in [(True, "#d62728", f"отвергнута ({reference_method})"),
                              (False, "#1f77b4", "не отвергнута")]:
        mask = rejected == sig
        if mask.any():
            ax_l.scatter(table["effect"][mask], y[mask], color=color, s=34, zorder=2, label=label)
    ax_l.axvline(0.0, color="0.3", ls="--", lw=1)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(targets, fontsize=8)
    ax_l.set_xlabel("Эффект (treatment − control)")
    ci_txt = f" ± {round((1 - alpha) * 100)}% ДИ" if has_ci else ""
    ax_l.set_title(f"Эффект{ci_txt} по метрикам")
    ax_l.legend(fontsize=8, loc="best")
    ax_l.grid(axis="x", ls=":", alpha=0.5)

    # --- Правая панель: raw и adjusted p-value -----------------------------------
    p_cols = p_adj_columns()
    ax_r.scatter(table["p_value"].clip(lower=1e-300), y, marker="|", s=130,
                 color="0.15", label="raw p", zorder=3)
    for name in METHOD_NAMES:
        ax_r.scatter(table[p_cols[name]].clip(lower=1e-300), y, s=26,
                     color=_METHOD_COLORS.get(name), label=name, alpha=0.85, zorder=2)
    ax_r.axvline(alpha, color="#d62728", ls="--", lw=1.2, label=f"α = {alpha:g}")
    ax_r.set_xscale("log")
    ax_r.set_xlabel("p-value (лог-шкала)")
    ax_r.set_title("raw и adjusted p-value по методам")
    ax_r.legend(fontsize=7, loc="best", ncol=2)
    ax_r.grid(axis="x", ls=":", alpha=0.5)

    fig.tight_layout()
    return fig
