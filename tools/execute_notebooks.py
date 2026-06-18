"""Пакетное выполнение целевых RnD-ноутбуков."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

# Корень с отчётными RnD-директориями. Ноутбуки находятся автоматически: и
# исследовательский `rnd/<topic>/notebook.ipynb`, и передаваемый во внутренний контур
# `rnd/<topic>/notebook_internal.ipynb` подхватываются без правки этого списка. Internal
# прогоняется здесь же, в репозитории, — это глобальное правило проверки запускаемости.
RND_ROOT = Path("rnd")


def discover_notebooks() -> list[Path]:
    return sorted(RND_ROOT.glob("*/notebook*.ipynb"))


def _capture_figures() -> list[dict]:
    """Снять все открытые matplotlib-фигуры как display_data image/png и закрыть их.

    Возвращает пустой список, если matplotlib не использовался в ячейке. Так ноутбуки
    хранят графики inline (видны при просмотре .ipynb), а не только в report.pdf.
    """
    if "matplotlib" not in sys.modules:
        return []
    import matplotlib

    matplotlib.use("Agg", force=True)  # неинтерактивный рендер для пакетного прогона
    import matplotlib.pyplot as plt

    outputs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
        png = base64.b64encode(buf.getvalue()).decode("ascii")
        outputs.append(
            {"output_type": "display_data", "data": {"image/png": png}, "metadata": {}}
        )
        plt.close(fig)
    return outputs


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {}
    counter = 1

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        code = "".join(cell.get("source", []))
        stdout = io.StringIO()
        outputs = []

        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, filename=str(path), mode="exec"), ns, ns)

            text = stdout.getvalue()
            if text:
                outputs.append({"name": "stdout", "output_type": "stream", "text": text})
            outputs.extend(_capture_figures())

            cell["execution_count"] = counter
            cell["outputs"] = outputs
        except Exception:
            cell["execution_count"] = counter
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": "ExecutionError",
                    "evalue": "Cell execution failed",
                    "traceback": traceback.format_exc().splitlines(),
                }
            ]
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            raise

        counter += 1

    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    # Точечный прогон: пути к конкретным ноутбукам через argv; иначе — автодискавери.
    args = [Path(a) for a in sys.argv[1:]]
    notebooks = args or discover_notebooks()
    if not notebooks:
        raise FileNotFoundError(f"Не найдены notebook*.ipynb в {RND_ROOT}/*/")
    for nb in notebooks:
        execute_notebook(nb)
        print(f"Executed: {nb}")


if __name__ == "__main__":
    main()
