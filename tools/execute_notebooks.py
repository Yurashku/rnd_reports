"""Выполнение всех ноутбуков в репозитории без внешних зависимостей."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


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
    notebooks = sorted(Path(".").glob("rnd_*/**/*.ipynb"))
    if not notebooks:
        print("Ноутбуки не найдены")
        return

    for nb in notebooks:
        execute_notebook(nb)
        print(f"Executed: {nb}")


if __name__ == "__main__":
    main()
