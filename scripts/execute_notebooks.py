"""Execute all notebooks in repository without external dependencies."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


def execute_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    ns: dict = {}
    counter = 1

    for cell in nb.get("cells", []):
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
                outputs.append(
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": text,
                    }
                )

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
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
            raise

        counter += 1

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def main() -> None:
    notebooks = sorted(Path('.').glob('*/*.ipynb'))
    if not notebooks:
        print('No notebooks found')
        return
    for nb in notebooks:
        execute_notebook(nb)
        print(f'Executed: {nb}')


if __name__ == '__main__':
    main()
