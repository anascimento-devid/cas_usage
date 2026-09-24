from pathlib import Path

path = Path(r"C:\Users\seugn\PycharmProjects\cas_usage\notebooks\notebook.ipynb")

content = path.read_text(encoding="utf-8-sig")
path.write_text(content, encoding="utf-8")