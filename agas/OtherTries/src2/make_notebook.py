"""Convert a percent-format python source (jupytext style) into a .ipynb notebook."""
import sys
import nbformat as nbf


def convert(src_path, out_path):
    with open(src_path) as f:
        lines = f.read().split("\n")

    cells, cur, kind = [], [], None
    for line in lines:
        if line.startswith("# %%"):
            if kind:
                cells.append((kind, "\n".join(cur).strip("\n")))
            opts = line[3:].strip()
            kind = "markdown" if "markdown" in opts else "code"
            cur = []
        else:
            cur.append(line)
    if kind:
        cells.append((kind, "\n".join(cur).strip("\n")))

    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, src in cells:
        if not src.strip():
            continue
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l
                             for l in src.split("\n"))
            nb.cells.append(nbf.v4.new_markdown_cell(text))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    nbf.write(nb, out_path)
    print(f"{out_path}: {len(nb.cells)} cells "
          f"({sum(c.cell_type == 'code' for c in nb.cells)} code, "
          f"{sum(c.cell_type == 'markdown' for c in nb.cells)} markdown)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])