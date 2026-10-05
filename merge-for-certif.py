"""Fusionne le notebook principal et le journal de bord pour le rendu certif.

Cherche le titre « D. Journal de bord » dans le notebook principal et insère après lui
toutes les cellules du journal (en sautant les éventuelles cellules d'intro). Produit un
nouveau notebook ; ne modifie pas les sources.

Usage :
    python scripts/merge_for_certif.py <main.ipynb> <journal.ipynb> <output.ipynb>

Le script utilise uniquement la stdlib (pas de nbformat). Si une dépendance manque côté
apprenant, c'est volontaire pour rester portable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANNEXE_D_TITLE_FRAGMENT = "D. Journal de bord"
JOURNAL_INTRO_FRAGMENTS = (
    "Copier-coller les cellules",
    "# Journal de bord",
)


def _cell_source_str(cell: dict) -> str:
    """Extrait la source d'une cellule sous forme de string.

    Args:
        cell: cellule Jupyter au format JSON.

    Returns:
        Le contenu textuel concaténé.
    """
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def _is_journal_intro(cell: dict) -> bool:
    """Détecte une cellule d'intro de journal à filtrer.

    Args:
        cell: cellule Jupyter.

    Returns:
        True si la cellule est juste l'intro à omettre.
    """
    if cell.get("cell_type") != "markdown":
        return False
    src = _cell_source_str(cell)
    return any(frag in src for frag in JOURNAL_INTRO_FRAGMENTS)


def _find_annexe_d(cells: list[dict]) -> int:
    """Trouve l'index de la cellule « D. Journal de bord » dans le notebook principal.

    Args:
        cells: liste des cellules du notebook.

    Returns:
        Index de la cellule, ou -1 si introuvable.
    """
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown" and ANNEXE_D_TITLE_FRAGMENT in _cell_source_str(c):
            return i
    return -1


def merge(main_path: Path, journal_path: Path, output_path: Path) -> int:
    """Fusionne le notebook principal et le journal de bord.

    Args:
        main_path: notebook principal (avec annexe D vide ou minimale).
        journal_path: journal-de-bord.ipynb.
        output_path: notebook fusionné en sortie.

    Returns:
        0 si succès, 1 si annexe D introuvable.
    """
    nb_main = json.loads(main_path.read_text(encoding="utf-8"))
    nb_journal = json.loads(journal_path.read_text(encoding="utf-8"))

    cells_main: list[dict] = nb_main["cells"]
    cells_journal: list[dict] = nb_journal["cells"]

    annexe_idx = _find_annexe_d(cells_main)
    if annexe_idx < 0:
        print(
            f"❌ Titre « {ANNEXE_D_TITLE_FRAGMENT} » introuvable dans {main_path.name}.",
            file=sys.stderr,
        )
        return 1

    cells_to_insert = [c for c in cells_journal if not _is_journal_intro(c)]

    nb_main["cells"] = (
        cells_main[: annexe_idx + 1] + cells_to_insert + cells_main[annexe_idx + 1 :]
    )

    output_path.write_text(
        json.dumps(nb_main, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"✅ Fusion : {output_path}\n"
        f"   {len(cells_to_insert)} cellule(s) du journal insérée(s) après l'annexe D."
    )
    return 0


def main() -> int:
    """Point d'entrée CLI.

    Returns:
        Code retour CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path, help="Notebook principal (.ipynb)")
    parser.add_argument("journal", type=Path, help="Journal de bord (.ipynb)")
    parser.add_argument("output", type=Path, help="Notebook fusionné de sortie (.ipynb)")
    args = parser.parse_args()

    for p in (args.main, args.journal):
        if not p.is_file():
            print(f"❌ Fichier introuvable : {p}", file=sys.stderr)
            return 2

    return merge(args.main, args.journal, args.output)


if __name__ == "__main__":
    sys.exit(main())