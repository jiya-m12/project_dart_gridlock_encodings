"""Loader for the disclosed pre-window notebook CSVs (used only to cross-check numbers)."""
from .instances import Instance


def load_nb_csv(path, n_banks):
    arcs, liq, sec = [], [], None
    for line in open(path):
        line = line.strip()
        if 'PAYMENTS' in line:
            sec = 'p'; continue
        if 'INSTITUTIONS' in line:
            sec = 'i'; continue
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if sec == 'p':
            arcs.append((int(parts[1]), int(parts[2]), int(parts[3])))
        elif sec == 'i':
            liq.append(int(parts[1]))
    return Instance(path.split('/')[-1].replace('.csv', ''), n_banks, arcs, liq)
