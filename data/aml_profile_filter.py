"""Run LOCALLY on the IBM AML file (Altman et al., NeurIPS 2023 D&B). Standard library only.

    python3 aml_profile_filter.py HI-Small_Trans.csv
    python3 aml_profile_filter.py          (or the IDE Run button: finds the AML .csv
                                            in this script's folder or the current folder)

Writes next to the input:
  aml_profile.json              counts + quantiles, small, safe to share
  aml_usd_interbank.csv.gz      rows with From Bank != To Bank, both currencies
                                "US Dollar", format Wire or ACH (laundering rows
                                kept, flagged, so the exclusion is done later and
                                is visible)
Nothing is decided here except that filter; batch rules come later, pre-registered.
"""
import csv, gzip, hashlib, json, os, sys
from collections import Counter

KEEP_FORMATS = {"Wire", "ACH"}
CUR = "US Dollar"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def quantiles(xs, qs=(0, .01, .1, .25, .5, .75, .9, .99, 1)):
    if not xs:
        return {}
    xs = sorted(xs)
    return {str(q): xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] for q in qs}


def find_input():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.getcwd()):
        c = [os.path.join(d, x) for x in os.listdir(d) if x.lower().endswith(".csv")]
        if c:   # prefer a name containing "trans", else the largest CSV
            c.sort(key=lambda q: ("trans" not in os.path.basename(q).lower(), -os.path.getsize(q)))
            return c[0]
    sys.exit(f"No .csv found in {here} or {os.getcwd()}. "
             "Pass the path: python3 aml_profile_filter.py HI-Small_Trans.csv")


def main(path):
    print("reading", os.path.abspath(path), flush=True)
    out_dir = os.path.dirname(os.path.abspath(path))
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rd = csv.reader(f)
        header = [h.strip() for h in next(rd)]
        # positional: the file has two columns both named "Account"
        need = ["Timestamp", "From Bank", "To Bank", "Amount Received", "Receiving Currency",
                "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
        idx = {}
        for name in need:
            if name not in header:
                sys.exit(f"column {name!r} missing; header = {header}")
            idx[name] = header.index(name)
        acc = [i for i, h in enumerate(header) if h.startswith("Account")]
        if len(acc) != 2:
            sys.exit(f"expected 2 Account columns, header = {header}")

        n = 0
        short_rows, first_short = 0, None
        width = max(max(idx.values()), max(acc)) + 1
        fmt, cur_pair, laund, day = Counter(), Counter(), Counter(), Counter()
        intra = inter = 0
        banks = set()
        kept = 0
        kept_fmt, kept_laund, kept_banks_out, kept_day = Counter(), Counter(), Counter(), Counter()
        amt_mismatch = 0
        amounts = []
        gz = gzip.open(os.path.join(out_dir, "aml_usd_interbank.csv.gz"), "wt", newline="")
        wr = csv.writer(gz)
        wr.writerow(["row", "timestamp", "from_bank", "to_bank",
                     "amount_paid", "amount_received", "format", "is_laundering"])
        for r in rd:
            if len(r) < width:          # blank line or truncated row: count, skip
                short_rows += 1
                if first_short is None:
                    first_short = {"line": rd.line_num, "fields": r}
                continue
            n += 1
            if n % 1_000_000 == 0:
                print(f"  {n:,} rows", flush=True)
            fb, tb = r[idx["From Bank"]], r[idx["To Bank"]]
            f_ = r[idx["Payment Format"]]
            pc, rc = r[idx["Payment Currency"]], r[idx["Receiving Currency"]]
            lab = r[idx["Is Laundering"]]
            fmt[f_] += 1
            cur_pair[(pc, rc) if pc == rc else ("cross", "cross")] += 1
            laund[lab] += 1
            d = r[idx["Timestamp"]][:10]
            day[d] += 1
            banks.add(fb); banks.add(tb)
            if fb == tb:
                intra += 1
                continue
            inter += 1
            if f_ in KEEP_FORMATS and pc == CUR and rc == CUR:
                ap, ar = r[idx["Amount Paid"]], r[idx["Amount Received"]]
                if abs(float(ap) - float(ar)) > 1e-6:
                    amt_mismatch += 1
                wr.writerow([n, r[idx["Timestamp"]], fb, tb, ap, ar, f_, lab])
                kept += 1
                kept_fmt[f_] += 1
                kept_laund[lab] += 1
                kept_banks_out[fb] += 1
                kept_day[d] += 1
                amounts.append(float(ap))
        gz.close()

    prof = {
        "input": os.path.basename(path), "sha256": sha256(path), "bytes": os.path.getsize(path),
        "header": header, "rows": n, "short_rows_skipped": short_rows, "first_short_row": first_short, "distinct_banks": len(banks),
        "intra_bank_rows": intra, "inter_bank_rows": inter,
        "format_counts": dict(fmt.most_common()),
        "same_currency_counts": {f"{a}|{b}": c for (a, b), c in cur_pair.most_common(20)},
        "is_laundering_counts": dict(laund),
        "rows_per_day": dict(sorted(day.items())),
        "kept": {"rows": kept, "by_format": dict(kept_fmt), "by_laundering": dict(kept_laund),
                 "paid_ne_received": amt_mismatch, "distinct_sending_banks": len(kept_banks_out),
                 "top30_sending_banks": kept_banks_out.most_common(30),
                 "rows_per_day": dict(sorted(kept_day.items())),
                 "amount_quantiles": quantiles(amounts)},
    }
    with open(os.path.join(out_dir, "aml_profile.json"), "w") as f:
        json.dump(prof, f, indent=1)
    print(json.dumps({k: prof[k] for k in ("rows", "short_rows_skipped", "distinct_banks",
                                           "inter_bank_rows")}), "kept", kept)
    print("wrote", os.path.join(out_dir, "aml_profile.json"), "and aml_usd_interbank.csv.gz")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".csv") else find_input())
