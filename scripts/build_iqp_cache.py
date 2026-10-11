"""Offline: derive IQP polynomials (scipy MILP) for every bank pattern in the
shipped instance files and save them to settle/iqp_data/iqp_cache.json."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settle.instances import load_instances_json
from settle.encodings import e1_iqp
from settle import iqp

PATH = "settle/iqp_data/iqp_cache.json"
cache = iqp.load_cache_text(open(PATH).read()) if os.path.exists(PATH) else {}
for f in ("data/demo_instances.json", "data/bench_instances.json", "data/t0_instances.json", "data/c_instances.json"):
    for inst in load_instances_json(open(f).read()):
        t = time.time()
        n0 = len(cache)
        e1_iqp(inst, cache=cache)
        print(inst.name, "new patterns", len(cache) - n0, "%.1fs" % (time.time() - t), flush=True)
iqp.save_cache(cache, PATH)
print("patterns:", len(cache), "not minimised:", sum(not v.get("minimised", True) for v in cache.values()))
