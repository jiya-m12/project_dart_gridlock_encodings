"""Encoding gates on the AML tier (cents), independent of settle.verify: E2/E4 exactness from the
penalty definitions in Python integers ("own_exact"), compared with the repo's float64 path;
E0 qubit counts; an own re-check of every shipped IQP polynomial; E1 exactness.
Strict ties: f within 0.1 cent of min, value == opt exactly. Run from repo root."""
import sys, os, json, math, collections; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from settle.instances import load_instances_json
from settle.encodings import e0_standard, e1_iqp, e2_indicator, e4_overflow_sq, penalty_weights, binding_banks
from settle.verify import reduced_cost, check_exact
from settle.exact import feasibility_mask, values
from settle.qaoa import Problem
from settle import poly as P, iqp
H=load_instances_json(open('data/aml_instances.json').read())
B=load_instances_json(open('data/bench_instances.json').read())
def own_exact(inst, enc_name, rule):
    """definition-level: f(x) = -value + sum_u lam_u * pen(g_u), python ints, no polynomial"""
    lam=[int(round(l)) for l in penalty_weights(inst,rule,2.0)]
    bb=set(binding_banks(inst)); n=inst.n; best=None; arg=[]
    for m in range(2**n):
        net=[0]*inst.n_banks; val=0
        for k,(a,b,w) in enumerate(inst.arcs):
            if m>>k&1: net[a]+=w; net[b]-=w; val+=w
        pen=0
        for u in bb:
            g=net[u]-inst.liquidity[u]
            if g>0: pen+= lam[u]*(1 if enc_name=='E2' else g*g)
        f=-val+pen
        if best is None or f<best: best,arg=f,[m]
        elif f==best: arg.append(m)
    feas=feasibility_mask(inst); v=values(inst); opt=v[feas].max()
    return all(feas[j] and v[j]==opt for j in arg), arg
def repo_strict(inst, enc):
    f=reduced_cost(enc); feas=feasibility_mask(inst); v=values(inst); opt=v[feas].max()
    am=np.flatnonzero(f<=f.min()+0.1)   # f differences between distinct states >= 1 cent here
    return bool(np.all(feas[am] & (v[am]==opt))), am
res=collections.Counter()
for inst in H:
    for name,fn in (('E2',e2_indicator),('E4',e4_overflow_sq)):
        for rule in ('local','neighbour','global'):
            enc=fn(inst,rule=rule)
            ok_own,arg_own=own_exact(inst,name,rule)
            ok_repo,am=repo_strict(inst,enc)
            ok_loose=check_exact(inst,enc)['exact']
            res[name,rule,'own_exact']+=ok_own; res[name,rule,'repo_strict']+=ok_repo; res[name,rule,'repo_check_exact']+=ok_loose
            res[name,rule,'own==repo_argmin']+= sorted(arg_own)==sorted(am.tolist())
print('E2/E4 exactness on',len(H),'hard AML instances (count exact):')
for k in sorted(res): print(' ',k,res[k])
def dyn(enc):
    z=P.to_ising(enc.poly); z.pop((),None); a=np.array([abs(c) for c in z.values()]); a=a[a>1e-6]
    return a.max()/a.min()
for lab,S in (('AML',H),('bench',B)):
    for name,fn in (('E2',e2_indicator),('E4',e4_overflow_sq)):
        d=[dyn(fn(i,rule='neighbour')) for i in S]
        print(lab,name,'neighbour max|J|/min|J| median %.3g max %.3g'%(np.median(d),np.max(d)))
    q=[]
    for i in S:
        tot=i.n
        for u in binding_banks(i):
            vs,cs=i.local_coeffs(u); smax=i.liquidity[u]+sum(-c for c in cs if c<0); tot+=max(1,math.ceil(math.log2(smax+1)))
        q.append(tot)
    print(lab,'E0 qubits median',np.median(q),'max',max(q),' payments median',np.median([i.n for i in S]))

# ---- E1 ----
from settle.exact import feasibility_mask, values
from settle import poly as P, iqp
H=load_instances_json(open('data/aml_instances.json').read())
cache=iqp.load_cache_text(open('settle/iqp_data/iqp_cache.json').read())
new=dict(cache)   # the AML patterns were merged into the shipped cache in v7: re-check every shipped polynomial
# independent re-check (own evaluation of the 3 conditions)
bad=0
for key,r in new.items():
    d=r['d']; k=r['k']; feas=[c=='1' for c in key.split(':')[1]]
    for m in range(2**d):
        vals=[]
        for s in range(2**k):
            bits=[(m>>j)&1 for j in range(d)]+[(s>>j)&1 for j in range(k)]
            vals.append(sum(c*np.prod([bits[i] for i in t]) for t,c in r['coefs'].items()))
        if feas[m]: bad+= not (max(vals)<=1e-7 and abs(max(vals))<=1e-7)
        else: bad+= not (max(vals)<=-1+1e-7)
print('shipped IQP polynomials violating conditions (own check):',bad,'of',len(new))
cache.update(new)
res=collections.Counter(); q=[]; dyn=[]
for inst in H:
    for rule in ('local','neighbour','global'):
        enc=e1_iqp(inst,rule=rule,cache=dict(cache),derive_missing=False)
        f=reduced_cost(enc); feas=feasibility_mask(inst); v=values(inst); opt=v[feas].max()
        am=np.flatnonzero(f<=f.min()+0.1)
        res[rule,'strict']+=bool(np.all(feas[am]&(v[am]==opt))); res[rule,'repo']+=check_exact(inst,enc)['exact']
        if rule=='neighbour':
            q.append(enc.n_total); z=P.to_ising(enc.poly); z.pop((),None); a=np.array([abs(c) for c in z.values()]); a=a[a>1e-6]; dyn.append(a.max()/a.min())
print('E1 exact counts on 22:',dict(res))
print('E1 qubits median',np.median(q),'max',max(q),' dyn range median %.3g max %.3g'%(np.median(dyn),max(dyn)))
