```python
# задача 1
import sys, time, random, math

def main():
    t0 = time.time()
    TL = 9.0
    data = sys.stdin.read().split()
    n = int(data[0]); W = int(data[1])
    RW = []; RH = []
    for i in range(n):
        RW.append(int(data[2 + 2 * i])); RH.append(int(data[3 + 2 * i]))
    rng = random.Random(12345)
    INF = float('inf')
    rngW = range(1, W)

    def run(order, k0, st_prev, ps_prev):
        if k0 == 0:
            sky = [0] * W; H = 0
            st = []; ps = []
        else:
            s, H = st_prev[k0]
            sky = s[:]
            st = st_prev[:k0]; ps = ps_prev[:k0]
        for k in range(k0, n):
            st.append((sky[:], H))
            i = order[k]; w = RW[i]; h = RH[i]
            B = [0]
            prev = sky[0]
            for x in rngW:
                v = sky[x]
                if v != prev:
                    B.append(x); prev = v
            B.append(W)
            by = INF; bx = 0
            lim = W - w
            for b in B:
                if b <= lim:
                    y = max(sky[b:b + w])
                    if y < by or (y == by and b < bx):
                        by = y; bx = b
                x = b - w
                if x >= 0:
                    y = max(sky[x:b])
                    if y < by or (y == by and x < bx):
                        by = y; bx = x
            top = by + h
            sky[bx:bx + w] = [top] * w
            if top > H:
                H = top
            ps.append((bx, by))
        sc = H + 0.5 * sum(sky) / (W * H) if H > 0 else 0
        return st, ps, H, sc

    idx = list(range(n))
    inits = [
        sorted(idx, key=lambda i: (-RH[i], -RW[i])),
        sorted(idx, key=lambda i: (-RW[i], -RH[i])),
        sorted(idx, key=lambda i: (-RW[i] * RH[i])),
        sorted(idx, key=lambda i: (-(RW[i] + RH[i]))),
        sorted(idx, key=lambda i: (-max(RW[i], RH[i]))),
    ]
    best = None
    for o in inits:
        st, ps, H, sc = run(o, 0, None, None)
        if best is None or sc < best[3]:
            best = (o[:], st, ps, sc, H)
    cur_o, cur_st, cur_ps, cur_sc, cur_H = best[0][:], best[1], best[2], best[3], best[4]
    best_o, best_ps, best_sc, best_H = cur_o[:], cur_ps, cur_sc, cur_H

    T0 = 0.6; T1 = 0.02
    it = 0
    last_imp = 0
    while True:
        it += 1
        el = time.time() - t0
        if el > TL:
            break
        frac = el / TL
        T = T0 * (T1 / T0) ** frac
        o = cur_o[:]
        r = rng.random()
        if r < 0.45:
            i = rng.randrange(n); j = rng.randrange(n)
            if i == j:
                continue
            o[i], o[j] = o[j], o[i]
            k0 = min(i, j)
        elif r < 0.9:
            i = rng.randrange(n); j = rng.randrange(n)
            if i == j:
                continue
            e = o.pop(i); o.insert(j, e)
            k0 = min(i, j)
        else:
            i = rng.randrange(n - 1)
            L = rng.randint(2, min(8, n - i))
            o[i:i + L] = o[i:i + L][::-1]
            k0 = i
        st, ps, H, sc = run(o, k0, cur_st, cur_ps)
        d = sc - cur_sc
        if d <= 0 or rng.random() < math.exp(-d / T):
            cur_o, cur_st, cur_ps, cur_sc, cur_H = o, st, ps, sc, H
            if sc < best_sc - 1e-12:
                best_o, best_ps, best_sc, best_H = o[:], ps, sc, H
                last_imp = it
        if it - last_imp > 4000:
            st, ps, H, sc = run(best_o, 0, None, None)
            cur_o, cur_st, cur_ps, cur_sc, cur_H = best_o[:], st, ps, sc, H
            last_imp = it

    res = [None] * n
    for k in range(n):
        res[best_o[k]] = best_ps[k]
    sys.stdout.write("\n".join("%d %d" % p for p in res) + "\n")

main()
```
