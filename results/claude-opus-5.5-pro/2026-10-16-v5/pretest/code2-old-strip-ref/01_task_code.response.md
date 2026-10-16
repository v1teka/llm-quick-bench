```python
# задача 1
import sys, math, random, time
from collections import deque

def main():
    t0 = time.time()
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    X = [float(data[1 + 2 * i]) for i in range(n)]
    Y = [float(data[2 + 2 * i]) for i in range(n)]
    if n <= 3:
        print(' '.join(map(str, range(n))))
        return
    deadline = t0 + 8.7
    sqrt = math.sqrt
    K = min(10, n - 1)
    minx = min(X); maxx = max(X); miny = min(Y); maxy = max(Y)
    span = max(maxx - minx, maxy - miny, 1.0)
    G = max(1, int(math.sqrt(n / 2.0)))
    cs = span / G * (1 + 1e-9)
    cells = {}
    cx = [0] * n; cy = [0] * n
    for i in range(n):
        a = int((X[i] - minx) / cs); b = int((Y[i] - miny) / cs)
        if a >= G: a = G - 1
        if b >= G: b = G - 1
        cx[i] = a; cy[i] = b
        cells.setdefault((a, b), []).append(i)
    neigh = [None] * n
    neighD = [None] * n
    for i in range(n):
        xi = X[i]; yi = Y[i]; a = cx[i]; b = cy[i]
        cand = []
        r = 0
        while True:
            if r == 0:
                lst = [(a, b)]
            else:
                lst = []
                for dx in range(-r, r + 1):
                    lst.append((a + dx, b - r)); lst.append((a + dx, b + r))
                for dy in range(-r + 1, r):
                    lst.append((a - r, b + dy)); lst.append((a + r, b + dy))
            for key in lst:
                c = cells.get(key)
                if c:
                    for j in c:
                        if j != i:
                            dx = X[j] - xi; dy = Y[j] - yi
                            cand.append((dx * dx + dy * dy, j))
            if len(cand) >= K:
                cand.sort()
                if cand[K - 1][0] <= (r * cs) ** 2 or r > G:
                    break
            elif r > G:
                break
            r += 1
        cand = cand[:K]
        neigh[i] = [j for _, j in cand]
        neighD[i] = [sqrt(d) for d, _ in cand]
    KK = [len(neigh[i]) for i in range(n)]

    # greedy edge construction
    edges = set()
    for i in range(n):
        for j in neigh[i]:
            if i < j: edges.add((i, j))
            else: edges.add((j, i))
    el = []
    for (i, j) in edges:
        dx = X[i] - X[j]; dy = Y[i] - Y[j]
        el.append((dx * dx + dy * dy, i, j))
    el.sort()
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    deg = [0] * n
    adj = [[] for _ in range(n)]
    for _, i, j in el:
        if deg[i] < 2 and deg[j] < 2:
            ri = find(i); rj = find(j)
            if ri != rj:
                parent[ri] = rj
                deg[i] += 1; deg[j] += 1
                adj[i].append(j); adj[j].append(i)
    frags = []
    vis = [False] * n
    for i in range(n):
        if not vis[i] and deg[i] <= 1:
            path = [i]; vis[i] = True; prev = -1; cur = i
            while True:
                nx = -1
                for v in adj[cur]:
                    if v != prev and not vis[v]:
                        nx = v
                if nx == -1:
                    break
                path.append(nx); vis[nx] = True; prev = cur; cur = nx
            frags.append(path)
    unused = list(range(1, len(frags)))
    tour = list(frags[0])
    while unused:
        e = tour[-1]; ex = X[e]; ey = Y[e]
        bd = float('inf'); bi = -1; brev = False
        for idx in range(len(unused)):
            p = frags[unused[idx]]
            s = p[0]; t = p[-1]
            d1 = (X[s] - ex) ** 2 + (Y[s] - ey) ** 2
            if d1 < bd: bd = d1; bi = idx; brev = False
            d2 = (X[t] - ex) ** 2 + (Y[t] - ey) ** 2
            if d2 < bd: bd = d2; bi = idx; brev = True
        p = frags[unused[bi]]
        if brev:
            tour.extend(reversed(p))
        else:
            tour.extend(p)
        unused[bi] = unused[-1]; unused.pop()

    pos = [0] * n
    for k, c in enumerate(tour):
        pos[c] = k

    def dist(a, b):
        dx = X[a] - X[b]; dy = Y[a] - Y[b]
        return sqrt(dx * dx + dy * dy)

    def rev(i, j):
        l = (j - i) % n + 1
        if 2 * l > n:
            i, j = (j + 1) % n, (i - 1) % n
            l = n - l
        if l < 2:
            return
        if i <= j:
            seg = tour[i:j + 1]; seg.reverse(); tour[i:j + 1] = seg
            for k in range(i, j + 1):
                pos[tour[k]] = k
        else:
            for _ in range(l // 2):
                a = tour[i]; b = tour[j]
                tour[i] = b; pos[b] = i; tour[j] = a; pos[a] = j
                i += 1
                if i == n: i = 0
                j -= 1
                if j < 0: j = n - 1

    def move2(a, b, c, d):
        pa = pos[a] + 1
        if pa == n: pa = 0
        if tour[pa] == b:
            rev(pos[b], pos[c])
        else:
            rev(pos[a], pos[d])

    inq = [False] * n
    Q = deque()
    def push(x):
        if not inq[x]:
            inq[x] = True
            Q.append(x)

    def try2(a):
        pa = pos[a]
        na = neigh[a]; nd = neighD[a]; ka = KK[a]
        for dirn in (1, -1):
            if dirn == 1:
                b = tour[pa + 1] if pa + 1 < n else tour[0]
            else:
                b = tour[pa - 1]
            dab = dist(a, b)
            for idx in range(ka):
                dac = nd[idx]
                if dac >= dab:
                    break
                c = na[idx]
                pc = pos[c]
                if dirn == 1:
                    d = tour[pc + 1] if pc + 1 < n else tour[0]
                else:
                    d = tour[pc - 1]
                if c == b or d == a:
                    continue
                delta = dac + dist(b, d) - dab - dist(c, d)
                if delta < -1e-9:
                    move2(a, b, c, d)
                    push(a); push(b); push(c); push(d)
                    return delta
        return 0.0

    def tryor(a):
        for dirn in (1, -1):
            for L in (1, 2, 3):
                if L + 4 > n:
                    break
                pa = pos[a]
                s1 = a
                if dirn == 1:
                    s2 = tour[(pa + L - 1) % n]; p = tour[pa - 1]; nx = tour[(pa + L) % n]
                else:
                    s2 = tour[(pa - L + 1) % n]; p = tour[(pa + 1) % n]; nx = tour[(pa - L) % n]
                g1 = dist(p, s1) + dist(s2, nx) - dist(p, nx)
                if g1 <= 1e-9:
                    continue
                ss = (s1,) if L == 1 else (s1, s2)
                for s in ss:
                    ns = neigh[s]; nds = neighD[s]
                    for idx in range(KK[s]):
                        if nds[idx] >= g1:
                            break
                        c = ns[idx]
                        pc = pos[c]
                        if dirn == 1:
                            if (pc - pa) % n < L: continue
                        else:
                            if (pa - pc) % n < L: continue
                        cf = tour[(pc + dirn) % n]; cb = tour[(pc - dirn) % n]
                        for (e, g) in ((c, cf), (cb, c)):
                            if g == p:
                                continue
                            pe = pos[e]; pg = pos[g]
                            if dirn == 1:
                                if (pe - pa) % n < L or (pg - pa) % n < L: continue
                            else:
                                if (pa - pe) % n < L or (pa - pg) % n < L: continue
                            deg_ = dist(e, g)
                            add1 = dist(e, s1) + dist(s2, g)
                            add2 = dist(e, s2) + dist(s1, g)
                            if add1 <= add2:
                                delta = add1 - deg_ - g1; rv = False
                            else:
                                delta = add2 - deg_ - g1; rv = True
                            if delta < -1e-9:
                                move2(p, s1, e, g)
                                if e != nx:
                                    move2(p, e, nx, s2)
                                if (not rv) and L > 1:
                                    move2(e, s2, s1, g)
                                push(p); push(nx); push(s1); push(s2); push(e); push(g)
                                return delta
        return 0.0

    def run_ls():
        tot = 0.0
        cnt = 0
        while Q:
            a = Q.popleft(); inq[a] = False
            while True:
                d = try2(a)
                if d == 0.0:
                    d = tryor(a)
                if d == 0.0:
                    break
                tot += d
            cnt += 1
            if (cnt & 127) == 0 and time.time() > deadline:
                break
        return tot

    for c in tour:
        push(c)
    run_ls()
    while Q:
        inq[Q.pop()] = False

    Lmax = min(50, (n - 2) // 2)
    rnd = random.randint
    if Lmax >= 1 and n >= 8:
        while time.time() < deadline:
            L1 = rnd(1, Lmax); L2 = rnd(1, Lmax)
            p1 = rnd(1, n - L1 - L2 - 1)
            p2 = p1 + L1; p3 = p2 + L2
            saved_tour = tour[:]; saved_pos = pos[:]
            a = tour[p1 - 1]; b = tour[p1]; c = tour[p2 - 1]; d = tour[p2]; e = tour[p3 - 1]; f = tour[p3]
            delta = dist(a, d) + dist(e, b) + dist(c, f) - dist(a, b) - dist(c, d) - dist(e, f)
            tour[p1:p3] = tour[p2:p3] + tour[p1:p2]
            for k in range(p1, p3):
                pos[tour[k]] = k
            push(a); push(b); push(c); push(d); push(e); push(f)
            delta += run_ls()
            while Q:
                inq[Q.pop()] = False
            if delta >= 1e-10:
                tour[:] = saved_tour; pos[:] = saved_pos

    sys.stdout.write(' '.join(map(str, tour)) + '\n')

main()
```

```python
# задача 2
import sys, random, time

def main():
    t0 = time.time()
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); m = int(data[1])
    nb = [set() for _ in range(n)]
    for i in range(m):
        u = int(data[2 + 2 * i]); v = int(data[3 + 2 * i])
        if u != v:
            nb[u].add(v); nb[v].add(u)
    adj = [list(s) for s in nb]
    deadline = t0 + 9.3
    if n == 0:
        print(); return

    # DSATUR
    col = [-1] * n
    sat = [set() for _ in range(n)]
    degu = [len(adj[v]) for v in range(n)]
    for _ in range(n):
        best = -1; bk = None
        for v in range(n):
            if col[v] < 0:
                key = (len(sat[v]), degu[v])
                if bk is None or key > bk:
                    bk = key; best = v
        v = best; c = 0; s = sat[v]
        while c in s:
            c += 1
        col[v] = c
        for u in adj[v]:
            sat[u].add(c)
            if col[u] < 0:
                degu[u] -= 1
    dsat = col[:]
    BIG = 1 << 60
    rr = random.randrange

    def tabucol(col, k):
        gamma = [[0] * k for _ in range(n)]
        for v in range(n):
            g = gamma[v]
            for u in adj[v]:
                g[col[u]] += 1
        tabu = [[0] * k for _ in range(n)]
        for v in range(n):
            tabu[v][col[v]] = BIG
        confl = []; cidx = [-1] * n
        f = 0
        for v in range(n):
            if gamma[v][col[v]] > 0:
                cidx[v] = len(confl); confl.append(v); f += gamma[v][col[v]]
        f //= 2
        bestf = f
        it = 0
        while f > 0:
            it += 1
            if (it & 127) == 0 and time.time() > deadline:
                return None
            bd = BIG; bv = []
            for v in confl:
                row = gamma[v]; cv = row[col[v]]; trow = tabu[v]
                mm = min((g for g, t in zip(row, trow) if t < it), default=BIG)
                d = mm - cv
                da = min(row) - cv
                if da < d and f + da < bestf:
                    d = da
                if d < bd:
                    bd = d; bv = [v]
                elif d == bd:
                    bv.append(v)
            if bd >= BIG // 2:
                v = confl[rr(len(confl))]
                newc = rr(k - 1)
                if newc >= col[v]:
                    newc += 1
            else:
                v = bv[rr(len(bv))] if len(bv) > 1 else bv[0]
                row = gamma[v]; cv = row[col[v]]; trow = tabu[v]
                target = bd + cv
                asp = f + bd < bestf
                cv_col = col[v]
                cs = [c for c in range(k) if row[c] == target and c != cv_col and (trow[c] < it or asp)]
                if not cs:
                    cs = [c for c in range(k) if row[c] == target and c != cv_col]
                newc = cs[rr(len(cs))] if len(cs) > 1 else cs[0]
            old = col[v]
            gv = gamma[v]
            f += gv[newc] - gv[old]
            col[v] = newc
            for u in adj[v]:
                gu = gamma[u]
                gu[old] -= 1; gu[newc] += 1
                cu = col[u]
                if cu == old:
                    if gu[old] == 0:
                        i = cidx[u]; last = confl.pop()
                        if last != u:
                            confl[i] = last; cidx[last] = i
                        cidx[u] = -1
                elif cu == newc:
                    if gu[newc] == 1:
                        cidx[u] = len(confl); confl.append(u)
            if gv[newc] == 0:
                if cidx[v] >= 0:
                    i = cidx[v]; last = confl.pop()
                    if last != v:
                        confl[i] = last; cidx[last] = i
                    cidx[v] = -1
            else:
                if cidx[v] < 0:
                    cidx[v] = len(confl); confl.append(v)
            tl = int(0.6 * len(confl)) + rr(10)
            tabu[v][old] = it + tl
            tabu[v][newc] = BIG
            if f < bestf:
                bestf = f
        return col

    def valid(c):
        for v in range(n):
            cv = c[v]
            for u in adj[v]:
                if c[u] == cv:
                    return False
        return True

    best = dsat[:]
    bestk = max(best) + 1
    while time.time() < deadline and bestk > 1:
        k2 = bestk - 1
        c = best[:]
        sizes = [0] * bestk
        for x in c:
            sizes[x] += 1
        mn = min(sizes)
        rs = [x for x in range(bestk) if sizes[x] == mn]
        r = rs[rr(len(rs))]
        for v in range(n):
            if c[v] == r:
                cnt = [0] * bestk
                for u in adj[v]:
                    if c[u] != r:
                        cnt[c[u]] += 1
                cnt[r] = BIG
                mv = min(cnt)
                ch = [x for x in range(bestk) if cnt[x] == mv]
                c[v] = ch[rr(len(ch))]
        c = [x - 1 if x > r else x for x in c]
        res = tabucol(c, k2)
        if res is None:
            break
        if not valid(res):
            break
        best = res[:]
        bestk = k2

    if not valid(best):
        best = dsat
    sys.stdout.write(' '.join(map(str, best)) + '\n')

main()
```

```python
# задача 3
import sys, random, time

def main():
    t0 = time.time()
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); W = int(data[1])
    ws = [int(data[2 + 2 * i]) for i in range(n)]
    hs = [int(data[3 + 2 * i]) for i in range(n)]
    deadline = t0 + 9.0
    if n == 0:
        return
    area = sum(w * h for w, h in zip(ws, hs))
    LB = max(max(hs), -(-area // W))
    BIG = 1 << 60
    rnd = random.random; rr = random.randrange

    def bl(order):
        sky = [0] * W
        PX = [0] * n; PY = [0] * n
        for i in order:
            w = ws[i]; h = hs[i]
            lim = W - w
            by = max(sky[0:w]); bx = 0
            for x in range(1, lim + 1):
                if sky[x] < sky[x - 1]:
                    y = max(sky[x:x + w])
                    if y < by:
                        by = y; bx = x
            PX[i] = bx; PY[i] = by
            sky[bx:bx + w] = [by + h] * w
        return max(sky), PX, PY

    def bf(policy, noise, keys):
        buckets = [[] for _ in range(W + 1)]
        for i in sorted(range(n), key=lambda i: keys[i]):
            buckets[ws[i]].append(i)
        maxw = W
        while maxw > 0 and not buckets[maxw]:
            maxw -= 1
        sky = [0] * W
        PX = [0] * n; PY = [0] * n
        remaining = n
        while remaining:
            m = min(sky); gx = sky.index(m); ge = gx + 1
            while ge < W and sky[ge] == m:
                ge += 1
            gw = ge - gx
            w = gw if gw < maxw else maxw
            while w > 0 and not buckets[w]:
                w -= 1
            if w == 0:
                L = sky[gx - 1] if gx > 0 else BIG
                R = sky[ge] if ge < W else BIG
                nh = L if L < R else R
                sky[gx:ge] = [nh] * gw
                continue
            if noise and rnd() < noise:
                w2 = w - 1
                while w2 > 0 and not buckets[w2]:
                    w2 -= 1
                if w2 > 0:
                    w = w2
            b = buckets[w]
            if noise and len(b) > 1 and rnd() < noise:
                i = b.pop(rr(len(b)))
            else:
                i = b.pop()
            remaining -= 1
            while maxw > 0 and not buckets[maxw]:
                maxw -= 1
            h = hs[i]
            if w == gw or policy == 0:
                x = gx
            elif policy == 1:
                x = ge - w
            else:
                L = sky[gx - 1] if gx > 0 else BIG
                R = sky[ge] if ge < W else BIG
                if policy == 2:
                    x = gx if L >= R else ge - w
                else:
                    x = gx if L < R else ge - w
            PX[i] = x; PY[i] = m
            sky[x:x + w] = [m + h] * w
        return max(sky), PX, PY

    best = [BIG, None, None]
    def upd(res):
        if res[0] < best[0]:
            best[0] = res[0]; best[1] = res[1]; best[2] = res[2]

    idx = list(range(n))
    orders = [
        sorted(idx, key=lambda i: (-hs[i], -ws[i])),
        sorted(idx, key=lambda i: (-ws[i], -hs[i])),
        sorted(idx, key=lambda i: (-ws[i] * hs[i], -hs[i])),
        sorted(idx, key=lambda i: (-(ws[i] + hs[i]), -hs[i])),
        sorted(idx, key=lambda i: (-max(ws[i], hs[i]), -hs[i])),
    ]
    blbest_h = BIG; blbest_order = None
    for o in orders:
        res = bl(o)
        upd(res)
        if res[0] < blbest_h:
            blbest_h = res[0]; blbest_order = o[:]
    keysets = [
        [hs[i] for i in range(n)],
        [hs[i] * ws[i] for i in range(n)],
        [-hs[i] for i in range(n)],
    ]
    for policy in range(4):
        for ks in keysets:
            upd(bf(policy, 0, ks))

    cur_order = blbest_order[:]; cur_h = blbest_h
    while time.time() < deadline and best[0] > LB:
        if rnd() < 0.6:
            policy = rr(4)
            noise = 0.02 + rnd() * 0.25
            ks = [hs[i] + rnd() * 0.3 * hs[i] * noise * 4 for i in range(n)]
            upd(bf(policy, noise, ks))
        else:
            o = cur_order[:]
            t = rr(3)
            if t == 0:
                a = rr(n); b = rr(n)
                o[a], o[b] = o[b], o[a]
            elif t == 1:
                a = rr(n); b = rr(n)
                x = o.pop(a); o.insert(b, x)
            else:
                a = rr(n); b = min(n - 1, a + 1 + rr(5))
                o[a], o[b] = o[b], o[a]
            res = bl(o)
            upd(res)
            if res[0] <= cur_h:
                cur_h = res[0]; cur_order = o

    PX = best[1]; PY = best[2]
    sys.stdout.write('\n'.join('%d %d' % (PX[i], PY[i]) for i in range(n)) + '\n')

main()
```
