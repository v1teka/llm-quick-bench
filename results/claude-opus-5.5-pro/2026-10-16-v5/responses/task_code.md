```python
# задача 1
import sys, math, random, time
from collections import deque

def main():
    T0 = time.time()
    TL = 9.0
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    P = [(float(data[1 + 2 * i]), float(data[2 + 2 * i])) for i in range(n)]
    if n <= 3:
        sys.stdout.write(" ".join(map(str, range(n))) + "\n")
        return
    dist = math.dist
    K = min(10, n - 1)
    xs = [p[0] for p in P]; ys = [p[1] for p in P]
    minx = min(xs); miny = min(ys)
    Wd = max(xs) - minx + 1.0; Hd = max(ys) - miny + 1.0
    cs = math.sqrt(Wd * Hd / n)
    for it in range(4):
        grid = {}
        for i in range(n):
            key = (int((xs[i] - minx) / cs), int((ys[i] - miny) / cs))
            l = grid.get(key)
            if l is None:
                grid[key] = [i]
            else:
                l.append(i)
        if n <= 3 * len(grid) or it == 3:
            break
        cs /= 2.0
    gmax = int(max(Wd, Hd) / cs) + 2
    nb = [None] * n; nd = [None] * n
    for i in range(n):
        cx = int((xs[i] - minx) / cs); cy = int((ys[i] - miny) / cs)
        cand = []
        r = 0
        pi = P[i]
        ds = None
        while True:
            if r == 0:
                l = grid.get((cx, cy))
                if l: cand.extend(l)
            else:
                for dx in range(-r, r + 1):
                    l = grid.get((cx + dx, cy - r))
                    if l: cand.extend(l)
                    l = grid.get((cx + dx, cy + r))
                    if l: cand.extend(l)
                for dy in range(-r + 1, r):
                    l = grid.get((cx - r, cy + dy))
                    if l: cand.extend(l)
                    l = grid.get((cx + r, cy + dy))
                    if l: cand.extend(l)
            if len(cand) > K:
                ds = sorted([(dist(pi, P[j]), j) for j in cand if j != i])
                if len(ds) >= K and (ds[K - 1][0] <= r * cs or r > gmax):
                    break
            if r > gmax + 2:
                ds = sorted([(dist(pi, P[j]), j) for j in range(n) if j != i])
                break
            r += 1
        ds = ds[:K]
        nb[i] = [j for _, j in ds]; nd[i] = [d for d, _ in ds]

    # greedy edge construction
    edges = []
    for i in range(n):
        nbi = nb[i]; ndi = nd[i]
        for k in range(K):
            j = nbi[k]
            if i < j:
                edges.append((ndi[k], i, j))
            elif i not in nb[j]:
                edges.append((ndi[k], j, i))
    edges.sort()
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    deg = [0] * n
    adj = [[] for _ in range(n)]
    cnt = 0
    for d, i, j in edges:
        if deg[i] < 2 and deg[j] < 2:
            ri = find(i); rj = find(j)
            if ri != rj:
                parent[ri] = rj
                deg[i] += 1; deg[j] += 1
                adj[i].append(j); adj[j].append(i)
                cnt += 1
                if cnt == n - 1:
                    break
    ends = [i for i in range(n) if deg[i] < 2]
    other = {}
    for e in ends:
        if e in other: continue
        if deg[e] == 0:
            other[e] = e; continue
        prev = e; cur = adj[e][0]
        while deg[cur] == 2:
            a0 = adj[cur][0]
            nx = a0 if a0 != prev else adj[cur][1]
            prev, cur = cur, nx
        other[e] = cur; other[cur] = e
    tour = []
    def walk(e):
        prev = -1; cur = e
        while True:
            tour.append(cur)
            nxt = -1
            for v in adj[cur]:
                if v != prev:
                    nxt = v; break
            if nxt == -1:
                break
            prev, cur = cur, nxt
    rem = set(ends)
    e = ends[0]
    while True:
        o = other[e]
        rem.discard(e); rem.discard(o)
        walk(e)
        if not rem:
            break
        Po = P[o]
        e = min(rem, key=lambda z: dist(Po, P[z]))
    if len(tour) != n:
        tour = list(range(n))

    t = tour
    pos = [0] * n
    for i, c in enumerate(t):
        pos[c] = i

    def rev(i, j):
        L = (j - i) % n + 1
        if L + L > n:
            i, j = (j + 1) % n, (i - 1) % n
            L = n - L
        if L < 2:
            return
        if i <= j:
            seg = t[i:j + 1]; seg.reverse(); t[i:j + 1] = seg
            for k in range(i, j + 1):
                pos[t[k]] = k
        else:
            for _ in range(L >> 1):
                ci = t[i]; cj = t[j]
                t[i] = cj; pos[cj] = i
                t[j] = ci; pos[ci] = j
                i += 1
                if i == n: i = 0
                j -= 1
                if j < 0: j = n - 1

    def move(a, b, c, d):
        pa = pos[a]
        if t[pa + 1 if pa + 1 < n else 0] == b:
            rev(pos[b], pos[c])
        else:
            rev(pa, pos[d])

    def try2(a):
        pa = pos[a]
        sa = t[pa + 1 if pa + 1 < n else 0]; ra = t[pa - 1]
        Pa = P[a]; nba = nb[a]; nda = nd[a]
        for b, succ in ((sa, True), (ra, False)):
            Pb = P[b]
            dab = dist(Pa, Pb)
            for k in range(K):
                g1 = dab - nda[k]
                if g1 <= 1e-9:
                    break
                c = nba[k]; pc = pos[c]
                if succ:
                    d = t[pc + 1 if pc + 1 < n else 0]
                else:
                    d = t[pc - 1]
                if c == b or d == a:
                    continue
                g = g1 + dist(P[c], P[d]) - dist(Pb, P[d])
                if g > 1e-9:
                    move(a, b, c, d)
                    return g, (a, b, c, d)
        return 0.0, None

    def tryor(a):
        for fwd in (True, False):
            s1 = a; s2 = a; seg = [a]
            for L in range(3):
                if L > 0:
                    q = pos[s2]
                    s2 = t[q + 1 if q + 1 < n else 0] if fwd else t[q - 1]
                    if s2 in seg:
                        break
                    seg.append(s2)
                q = pos[s1]
                p = t[q - 1] if fwd else t[q + 1 if q + 1 < n else 0]
                q = pos[s2]
                n2 = t[q + 1 if q + 1 < n else 0] if fwd else t[q - 1]
                if p in seg or n2 in seg or p == n2:
                    break
                G1 = dist(P[p], P[s1]) + dist(P[s2], P[n2]) - dist(P[p], P[n2])
                if G1 <= 1e-9:
                    continue
                ends_ = (s1,) if s1 == s2 else (s1, s2)
                for s in ends_:
                    nbs = nb[s]; nds = nd[s]
                    for k in range(K):
                        if nds[k] >= G1:
                            break
                        c = nbs[k]
                        if c in seg:
                            continue
                        qc = pos[c]
                        csn = t[qc + 1 if qc + 1 < n else 0]; cpv = t[qc - 1]
                        if fwd:
                            e1 = (c, csn); e2 = (cpv, c)
                        else:
                            e1 = (c, cpv); e2 = (csn, c)
                        for u, v in (e1, e2):
                            if u in seg or v in seg or v == p or u == n2:
                                continue
                            Pu = P[u]; Pv = P[v]
                            duv = dist(Pu, Pv)
                            addA = dist(Pu, P[s2]) + dist(P[s1], Pv) - duv
                            addB = dist(Pu, P[s1]) + dist(P[s2], Pv) - duv
                            if addA <= addB:
                                add = addA; opt = 0
                            else:
                                add = addB; opt = 1
                            if G1 - add > 1e-9:
                                move(p, s1, u, v)
                                move(p, u, n2, s2)
                                if opt == 1 and s1 != s2:
                                    move(u, s2, s1, v)
                                return G1 - add, (p, n2, s1, s2, u, v)
        return 0.0, None

    dq = deque()
    inq = [False] * n

    def local_search(check_time):
        total = 0.0
        cntp = 0
        while dq:
            a = dq.popleft(); inq[a] = False
            g, nodes = try2(a)
            if nodes is None:
                g, nodes = tryor(a)
            if nodes is not None:
                total += g
                for v in nodes:
                    if not inq[v]:
                        inq[v] = True; dq.append(v)
            if check_time:
                cntp += 1
                if (cntp & 511) == 0 and time.time() - T0 > TL:
                    break
        return total

    for c in t:
        inq[c] = True; dq.append(c)
    local_search(True)
    dq.clear()
    for i in range(n): inq[i] = False

    def tour_len():
        s = 0.0
        for i in range(n):
            s += dist(P[t[i - 1]], P[t[i]])
        return s

    cur = tour_len()
    best = cur
    best_t = t[:]; best_pos = pos[:]
    m = max(1, min(30, n // 4))
    rnd = random.Random(12345)
    while time.time() - T0 < TL:
        for _ in range(8):
            i = rnd.randrange(n)
            l1 = rnd.randint(1, m); l2 = rnd.randint(1, m)
            idx = [(i + 1 + k) % n for k in range(l1 + l2)]
            seg = [t[q] for q in idx]
            a = t[i]; b1 = seg[0]; b2 = seg[l1 - 1]; c1 = seg[l1]; c2 = seg[-1]
            e = t[(i + l1 + l2 + 1) % n]
            delta = (dist(P[a], P[c1]) + dist(P[c2], P[b1]) + dist(P[b2], P[e])
                     - dist(P[a], P[b1]) - dist(P[b2], P[c1]) - dist(P[c2], P[e]))
            new = seg[l1:] + seg[:l1]
            for q, c in zip(idx, new):
                t[q] = c; pos[c] = q
            cur += delta
            for v in (a, b1, b2, c1, c2, e):
                if not inq[v]:
                    inq[v] = True; dq.append(v)
            cur -= local_search(False)
            if cur < best - 1e-7:
                best = cur
                best_t = t[:]; best_pos = pos[:]
            elif cur <= best + 1e-9:
                best = cur
                best_t = t[:]; best_pos = pos[:]
            else:
                t[:] = best_t; pos[:] = best_pos
                cur = best
    sys.stdout.write(" ".join(map(str, best_t)) + "\n")

main()
```

```python
# задача 2
import sys, random, time

def main():
    T0 = time.time()
    TL = 9.2
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0]); m = int(data[1])
    adjs = [set() for _ in range(n)]
    for i in range(m):
        u = int(data[2 + 2 * i]); v = int(data[3 + 2 * i])
        if u != v:
            adjs[u].add(v); adjs[v].add(u)
    adj = [list(s) for s in adjs]
    if n == 0:
        return
    # DSATUR
    col = [-1] * n
    sat = [set() for _ in range(n)]
    deg = [len(a) for a in adj]
    for _ in range(n):
        bv = -1; bk = None
        for v in range(n):
            if col[v] < 0:
                key = (len(sat[v]), deg[v])
                if bk is None or key > bk:
                    bk = key; bv = v
        v = bv; c = 0; s = sat[v]
        while c in s:
            c += 1
        col[v] = c
        for u in adj[v]:
            sat[u].add(c)
    k = max(col) + 1
    best_col = col[:]
    rnd = random.Random(2024)

    def tabucol(init, k, deadline):
        col = init[:]
        rk = range(k)
        gamma = [[0] * k for _ in range(n)]
        for v in range(n):
            gv = gamma[v]
            for u in adj[v]:
                gv[col[u]] += 1
        f = sum(gamma[v][col[v]] for v in range(n)) // 2
        conf = []; cpos = [-1] * n
        for v in range(n):
            if gamma[v][col[v]] > 0:
                cpos[v] = len(conf); conf.append(v)
        tabu = [[0] * k for _ in range(n)]
        it = 0
        bestf = f
        BIG = 1 << 30
        while f > 0:
            it += 1
            if (it & 63) == 0 and time.time() > deadline:
                return None
            bd = BIG; bv = -1; bc = -1; nbst = 0
            for v in conf:
                row = gamma[v]; cv = col[v]; base = row[cv]
                thr = base + bd
                if min(row) > thr:
                    continue
                tv = tabu[v]
                for c in [c for c in rk if row[c] <= thr]:
                    if c == cv:
                        continue
                    d = row[c] - base
                    if d > bd:
                        continue
                    if tv[c] > it and f + d >= bestf:
                        continue
                    if d < bd:
                        bd = d; bv = v; bc = c; nbst = 1
                    else:
                        nbst += 1
                        if rnd.random() * nbst < 1.0:
                            bv = v; bc = c
            if bv < 0:
                bv = conf[rnd.randrange(len(conf))]
                bc = rnd.randrange(k - 1)
                if bc >= col[bv]:
                    bc += 1
                bd = gamma[bv][bc] - gamma[bv][col[bv]]
            v = bv; c = bc
            a = col[v]
            col[v] = c
            f += bd
            tabu[v][a] = it + int(0.6 * len(conf)) + rnd.randrange(10)
            for u in adj[v]:
                gu = gamma[u]
                gu[a] -= 1; gu[c] += 1
                cu = col[u]
                if cu == a:
                    if gu[a] == 0 and cpos[u] >= 0:
                        p = cpos[u]; last = conf[-1]
                        conf[p] = last; cpos[last] = p
                        conf.pop(); cpos[u] = -1
                elif cu == c:
                    if cpos[u] < 0:
                        cpos[u] = len(conf); conf.append(u)
            if gamma[v][c] == 0:
                if cpos[v] >= 0:
                    p = cpos[v]; last = conf[-1]
                    conf[p] = last; cpos[last] = p
                    conf.pop(); cpos[v] = -1
            else:
                if cpos[v] < 0:
                    cpos[v] = len(conf); conf.append(v)
            if f < bestf:
                bestf = f
        return col

    deadline = T0 + TL
    cur = best_col[:]
    while k > 1 and time.time() < deadline:
        sizes = [0] * k
        for v in range(n):
            sizes[cur[v]] += 1
        s = min(range(k), key=lambda c: sizes[c])
        init = cur[:]
        for v in range(n):
            if init[v] == s:
                cntc = [0] * k
                for u in adj[v]:
                    if init[u] != s:
                        cntc[init[u]] += 1
                bc = -1; bvv = None
                for c in range(k):
                    if c != s and (bvv is None or cntc[c] < bvv):
                        bvv = cntc[c]; bc = c
                init[v] = bc
        init = [c if c < s else c - 1 for c in init]
        res = tabucol(init, k - 1, deadline)
        if res is None:
            break
        k -= 1
        cur = res
        best_col = res[:]
    sys.stdout.write(" ".join(map(str, best_col)) + "\n")

main()
```

```python
# задача 3
import sys, random, time

def main():
    T0 = time.time()
    TL = 9.0
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); W = int(data[1])
    ws = [int(data[2 + 2 * i]) for i in range(n)]
    hs = [int(data[3 + 2 * i]) for i in range(n)]
    INF = 1 << 60
    LB = -(-sum(ws[i] * hs[i] for i in range(n)) // W)

    def decode_bl(order):
        h = [0] * W
        starts = [0]
        xs = [0] * n; ys = [0] * n
        for i in order:
            w = ws[i]; lim = W - w
            by = INF; bx = 0
            for x in starts:
                if x > lim:
                    break
                y = max(h[x:x + w])
                if y < by:
                    by = y; bx = x
            xs[i] = bx; ys[i] = by
            if w > 0:
                h[bx:bx + w] = [by + hs[i]] * w
                starts = [0] + [x for x in range(1, W) if h[x] != h[x - 1]]
        H = max(ys[i] + hs[i] for i in range(n)) if n else 0
        return H, sum(h), xs, ys

    def decode_bf(order):
        h = [0] * W
        rem = list(order)
        xs = [0] * n; ys = [0] * n
        while rem:
            y = min(h); x0 = h.index(y); x1 = x0 + 1
            while x1 < W and h[x1] == y:
                x1 += 1
            gw = x1 - x0
            pick = -1
            for idx, i in enumerate(rem):
                if ws[i] <= gw:
                    pick = idx; break
            l = h[x0 - 1] if x0 > 0 else INF
            r = h[x1] if x1 < W else INF
            if pick < 0:
                nh = min(l, r)
                h[x0:x1] = [nh] * gw
                continue
            i = rem.pop(pick); w = ws[i]
            if l >= r:
                x = x0
            else:
                x = x1 - w
            xs[i] = x; ys[i] = y
            if w > 0:
                h[x:x + w] = [y + hs[i]] * w
        H = max(ys[i] + hs[i] for i in range(n)) if n else 0
        return H, sum(h), xs, ys

    idxs = list(range(n))
    orders = [
        sorted(idxs, key=lambda i: (-hs[i], -ws[i])),
        sorted(idxs, key=lambda i: (-ws[i], -hs[i])),
        sorted(idxs, key=lambda i: (-ws[i] * hs[i], -hs[i])),
        sorted(idxs, key=lambda i: (-(ws[i] + hs[i]), -hs[i])),
        sorted(idxs, key=lambda i: (-max(ws[i], hs[i]), -hs[i])),
    ]
    best = None  # (H, xs, ys)
    starts = {}
    for name, dec in (("bl", decode_bl), ("bf", decode_bf)):
        bk = None; bo = None
        for o in orders:
            res = dec(o)
            key = (res[0], res[1])
            if bk is None or key < bk:
                bk = key; bo = o
            if best is None or res[0] < best[0]:
                best = (res[0], res[2], res[3])
        starts[name] = (bo[:], bk)

    rnd = random.Random(7)

    def ls(dec, order, ck, deadline):
        nonlocal best
        cur = order[:]
        while time.time() < deadline:
            if best[0] <= LB:
                return
            new = cur[:]
            i = rnd.randrange(n)
            if rnd.random() < 0.5:
                j = rnd.randrange(n)
            else:
                j = min(n - 1, max(0, i + rnd.randint(-8, 8)))
            if i == j:
                continue
            if rnd.random() < 0.5:
                new[i], new[j] = new[j], new[i]
            else:
                x = new.pop(i); new.insert(j, x)
            res = dec(new)
            k2 = (res[0], res[1])
            if k2 <= ck:
                cur = new; ck = k2
                if res[0] < best[0]:
                    best = (res[0], res[2], res[3])

    if n >= 2:
        now = time.time()
        mid = now + (T0 + TL - now) / 2.0
        o, k0 = starts["bf"]
        ls(decode_bf, o, k0, mid)
        o, k0 = starts["bl"]
        ls(decode_bl, o, k0, T0 + TL)

    _, xs, ys = best
    sys.stdout.write("\n".join("%d %d" % (xs[i], ys[i]) for i in range(n)) + "\n")

main()
```
