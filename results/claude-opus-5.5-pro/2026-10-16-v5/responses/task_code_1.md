```python
# задача 1
import sys, math, random, time
from collections import deque
from heapq import heappush, heapreplace

def main():
    t0 = time.time()
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    X = [float(v) for v in data[1:2*n+1:2]]
    Y = [float(v) for v in data[2:2*n+2:2]]
    if n <= 3:
        print(' '.join(map(str, range(n))))
        return
    deadline = t0 + 8.5
    hyp = math.hypot
    random.seed(12345)
    K = min(10, n - 1)
    order = sorted(range(n), key=lambda i: (X[i], Y[i]))
    neigh = [None] * n
    neighd = [None] * n
    for idx in range(n):
        a = order[idx]; xa = X[a]; ya = Y[a]
        best = []
        for rng in (range(idx + 1, n), range(idx - 1, -1, -1)):
            for j in rng:
                b = order[j]
                dx = X[b] - xa; dx2 = dx * dx
                if len(best) == K and dx2 >= -best[0][0]:
                    break
                dy = Y[b] - ya; d2 = dx2 + dy * dy
                if len(best) < K:
                    heappush(best, (-d2, b))
                elif d2 < -best[0][0]:
                    heapreplace(best, (-d2, b))
        best.sort(reverse=True)
        neigh[a] = [b for _, b in best]
        neighd[a] = [math.sqrt(-x) for x, _ in best]

    # greedy edge construction
    es = set()
    for a in range(n):
        for b, d in zip(neigh[a], neighd[a]):
            if a < b:
                es.add((d, a, b))
            else:
                es.add((d, b, a))
    es = sorted(es)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    deg = [0] * n
    adj = [[] for _ in range(n)]
    for d, a, b in es:
        if deg[a] < 2 and deg[b] < 2:
            ra = find(a); rb = find(b)
            if ra != rb:
                parent[ra] = rb
                adj[a].append(b); adj[b].append(a)
                deg[a] += 1; deg[b] += 1
    seen = [False] * n
    frags = []
    for s in range(n):
        if not seen[s] and len(adj[s]) < 2:
            path = [s]; seen[s] = True; cur = s
            while True:
                nn = -1
                for w in adj[cur]:
                    if not seen[w]:
                        nn = w; break
                if nn < 0:
                    break
                seen[nn] = True; path.append(nn); cur = nn
            frags.append(path)
    for s in range(n):
        if not seen[s]:
            frags.append([s]); seen[s] = True
    F = len(frags)
    used = [False] * F
    used[0] = True
    tl = frags[0][:]
    for _ in range(F - 1):
        e = tl[-1]; xe = X[e]; ye = Y[e]
        bd = 1e300; bf = -1; brev = False
        for fi in range(F):
            if used[fi]:
                continue
            f = frags[fi]
            d1 = hyp(xe - X[f[0]], ye - Y[f[0]])
            if d1 < bd:
                bd = d1; bf = fi; brev = False
            d2 = hyp(xe - X[f[-1]], ye - Y[f[-1]])
            if d2 < bd:
                bd = d2; bf = fi; brev = True
        used[bf] = True
        f = frags[bf]
        if brev:
            tl.extend(reversed(f))
        else:
            tl.extend(f)
    tour = tl
    pos = [0] * n
    for i, c in enumerate(tour):
        pos[c] = i

    q = deque()
    inq = [False] * n

    def push(x):
        if not inq[x]:
            inq[x] = True
            q.append(x)

    def move(a, b, c, d):
        # b = next(a), d = next(c); reverse path b..c
        i = pos[b]; j = pos[c]
        m = (j - i) % n + 1
        if 2 * m > n:
            i = pos[d]; j = pos[a]; m = n - m
        if m <= 1:
            return
        if i <= j:
            seg = tour[i:j + 1]
            seg.reverse()
            tour[i:j + 1] = seg
            k = i
            for x in seg:
                pos[x] = k; k += 1
        else:
            for k in range(m // 2):
                ii = (i + k) % n; jj = (j - k) % n
                x = tour[ii]; y = tour[jj]
                tour[ii] = y; pos[y] = ii
                tour[jj] = x; pos[x] = jj

    def two_opt(a):
        pa = pos[a]; xa = X[a]; ya = Y[a]
        nb = neigh[a]; nd = neighd[a]
        b = tour[pa + 1] if pa + 1 < n else tour[0]
        xb = X[b]; yb = Y[b]
        dab = hyp(xa - xb, ya - yb)
        for k in range(len(nb)):
            dac = nd[k]
            if dac >= dab - 1e-10:
                break
            c = nb[k]
            pc = pos[c] + 1
            d = tour[pc] if pc < n else tour[0]
            if c == b or d == a:
                continue
            xd = X[d]; yd = Y[d]
            delta = dac + hyp(xb - xd, yb - yd) - dab - hyp(X[c] - xd, Y[c] - yd)
            if delta < -1e-9:
                move(a, b, c, d)
                push(a); push(b); push(c); push(d)
                return -delta
        b = tour[pa - 1]
        xb = X[b]; yb = Y[b]
        dab = hyp(xa - xb, ya - yb)
        for k in range(len(nb)):
            dac = nd[k]
            if dac >= dab - 1e-10:
                break
            c = nb[k]
            d = tour[pos[c] - 1]
            if c == b or d == a:
                continue
            xd = X[d]; yd = Y[d]
            delta = dac + hyp(xb - xd, yb - yd) - dab - hyp(X[c] - xd, Y[c] - yd)
            if delta < -1e-9:
                move(b, a, d, c)
                push(a); push(b); push(c); push(d)
                return -delta
        return 0.0

    def apply_or(p, s1, s2, nx, u, v, rev):
        move(p, s1, u, v)
        pp = pos[p] + 1
        if (tour[pp] if pp < n else tour[0]) == u:
            move(p, u, nx, s2)
        else:
            move(s2, nx, u, p)
        if not rev:
            pu = pos[u] + 1
            if (tour[pu] if pu < n else tour[0]) == s2:
                move(u, s2, s1, v)
            else:
                move(v, s1, s2, u)
        push(p); push(nx); push(s1); push(s2); push(u); push(v)

    def or_opt(a):
        for L in (1, 2, 3):
            for back in (False, True):
                if L == 1 and back:
                    continue
                pa = pos[a]
                if not back:
                    i1 = pa; s1 = a; s2 = tour[(pa + L - 1) % n]
                else:
                    i1 = (pa - L + 1) % n; s1 = tour[i1]; s2 = a
                p = tour[i1 - 1]
                nx = tour[(i1 + L) % n]
                if p == nx:
                    continue
                xs1 = X[s1]; ys1 = Y[s1]; xs2 = X[s2]; ys2 = Y[s2]
                G = hyp(X[p] - xs1, Y[p] - ys1) + hyp(xs2 - X[nx], ys2 - Y[nx]) - hyp(X[p] - X[nx], Y[p] - Y[nx])
                if G <= 1e-9:
                    continue
                ends = (s1,) if L == 1 else (s1, s2)
                for end in ends:
                    nb = neigh[end]; nd = neighd[end]
                    for k in range(len(nb)):
                        if nd[k] >= G:
                            break
                        c = nb[k]
                        pcn = pos[c]
                        if (pcn - i1) % n < L:
                            continue
                        for e in (0, 1):
                            if e == 0:
                                u = c
                                v = tour[pcn + 1] if pcn + 1 < n else tour[0]
                            else:
                                u = tour[pcn - 1]; v = c
                            if v == p:
                                continue
                            if (pos[u] - i1) % n < L or (pos[v] - i1) % n < L:
                                continue
                            xu = X[u]; yu = Y[u]; xv = X[v]; yv = Y[v]
                            duv = hyp(xu - xv, yu - yv)
                            add1 = hyp(xu - xs1, yu - ys1) + hyp(xs2 - xv, ys2 - yv) - duv
                            add2 = hyp(xu - xs2, yu - ys2) + hyp(xs1 - xv, ys1 - yv) - duv
                            if add1 < add2:
                                add = add1; rev = False
                            else:
                                add = add2; rev = True
                            if G - add > 1e-9:
                                apply_or(p, s1, s2, nx, u, v, rev)
                                return G - add
        return 0.0

    tm = time.time
    def improve():
        gain = 0.0
        cnt = 0
        while q:
            cnt += 1
            if (cnt & 15) == 0 and tm() > deadline:
                break
            a = q.popleft(); inq[a] = False
            r = two_opt(a)
            if r == 0.0:
                r = or_opt(a)
            gain += r
        return gain

    def tour_len():
        s = 0.0
        prev = tour[-1]
        for c in tour:
            s += hyp(X[prev] - X[c], Y[prev] - Y[c])
            prev = c
        return s

    init = list(range(n))
    random.shuffle(init)
    for x in init:
        push(x)
    improve()
    q.clear()
    for i in range(n):
        inq[i] = False
    cur = tour_len()
    bt = tour[:]; bp = pos[:]

    while tm() < deadline:
        v = random.randrange(n)
        nb = neigh[v]
        c1 = nb[random.randrange(len(nb))]
        nb2 = neigh[c1]
        c2 = nb2[random.randrange(len(nb2))]
        qs = sorted({pos[v], pos[c1], pos[c2]})
        if len(qs) < 3:
            continue
        q1, q2, q3 = qs
        a1 = tour[q1]; b1 = tour[q1 + 1]; a2 = tour[q2]; b2 = tour[q2 + 1]
        a3 = tour[q3]; b3 = tour[(q3 + 1) % n]
        delta = (hyp(X[a1] - X[b2], Y[a1] - Y[b2]) + hyp(X[a3] - X[b1], Y[a3] - Y[b1]) +
                 hyp(X[a2] - X[b3], Y[a2] - Y[b3]) - hyp(X[a1] - X[b1], Y[a1] - Y[b1]) -
                 hyp(X[a2] - X[b2], Y[a2] - Y[b2]) - hyp(X[a3] - X[b3], Y[a3] - Y[b3]))
        new = tour[q2 + 1:q3 + 1] + tour[q1 + 1:q2 + 1]
        tour[q1 + 1:q3 + 1] = new
        k = q1 + 1
        for x in new:
            pos[x] = k; k += 1
        for x in (a1, b1, a2, b2, a3, b3):
            push(x)
        g = improve()
        if q:
            while q:
                inq[q.pop()] = False
        newlen = cur + delta - g
        if newlen <= cur + 1e-9:
            if newlen < cur - 1e-9 or True:
                cur = newlen
                bt = tour[:]; bp = pos[:]
        else:
            tour[:] = bt
            pos[:] = bp

    out = bt
    if len(set(out)) != n:
        out = list(range(n))
    sys.stdout.write(' '.join(map(str, out)) + '\n')

main()
```
