```python
# задача 1
import sys, random, time

def main():
    t0 = time.time()
    deadline = t0 + 9.0
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); m = int(data[1])
    adjs = [set() for _ in range(n)]
    idx = 2
    for _ in range(m):
        if idx + 1 >= len(data) + 0 and idx + 1 > len(data) - 1:
            if idx + 1 > len(data) - 1:
                break
        u = int(data[idx]); v = int(data[idx + 1]); idx += 2
        if u != v:
            adjs[u].add(v); adjs[v].add(u)
    adj = [list(s) for s in adjs]
    random.seed(12345)

    # DSATUR
    col = [-1] * n
    sat = [set() for _ in range(n)]
    deg = [len(a) for a in adj]
    uncol = set(range(n))
    for _ in range(n):
        bv = -1; bs = -1; bd = -1
        for v in uncol:
            s = len(sat[v])
            if s > bs or (s == bs and deg[v] > bd):
                bv = v; bs = s; bd = deg[v]
        v = bv
        c = 0
        sv = sat[v]
        while c in sv:
            c += 1
        col[v] = c
        uncol.discard(v)
        for u in adj[v]:
            if col[u] < 0:
                sat[u].add(c)
    best_col = col[:]
    best_k = (max(col) + 1) if n > 0 else 0

    def valid(cl):
        for v in range(n):
            cv = cl[v]
            for u in adj[v]:
                if cl[u] == cv:
                    return False
        return True

    def reduce_coloring(cl, k):
        # cl uses colors 0..k-1 ; produce coloring with k-1 colors (may have conflicts)
        cnt = [0] * k
        for c in cl:
            cnt[c] += 1
        rem = min(range(k), key=lambda c: cnt[c])
        mp = {}
        j = 0
        for c in range(k):
            if c != rem:
                mp[c] = j; j += 1
        newc = [-1] * n
        for v in range(n):
            if cl[v] != rem:
                newc[v] = mp[cl[v]]
        order = [v for v in range(n) if cl[v] == rem]
        random.shuffle(order)
        for v in order:
            g = [0] * (k - 1)
            for u in adj[v]:
                cu = newc[u]
                if cu >= 0:
                    g[cu] += 1
            mn = min(g)
            cands = [c for c in range(k - 1) if g[c] == mn]
            newc[v] = random.choice(cands)
        return newc

    def tabucol(cl, k):
        col = cl[:]
        gamma = [[0] * k for _ in range(n)]
        for v in range(n):
            gv = gamma[v]
            for u in adj[v]:
                gv[col[u]] += 1
        f = 0
        for v in range(n):
            f += gamma[v][col[v]]
        f //= 2
        conf = []
        pos = [-1] * n
        for v in range(n):
            if gamma[v][col[v]] > 0:
                pos[v] = len(conf); conf.append(v)
        tabu = [[0] * k for _ in range(n)]
        bestf = f
        it = 0
        BIG = 1 << 30
        rnd = random.random
        rint = random.randint
        while True:
            if f == 0:
                return col
            it += 1
            if (it & 31) == 0 and time.time() > deadline:
                return None
            bestd = BIG
            moves = []
            for v in conf:
                cv = col[v]; gv = gamma[v]; base = gv[cv]; tv = tabu[v]
                c = -1
                for x in gv:
                    c += 1
                    d = x - base
                    if d <= bestd and c != cv:
                        if tv[c] <= it or f + d < bestf:
                            if d < bestd:
                                bestd = d; moves = [(v, c)]
                            else:
                                moves.append((v, c))
            if moves:
                v, c = moves[int(rnd() * len(moves))]
            else:
                v = conf[int(rnd() * len(conf))]
                c = int(rnd() * k)
                if c == col[v]:
                    c = (c + 1) % k
                bestd = gamma[v][c] - gamma[v][col[v]]
            old = col[v]
            col[v] = c
            f += bestd
            for u in adj[v]:
                gu = gamma[u]
                gu[old] -= 1
                gu[c] += 1
                cu = col[u]
                if cu == old:
                    if gu[old] == 0:
                        p = pos[u]; last = conf[-1]
                        conf[p] = last; pos[last] = p
                        conf.pop(); pos[u] = -1
                elif cu == c:
                    if gu[c] == 1:
                        pos[u] = len(conf); conf.append(u)
            gvv = gamma[v]
            if gvv[c] > 0:
                if pos[v] < 0:
                    pos[v] = len(conf); conf.append(v)
            else:
                if pos[v] >= 0:
                    p = pos[v]; last = conf[-1]
                    conf[p] = last; pos[last] = p
                    conf.pop(); pos[v] = -1
            tabu[v][old] = it + rint(0, 9) + int(0.6 * len(conf))
            if f < bestf:
                bestf = f

    cur = best_col[:]
    k = best_k
    while k > 1 and time.time() < deadline:
        start = reduce_coloring(cur, k)
        res = tabucol(start, k - 1)
        if res is None:
            break
        if valid(res):
            k -= 1
            cur = res
            best_col = res[:]
            best_k = k
        else:
            break

    if not valid(best_col):
        best_col = list(range(n))
    sys.stdout.write(" ".join(map(str, best_col)) + "\n")

main()
```
