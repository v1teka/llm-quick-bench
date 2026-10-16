```python
# задача 1
import sys
from math import gcd
def main():
    data = sys.stdin.buffer.read().split()
    t = int(data[0]); pos = 1
    out = []
    for _ in range(t):
        n = int(data[pos]); k = int(data[pos+1]); s = data[pos+2]; pos += 3
        sv = [c - 48 for c in s]
        d = [sv[j] ^ sv[j-1] for j in range(n)]
        g = gcd(n, k); L = n // g
        x = [0] * n
        ok = True
        total = 0
        flips = 0
        bestdiff = n + 5
        for c in range(g):
            j = c; cur = 0; cnt = 0
            for _ in range(L - 1):
                j += k
                if j >= n:
                    j -= n
                cur ^= d[j]
                x[j] = cur
                cnt += cur
            if cur ^ d[c]:
                ok = False
                break
            a = cnt; b = L - cnt
            if a <= b:
                total += a
            else:
                total += b
                flips ^= 1
            df = abs(a - b)
            if df < bestdiff:
                bestdiff = df
        if not ok:
            out.append("-1"); continue
        w = (sum(x[n-k+1:]) + x[0]) & 1
        if (k // g) % 2 == 0:
            if w != sv[0]:
                out.append("-1")
            else:
                out.append(str(total))
        else:
            if (w ^ flips) != sv[0]:
                total += bestdiff
            out.append(str(total))
    sys.stdout.write("\n".join(out) + "\n")
main()
```

```python
# задача 2
import sys
def main():
    s = sys.stdin.buffer.read().split()[0]
    N = len(s)
    maxst = 2 * N + 2
    nxt = [-1] * (26 * maxst)
    link = [-1] * maxst
    ln = [0] * maxst
    INF = 1 << 30
    mn = [INF] * maxst
    mx = [-1] * maxst
    sz = 1; last = 0
    for i in range(N):
        c = s[i] - 97
        cur = sz; sz += 1
        ln[cur] = ln[last] + 1
        mn[cur] = i; mx[cur] = i
        p = last
        while p != -1 and nxt[p*26+c] == -1:
            nxt[p*26+c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p*26+c]
            if ln[p] + 1 == ln[q]:
                link[cur] = q
            else:
                cl = sz; sz += 1
                ln[cl] = ln[p] + 1
                nxt[cl*26:cl*26+26] = nxt[q*26:q*26+26]
                link[cl] = link[q]
                while p != -1 and nxt[p*26+c] == q:
                    nxt[p*26+c] = cl
                    p = link[p]
                link[q] = cl
                link[cur] = cl
        last = cur
    cnt = [0] * (N + 2)
    for v in range(sz):
        cnt[ln[v]] += 1
    for i in range(1, N + 2):
        cnt[i] += cnt[i-1]
    order = [0] * sz
    for v in range(sz - 1, -1, -1):
        cnt[ln[v]] -= 1
        order[cnt[ln[v]]] = v
    for idx in range(sz - 1, 0, -1):
        v = order[idx]
        l = link[v]
        if mn[v] < mn[l]: mn[l] = mn[v]
        if mx[v] > mx[l]: mx[l] = mx[v]
    ans = 0
    for v in range(1, sz):
        diff = mx[v] - mn[v]
        top = ln[v] if ln[v] < diff else diff
        val = top - ln[link[v]]
        if val > 0:
            ans += val
    print(ans)
main()
```

```python
# задача 3
import sys
from math import isqrt
from operator import mul
from itertools import accumulate
def main():
    n = int(sys.stdin.readline())
    MOD = 998244353
    L = min(n, 2000000)
    r0 = isqrt(L)
    comp = bytearray(r0 + 1)
    ps = []
    for i in range(2, r0 + 1):
        if not comp[i]:
            ps.append(i)
            comp[i*i::i] = b'\x01' * len(range(i*i, r0 + 1, i))
    spf = [0] * (L + 1)
    for p in reversed(ps):
        spf[p*p::p] = [p] * len(range(p*p, L + 1, p))
    phi = [0] * (L + 1)
    phi[1] = 1
    for i in range(2, L + 1):
        p = spf[i]
        if p == 0:
            phi[i] = i - 1
        else:
            j = i // p
            if j % p:
                phi[i] = phi[j] * (p - 1)
            else:
                phi[i] = phi[j] * p
    pre = list(accumulate(phi))
    K = n // (L + 1)
    big = [0] * (K + 1)
    for k in range(K, 0, -1):
        v = n // k
        r = isqrt(v)
        m = min(r, K // k)
        s = sum(big[2*k: k*m + 1: k]) if m >= 2 else 0
        a = max(m + 1, 2)
        if a <= r:
            s += sum([pre[v // d] for d in range(a, r + 1)])
        Q = v // (r + 1)
        s += sum(map(mul, [v // qq for qq in range(1, Q + 1)], phi[1:Q + 1])) - r * pre[Q]
        big[k] = (v * (v + 1) // 2 - s) % MOD
    def sq(x):
        return x * (x + 1) * (2 * x + 1) // 6
    res = 0
    d = 1
    while d <= n:
        q = n // d
        d2 = n // q
        ph = big[d] if q > L else pre[q] % MOD
        res = (res + (2 * ph - 1) * ((sq(d2) - sq(d - 1)) % MOD)) % MOD
        d = d2 + 1
    print(res % MOD)
main()
```

```python
# задача 4
import sys
def main():
    data = sys.stdin.buffer.read().split()
    MOD = 998244353
    n = int(data[0]); m = int(data[1])
    inv100 = pow(100, MOD - 2, MOD)
    q = [(100 - int(data[2 + i])) * inv100 % MOD for i in range(n)]
    base = 2 + n
    U = [int(data[base + 2*i]) - 1 for i in range(m)]
    V = [int(data[base + 2*i + 1]) - 1 for i in range(m)]
    ans = sum(q)
    se = 0
    for i in range(m):
        se += q[U[i]] * q[V[i]]
    ans = (ans - se) % MOD
    deg = [0] * (n + 1)
    for u in U: deg[u + 1] += 1
    for v in V: deg[v + 1] += 1
    for i in range(n):
        deg[i + 1] += deg[i]
    start = deg[:]
    adj = [0] * (2 * m)
    fill = deg[:n]
    for i in range(m):
        u = U[i]; v = V[i]
        adj[fill[u]] = v; fill[u] += 1
        adj[fill[v]] = u; fill[v] += 1
    ptr = start[:n]
    end = start[1:n + 1]
    depth = [-1] * n
    par = [-1] * n
    for s0 in range(n):
        if depth[s0] != -1:
            continue
        depth[s0] = 0
        stack = [s0]
        while stack:
            u = stack[-1]
            i = ptr[u]
            if i < end[u]:
                ptr[u] = i + 1
                w = adj[i]
                dw = depth[w]
                if dw == -1:
                    depth[w] = depth[u] + 1
                    par[w] = u
                    stack.append(w)
                elif dw < depth[u] - 1:
                    prod = q[u]
                    x = u
                    while x != w:
                        x = par[x]
                        prod = prod * q[x] % MOD
                    ans += prod
            else:
                stack.pop()
    print(ans % MOD)
main()
```
