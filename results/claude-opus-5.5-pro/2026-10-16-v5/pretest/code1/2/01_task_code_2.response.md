```python
# задача 1
import sys
def main():
    s = sys.stdin.readline().strip()
    n = len(s)
    maxst = 2 * n + 2
    nxt = [None] * maxst
    link = [-1] * maxst
    length = [0] * maxst
    INF = 10 ** 9
    mn = [INF] * maxst
    mx = [-1] * maxst
    nxt[0] = {}
    sz = 1
    last = 0
    for i in range(n):
        c = s[i]
        cur = sz; sz += 1
        length[cur] = length[last] + 1
        nxt[cur] = {}
        mn[cur] = i
        mx[cur] = i
        p = last
        while p != -1 and c not in nxt[p]:
            nxt[p][c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                cl = sz; sz += 1
                length[cl] = length[p] + 1
                nxt[cl] = dict(nxt[q])
                link[cl] = link[q]
                while p != -1 and nxt[p].get(c) == q:
                    nxt[p][c] = cl
                    p = link[p]
                link[q] = cl
                link[cur] = cl
        last = cur
    # counting sort by length
    cnt = [0] * (n + 2)
    for v in range(sz):
        cnt[length[v]] += 1
    for k in range(1, n + 2):
        cnt[k] += cnt[k - 1]
    order = [0] * sz
    for v in range(sz - 1, -1, -1):
        cnt[length[v]] -= 1
        order[cnt[length[v]]] = v
    for idx in range(sz - 1, 0, -1):
        v = order[idx]
        u = link[v]
        if mn[v] < mn[u]:
            mn[u] = mn[v]
        if mx[v] > mx[u]:
            mx[u] = mx[v]
    total = 0
    for v in range(1, sz):
        d = mx[v] - mn[v]
        L = length[v]
        if d < L:
            L = d
        t = L - length[link[v]]
        if t > 0:
            total += t
    print(total)
main()
```
