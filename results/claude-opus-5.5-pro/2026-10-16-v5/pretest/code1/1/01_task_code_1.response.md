```python
# задача 1
import sys
from math import gcd

def main():
    data = sys.stdin.buffer.read().split()
    t = int(data[0]); p = 1
    out = []
    for _ in range(t):
        n = int(data[p]); k = int(data[p+1]); s = data[p+2]; p += 3
        sv = [c - 48 for c in s]
        d = [sv[i] ^ sv[i-1] for i in range(n)]
        g = gcd(n, k)
        L = n // g
        x = [0] * n
        ok = True
        total = 0
        best_fix = None
        cs = []
        for r in range(g):
            cur = 0
            j = r
            cnt = 0
            for _ in range(L):
                x[j] = cur
                cnt += cur
                nj = j + k
                if nj >= n:
                    nj -= n
                cur ^= d[nj]
                j = nj
            # after L steps, j == r, cur should equal x[r] (=0)
            if cur != 0:
                ok = False
                break
            cs.append(cnt)
        if not ok:
            out.append("-1")
            continue
        P = 0
        for i in range(k):
            P ^= x[i]
        target = sv[k-1]
        if (k // g) % 2 == 0:
            if P != target:
                out.append("-1")
                continue
            total = 0
            for c in cs:
                total += c if c < L - c else L - c
            out.append(str(total))
        else:
            need = P ^ target
            total = 0
            flips = 0
            fix = None
            for c in cs:
                o = L - c
                if o < c:
                    total += o
                    flips ^= 1
                else:
                    total += c
                diff = abs(c - o)
                if fix is None or diff < fix:
                    fix = diff
            if flips != need:
                total += fix
            out.append(str(total))
    sys.stdout.write("\n".join(out) + "\n")

main()
```
