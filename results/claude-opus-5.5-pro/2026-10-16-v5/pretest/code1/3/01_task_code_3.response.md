```python
# задача 1
import sys
from math import isqrt
from itertools import accumulate

def main():
    n = int(sys.stdin.readline())
    M = 998244353
    L = min(n, 2500000)
    if L < 2:
        L = 2
    spf = [0] * (L + 1)
    r = isqrt(L)
    sm = bytearray([1]) * (r + 1)
    primes = []
    for i in range(2, r + 1):
        if sm[i]:
            primes.append(i)
            for j in range(i * i, r + 1, i):
                sm[j] = 0
    for p in reversed(primes):
        st = p * p
        cnt = len(range(st, L + 1, p))
        spf[st::p] = [p] * cnt
    phi = [0] * (L + 1)
    phi[1] = 1
    for i in range(2, L + 1):
        p = spf[i]
        if p == 0:
            phi[i] = i - 1
        else:
            j = i // p
            if j % p == 0:
                phi[i] = phi[j] * p
            else:
                phi[i] = phi[j] * (p - 1)
    P = list(accumulate(phi))
    K = n // L
    memo = [0] * (K + 2)
    for k in range(K, 0, -1):
        m = n // k
        rr = isqrt(m)
        s = 0
        for d in range(2, rr + 1):
            q = m // d
            if q <= L:
                s += P[q]
            else:
                s += memo[k * d]
        top = m // (rr + 1)
        for q in range(1, top + 1):
            s += P[q] * (m // q - m // (q + 1))
        memo[k] = (m * (m + 1) // 2 - s) % M

    def Phi(q, l):
        if q <= L:
            return P[q] % M
        return memo[l]

    def sq(x):
        return x * (x + 1) * (2 * x + 1) // 6

    res = 0
    l = 1
    while l <= n:
        q = n // l
        rgt = n // q
        f = Phi(q, l)
        res = (res + (sq(rgt) - sq(l - 1)) % M * ((2 * f - 1) % M)) % M
        l = rgt + 1
    print(res % M)

main()
```
