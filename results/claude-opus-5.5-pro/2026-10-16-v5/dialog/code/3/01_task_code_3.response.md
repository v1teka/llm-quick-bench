```python
# задача 1
import sys, time, random, math

def main():
    t0 = time.time()
    LIMIT = 8.6
    data = sys.stdin.read().split()
    n = int(data[0]); W = int(data[1])
    ws = [int(data[2 + 2 * i]) for i in range(n)]
    hs = [int(data[3 + 2 * i]) for i in range(n)]

    def place(h, w, hh):
        b = [0]
        prev = h[0]
        for i in range(1, W):
            v = h[i]
            if v != prev:
                b.append(i)
                prev = v
        b.append(W)
        cands = []
        for s in b[:-1]:
            if s + w <= W:
                cands.append(s)
        for e in b[1:]:
            x = e - w
            if x >= 0:
                cands.append(x)
        bt = None; bw = None; bx = 0
        for x in cands:
            seg = h[x:x + w]
            top = max(seg)
            if bt is None or top < bt:
                bt = top; bw = top * w - sum(seg); bx = x
            elif top == bt:
                wa = top * w - sum(seg)
                if wa < bw or (wa == bw and x < bx):
                    bw = wa; bx = x
        nt = bt + hh
        h[bx:bx + w] = [nt] * w
        return bx, bt, nt

    def evaluate(order, i0, hstart, Hstart):
        h = hstart[:]
        H = Hstart
        states = []
        Hs = []
        poss = []
        for k in range(i0, n):
            states.append(h[:])
            Hs.append(H)
            r = order[k]
            x, y, nt = place(h, ws[r], hs[r])
            if nt > H:
                H = nt
            poss.append((x, y))
        cost = H + 0.5 * (sum(h) / float(W * H)) if H > 0 else 0
        return H, cost, states, Hs, poss

    idx = list(range(n))
    inits = [
        sorted(idx, key=lambda i: (-hs[i], -ws[i])),
        sorted(idx, key=lambda i: (-ws[i], -hs[i])),
        sorted(idx, key=lambda i: (-ws[i] * hs[i], -hs[i])),
        sorted(idx, key=lambda i: (-(ws[i] + hs[i]), -hs[i])),
        sorted(idx, key=lambda i: (-max(ws[i], hs[i]), -hs[i])),
    ]
    zero = [0] * W
    best = None
    cur = None
    for order in inits:
        H, cost, states, Hs, poss = evaluate(order, 0, zero, 0)
        if cur is None or cost < cur[1]:
            cur = (H, cost, order[:], states, Hs, poss)
    curH, curCost, curOrder, curStates, curHs, curPos = cur
    bestH = curH
    bestOrder = curOrder[:]
    bestPos = curPos[:]

    rnd = random.Random(12345)
    T0 = 1.5; T1 = 0.03
    it = 0
    T = T0
    while True:
        if (it & 7) == 0:
            el = time.time() - t0
            if el > LIMIT:
                break
            frac = el / LIMIT
            T = T0 * (T1 / T0) ** frac
        it += 1
        i = rnd.randrange(n)
        j = rnd.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        newOrder = curOrder[:]
        if rnd.random() < 0.5:
            newOrder[i], newOrder[j] = newOrder[j], newOrder[i]
        else:
            if rnd.random() < 0.5:
                e = newOrder.pop(j); newOrder.insert(i, e)
            else:
                e = newOrder.pop(i); newOrder.insert(j, e)
        H, cost, states, Hs, poss = evaluate(newOrder, i, curStates[i], curHs[i])
        d = cost - curCost
        if d <= 0 or rnd.random() < math.exp(-d / T):
            curOrder = newOrder
            curCost = cost
            curH = H
            curStates[i:] = states
            curHs[i:] = Hs
            curPos[i:] = poss
            if H < bestH:
                bestH = H
                bestOrder = curOrder[:]
                bestPos = curPos[:]

    res = [None] * n
    for k in range(n):
        res[bestOrder[k]] = bestPos[k]
    out = []
    for i in range(n):
        out.append("%d %d" % res[i])
    sys.stdout.write("\n".join(out) + "\n")

main()
```
