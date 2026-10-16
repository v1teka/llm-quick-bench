```python
# задача 1
import sys
def main():
    data=sys.stdin.buffer.read().split()
    MOD=998244353
    n=int(data[0]); m=int(data[1])
    inv100=pow(100,MOD-2,MOD)
    q=[(100-int(data[2+i]))*inv100%MOD for i in range(n)]
    ans=sum(q)%MOD
    adj=[[] for _ in range(n)]
    pos=2+n
    es=0
    for _ in range(m):
        u=int(data[pos])-1; v=int(data[pos+1])-1; pos+=2
        adj[u].append(v); adj[v].append(u)
        es+=q[u]*q[v]
    ans=(ans-es)%MOD
    depth=[-1]*n
    parent=[-1]*n
    idx=[0]*n
    for s in range(n):
        if depth[s]!=-1: continue
        depth[s]=0
        stack=[s]
        while stack:
            u=stack[-1]
            au=adj[u]
            if idx[u]<len(au):
                w=au[idx[u]]; idx[u]+=1
                if depth[w]==-1:
                    depth[w]=depth[u]+1; parent[w]=u; stack.append(w)
                elif w!=parent[u] and depth[w]<depth[u]:
                    prod=q[w]; x=u
                    while x!=w:
                        prod=prod*q[x]%MOD
                        x=parent[x]
                    ans=(ans+prod)%MOD
            else:
                stack.pop()
    print(ans%MOD)
main()
```
