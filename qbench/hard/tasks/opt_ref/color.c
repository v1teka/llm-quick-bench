// Эталон для «Раскраски графа»: TabuCol (Hertz, de Werra) с уменьшением числа цветов.
// gcc -O2 -o color color.c && ./color SECONDS SEED < input  -> число цветов и раскраска
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int n, m, *deg, **adj;
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)rs; }
int tabucol(int k, int *col, double deadline) {
  int *gam = calloc((size_t)n * k, sizeof(int)); long long *tabu = calloc((size_t)n * k, sizeof(long long));
  for (int v = 0; v < n; v++) col[v] = rnd() % k;
  int f = 0;
  for (int v = 0; v < n; v++) for (int j = 0; j < deg[v]; j++) { int u = adj[v][j]; gam[v * k + col[u]]++; if (u > v && col[u] == col[v]) f++; }
  long long it = 0; int bestf = f;
  while (f > 0) {
    if ((it & 1023) == 0 && (double)clock() / CLOCKS_PER_SEC > deadline) break;
    it++;
    int bd = 1 << 30, bv = -1, bc = -1, cnt = 0;
    for (int v = 0; v < n; v++) {
      int cv = col[v]; int g0 = gam[v * k + cv]; if (g0 == 0) continue;
      for (int c = 0; c < k; c++) if (c != cv) {
        int d = gam[v * k + c] - g0;
        if (tabu[v * k + c] > it && f + d >= bestf) continue;
        if (d < bd) { bd = d; bv = v; bc = c; cnt = 1; }
        else if (d == bd && rnd() % (++cnt) == 0) { bv = v; bc = c; }
      }
    }
    if (bv < 0) continue;
    int old = col[bv];
    for (int j = 0; j < deg[bv]; j++) { int u = adj[bv][j]; gam[u * k + old]--; gam[u * k + bc]++; }
    col[bv] = bc; f += bd;
    int ncf = 0; for (int v = 0; v < n; v++) if (gam[v * k + col[v]] > 0) ncf++;
    tabu[bv * k + old] = it + (long long)(0.6 * f) + rnd() % 10;
    if (f < bestf) bestf = f;
  }
  free(gam); free(tabu);
  return f == 0;
}
int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 60; if (argc > 2) rs += atoi(argv[2]) * 7919ULL;
  if (scanf("%d %d", &n, &m) != 2) return 1;
  int *U = malloc(sizeof(int) * m), *V = malloc(sizeof(int) * m); deg = calloc(n, sizeof(int));
  for (int e = 0; e < m; e++) { if (scanf("%d %d", &U[e], &V[e]) != 2) return 1; deg[U[e]]++; deg[V[e]]++; }
  adj = malloc(sizeof(int *) * n); int *fill = calloc(n, sizeof(int));
  for (int v = 0; v < n; v++) adj[v] = malloc(sizeof(int) * (deg[v] + 1));
  for (int e = 0; e < m; e++) { adj[U[e]][fill[U[e]]++] = V[e]; adj[V[e]][fill[V[e]]++] = U[e]; }
  int *col = malloc(sizeof(int) * n), *best = malloc(sizeof(int) * n);
  // старт: жадная раскраска
  for (int v = 0; v < n; v++) { char used[4096] = {0}; for (int j = 0; j < deg[v]; j++) if (adj[v][j] < v) used[best[adj[v][j]]] = 1; int c = 0; while (used[c]) c++; best[v] = c; }
  int K = 0; for (int v = 0; v < n; v++) if (best[v] + 1 > K) K = best[v] + 1;
  double deadline = (double)clock() / CLOCKS_PER_SEC + secs;
  for (int k = K - 1; k >= 1; k--) {
    if (!tabucol(k, col, deadline)) break;
    K = k; memcpy(best, col, sizeof(int) * n);
    fprintf(stderr, "%d colors at %.1fs\n", k, (double)clock() / CLOCKS_PER_SEC);
  }
  printf("%d\n", K);
  for (int v = 0; v < n; v++) printf("%d%c", best[v], v + 1 < n ? ' ' : '\n');
  return 0;
}
