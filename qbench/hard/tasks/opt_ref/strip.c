// Эталон для «Упаковки в полосу»: нижне-левая укладка с заполнением дыр и локальный поиск по порядку укладки (перестановки, отжиг).
// gcc -O2 -o strip strip.c && ./strip SECONDS SEED < input  -> высота и координаты
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
int n, W, *w, *h;
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)rs; }
typedef unsigned __int128 u128;
static u128 occ[1 << 14];
// нижне-левая укладка с дырами: каждый прямоугольник — в самую низкую, затем самую левую свободную позицию
int place(int *ord, int *X, int *Y) {
  int H = 0;
  u128 full = W >= 128 ? ~(u128)0 : (((u128)1 << W) - 1);
  for (int t = 0; t < n; t++) {
    int i = ord[t];
    for (int y = 0;; y++) {
      if (y + h[i] > H) for (int q = H; q < y + h[i]; q++) occ[q] = 0;
      u128 fr = full;
      for (int q = y; q < y + h[i] && fr; q++) fr &= ~occ[q];
      u128 m = fr;  // бит x: свободны x … x+w−1
      for (int k = 1; k < w[i] && m; k++) m &= fr >> k;
      if (m) {
        int x = 0; while (!((m >> x) & 1)) x++;
        u128 bits = (((u128)1 << w[i]) - 1) << x;
        if (y + h[i] > H) H = y + h[i];
        for (int q = y; q < y + h[i]; q++) occ[q] |= bits;
        X[i] = x; Y[i] = y; break;
      }
    }
  }
  return H;
}
int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 60; if (argc > 2) rs += atoi(argv[2]) * 7919ULL;
  if (scanf("%d %d", &n, &W) != 2) return 1;
  w = malloc(sizeof(int) * n); h = malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) if (scanf("%d %d", &w[i], &h[i]) != 2) return 1;
  int *ord = malloc(sizeof(int) * n), *X = malloc(sizeof(int) * n), *Y = malloc(sizeof(int) * n);
  int *bX = malloc(sizeof(int) * n), *bY = malloc(sizeof(int) * n);
  for (int i = 0; i < n; i++) ord[i] = i;
  for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (h[ord[b]] > h[ord[a]]) { int t = ord[a]; ord[a] = ord[b]; ord[b] = t; }
  int cur = place(ord, X, Y), best = cur; memcpy(bX, X, sizeof(int) * n); memcpy(bY, Y, sizeof(int) * n);
  clock_t st = clock(); double T0 = 3.0;
  while (1) {
    double el = (double)(clock() - st) / CLOCKS_PER_SEC; if (el > secs) break;
    double T = T0 * (1 - el / secs) + 0.05;
    int a = rnd() % n, b = rnd() % n; if (a == b) continue;
    int mode = rnd() % 2, tmp;
    if (mode == 0) { tmp = ord[a]; ord[a] = ord[b]; ord[b] = tmp; }
    else { // перенос a -> b
      tmp = ord[a];
      if (a < b) memmove(ord + a, ord + a + 1, sizeof(int) * (b - a)); else memmove(ord + b + 1, ord + b, sizeof(int) * (a - b));
      ord[b] = tmp;
    }
    int v = place(ord, X, Y);
    if (v <= cur || exp((cur - v) / T) > (rnd() % 1000000) / 1e6) {
      cur = v;
      if (v < best) { best = v; memcpy(bX, X, sizeof(int) * n); memcpy(bY, Y, sizeof(int) * n); }
    } else { // откат
      if (mode == 0) { tmp = ord[a]; ord[a] = ord[b]; ord[b] = tmp; }
      else {
        tmp = ord[b];
        if (a < b) memmove(ord + a + 1, ord + a, sizeof(int) * (b - a)); else memmove(ord + b, ord + b + 1, sizeof(int) * (a - b));
        ord[a] = tmp;
      }
    }
  }
  printf("%d\n", best);
  for (int i = 0; i < n; i++) printf("%d %d\n", bX[i], bY[i]);
  return 0;
}
