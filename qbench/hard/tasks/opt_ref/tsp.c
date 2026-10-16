// Эталон для «Коммивояжёра»: итеративный локальный поиск (2-opt + Or-opt по спискам соседей,
// возмущение double-bridge). Ввод как у задачи, вывод — длина лучшего тура и тур.
// gcc -O2 -o tsp tsp.c -lm && ./tsp SECONDS SEED < input
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#define K 10
int n; double *X, *Y; int *nb; int *tour, *pos, *best;
static double D(int a, int b) { double dx = X[a] - X[b], dy = Y[a] - Y[b]; return sqrt(dx * dx + dy * dy); }
static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)rs; }
static double len(int *t) { double s = 0; for (int i = 0; i < n; i++) s += D(t[i], t[(i + 1) % n]); return s; }
static void rev(int i, int j) { // развернуть tour[i..j] (по кругу)
  int L = (j - i + n) % n + 1;
  for (int k = 0; k < L / 2; k++) {
    int a = (i + k) % n, b = (j - k + n) % n;
    int t = tour[a]; tour[a] = tour[b]; tour[b] = t; pos[tour[a]] = a; pos[tour[b]] = b;
  }
}
static int two_opt() {
  int improved = 1, any = 0;
  while (improved) {
    improved = 0;
    for (int i = 0; i < n; i++) {
      int a = tour[i], b = tour[(i + 1) % n];
      for (int k = 0; k < K; k++) {
        int c = nb[a * K + k];
        double g = D(a, b) - D(a, c);
        if (g <= 1e-9) break;
        int pc = pos[c], d = tour[(pc + 1) % n];
        if (c == b || d == a) continue;
        double delta = D(a, c) + D(b, d) - D(a, b) - D(c, d);
        if (delta < -1e-9) {
          int pi = pos[a];
          int i1 = (pi + 1) % n, j1 = pc;
          if ((j1 - i1 + n) % n > n / 2) { i1 = (pc + 1) % n; j1 = pi; }
          rev(i1, j1); improved = 1; any = 1; break;
        }
      }
      // в обратную сторону
      a = tour[i]; b = tour[(i - 1 + n) % n];
      for (int k = 0; k < K; k++) {
        int c = nb[a * K + k];
        double g = D(a, b) - D(a, c);
        if (g <= 1e-9) break;
        int pc = pos[c], d = tour[(pc - 1 + n) % n];
        if (c == b || d == a) continue;
        double delta = D(a, c) + D(b, d) - D(a, b) - D(c, d);
        if (delta < -1e-9) {
          int pi = pos[a];
          int i1 = pc, j1 = (pi - 1 + n) % n;
          if ((j1 - i1 + n) % n > n / 2) { i1 = pi; j1 = (pc - 1 + n) % n; }
          rev(i1, j1); improved = 1; any = 1; break;
        }
      }
    }
  }
  return any;
}
static int or_opt() { // перенос отрезка из 1..3 городов к соседу
  int any = 0, improved = 1;
  while (improved) {
    improved = 0;
    for (int seg = 1; seg <= 3; seg++)
      for (int i = 0; i < n; i++) {
        int s0 = tour[i], s1 = tour[(i + seg - 1) % n];
        int p = tour[(i - 1 + n) % n], q = tour[(i + seg) % n];
        double rem = D(p, s0) + D(s1, q) - D(p, q);
        for (int k = 0; k < K; k++) {
          int c = nb[s0 * K + k];
          int pc = pos[c];
          int inside = (pc - i + n) % n < seg; if (inside || c == p) continue;
          int d = tour[(pc + 1) % n]; if ((pos[d] - i + n) % n < seg) continue;
          // вставка между c и d, отрезок развёрнут (s1 рядом с d? нет: c—s0 … s1—d)
          double add1 = D(c, s1) + D(s0, d) - D(c, d); // c-s1..s0-d
          double add2 = D(c, s0) + D(s1, d) - D(c, d);
          double add = add1 < add2 ? add1 : add2;
          if (add < rem - 1e-9) {
            // перестроить массив тура
            int *seq = malloc(sizeof(int) * seg); for (int t = 0; t < seg; t++) seq[t] = tour[(i + t) % n];
            int *nt = malloc(sizeof(int) * n); int m = 0;
            for (int t = 0; t < n; t++) {
              int v = tour[(i + seg + t) % n]; if (t >= n - seg) break;
              nt[m++] = v;
              if (v == c) {
                if (add1 < add2) for (int u = seg - 1; u >= 0; u--) nt[m++] = seq[u];
                else for (int u = 0; u < seg; u++) nt[m++] = seq[u];
              }
            }
            memcpy(tour, nt, sizeof(int) * n); for (int t = 0; t < n; t++) pos[tour[t]] = t;
            free(seq); free(nt); improved = 1; any = 1; break;
          }
        }
      }
  }
  return any;
}
static void local() { while (two_opt() | or_opt()); }
int main(int argc, char **argv) {
  double secs = argc > 1 ? atof(argv[1]) : 30; if (argc > 2) rs += atoi(argv[2]) * 7919ULL;
  if (scanf("%d", &n) != 1) return 1;
  X = malloc(sizeof(double) * n); Y = malloc(sizeof(double) * n);
  for (int i = 0; i < n; i++) scanf("%lf %lf", &X[i], &Y[i]);
  nb = malloc(sizeof(int) * n * K);
  for (int i = 0; i < n; i++) { // K ближайших
    double bd[K]; int bi[K]; int c = 0;
    for (int j = 0; j < n; j++) if (j != i) {
      double d = D(i, j);
      if (c < K) { bd[c] = d; bi[c] = j; c++; }
      else if (d < bd[K - 1]) { bd[K - 1] = d; bi[K - 1] = j; }
      else continue;
      for (int t = c - 1; t > 0 && bd[t] < bd[t - 1]; t--) { double x = bd[t]; bd[t] = bd[t - 1]; bd[t - 1] = x; int y = bi[t]; bi[t] = bi[t - 1]; bi[t - 1] = y; }
    }
    for (int k = 0; k < K; k++) nb[i * K + k] = bi[k];
  }
  tour = malloc(sizeof(int) * n); pos = malloc(sizeof(int) * n); best = malloc(sizeof(int) * n);
  char *used = calloc(n, 1); int cur = 0; used[0] = 1; tour[0] = 0;
  for (int t = 1; t < n; t++) { int bj = -1; double bd = 1e300; for (int j = 0; j < n; j++) if (!used[j] && D(cur, j) < bd) { bd = D(cur, j); bj = j; } tour[t] = bj; used[bj] = 1; cur = bj; }
  for (int t = 0; t < n; t++) pos[tour[t]] = t;
  local(); double bl = len(tour); memcpy(best, tour, sizeof(int) * n);
  clock_t st = clock();
  int *tmp = malloc(sizeof(int) * n);
  while ((double)(clock() - st) / CLOCKS_PER_SEC < secs) {
    // double-bridge на локальном участке
    int L = 30 + rnd() % 70; int s = rnd() % n;
    int a = 1 + rnd() % (L - 3), b = a + 1 + rnd() % (L - a - 2), c = b + 1 + rnd() % (L - b - 1);
    int m = 0;
    for (int t = 0; t < a; t++) tmp[m++] = tour[(s + t) % n];
    for (int t = b; t < c; t++) tmp[m++] = tour[(s + t) % n];
    for (int t = a; t < b; t++) tmp[m++] = tour[(s + t) % n];
    for (int t = c; t < L; t++) tmp[m++] = tour[(s + t) % n];
    for (int t = 0; t < L; t++) { tour[(s + t) % n] = tmp[t]; pos[tmp[t]] = (s + t) % n; }
    local();
    double l = len(tour);
    if (l < bl - 1e-9) { bl = l; memcpy(best, tour, sizeof(int) * n); }
    else { memcpy(tour, best, sizeof(int) * n); for (int t = 0; t < n; t++) pos[tour[t]] = t; }
  }
  printf("%.3f\n", bl);
  for (int i = 0; i < n; i++) printf("%d%c", best[i], i + 1 < n ? ' ' : '\n');
  return 0;
}
