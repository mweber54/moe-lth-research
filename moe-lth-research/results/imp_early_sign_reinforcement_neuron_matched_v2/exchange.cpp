#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// The proposals use only selected edges, eligibility and covariates.
extern "C" void exchange(int rows, int cols, const int *stratum,
                         const double *energy, int *edges, int count,
                         uint64_t seed, int rounds) {
    if (count < 2) return;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> pick(0, count - 1);
    std::uniform_real_distribution<double> uniform(0., 1.);
    std::vector<bool> selected(rows * cols);
    std::vector<double> target_row(rows), target_col(cols);
    std::vector<double> row_error(rows), col_error(cols);
    for (int i = 0; i < count; ++i) {
        int edge = edges[i];
        selected[edge] = true;
        target_row[edge / cols] += energy[edge];
        target_col[edge % cols] += energy[edge];
    }
    auto cost = [](double error, double change, double target) {
        double before = error / target;
        double after = (error + change) / target;
        return (after * after - before * before) +
               8 * (std::pow(after, 4) - std::pow(before, 4));
    };
    for (int round = 0; round < rounds; ++round) {
        int i = pick(rng), j = pick(rng);
        int a = edges[i], b = edges[j];
        int r = a / cols, s = b / cols, c = a % cols, d = b % cols;
        int x = r * cols + d, y = s * cols + c;
        if (selected[x] || selected[y] || stratum[x] < 0 || stratum[y] < 0) continue;
        if (!((stratum[a] == stratum[x] && stratum[b] == stratum[y]) ||
              (stratum[a] == stratum[y] && stratum[b] == stratum[x]))) continue;
        double dr = energy[x] - energy[a], ds = energy[y] - energy[b];
        double dc = energy[y] - energy[a], dd = energy[x] - energy[b];
        double delta = cost(row_error[r], dr, target_row[r]) + cost(row_error[s], ds, target_row[s]) +
                       cost(col_error[c], dc, target_col[c]) + cost(col_error[d], dd, target_col[d]);
        double phase = static_cast<double>(round) / rounds;
        double temperature = .01 * std::exp(-4 * phase) + .0001;
        if (delta > 0 && uniform(rng) >= std::exp(-delta / temperature)) continue;
        selected[a] = selected[b] = false;
        selected[x] = selected[y] = true;
        edges[i] = x;
        edges[j] = y;
        row_error[r] += dr;
        row_error[s] += ds;
        col_error[c] += dc;
        col_error[d] += dd;
    }
}
