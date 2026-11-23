#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Corrected, separately named implementation. Each proposal explicitly uses
// all four corners of one alternating rectangle, and all bookkeeping is based
// on the indices actually returned.
extern "C" int rectangles_validated(int rows, int cols, const int *stratum,
                                     const double *energy, const double *activity,
                                     const int *high, int n_high, double target,
                                     int *output_high, int *output_control,
                                     uint64_t seed, int proposals) {
    constexpr int strata = 128;
    if (!n_high || target <= 0) return 0;
    std::mt19937_64 rng(seed);
    std::vector<std::vector<int>> row_groups(rows * strata), col_groups(cols * strata);
    std::vector<std::vector<int>> high_rows(rows), high_cols(cols);
    std::vector<bool> used_high(rows * cols), used_control(rows * cols);
    std::vector<double> high_row(rows), high_col(cols), control_row(rows), control_col(cols);
    for (int i = 0; i < rows * cols; ++i) {
        if (stratum[i] < 0) continue;
        row_groups[(i / cols) * strata + stratum[i]].push_back(i);
        col_groups[(i % cols) * strata + stratum[i]].push_back(i);
    }
    for (int i = 0; i < n_high; ++i) {
        high_rows[high[i] / cols].push_back(high[i]);
        high_cols[high[i] % cols].push_back(high[i]);
    }
    auto balanced = [](double a, double b, double limit) {
        return std::abs(a - b) <= limit * std::max(a, b);
    };
    int count = 0;
    double high_energy = 0., control_energy = 0.;
    double high_activity = 0., control_activity = 0.;
    double high_magnitude = 0., control_magnitude = 0.;
    for (int trial = 0; trial < proposals && high_energy < target; ++trial) {
        int a = high[rng() % n_high];
        if (used_high[a]) continue;
        int r = a / cols, c = a % cols;
        bool row = trial % 2 == 0;
        const auto &neighbors = row ? row_groups[r * strata + stratum[a]]
                                    : col_groups[c * strata + stratum[a]];
        if (neighbors.empty()) continue;
        int x = neighbors[rng() % neighbors.size()];
        if (x == a) continue;
        const auto &anchors = row ? high_cols[x % cols] : high_rows[x / cols];
        if (anchors.empty()) continue;
        int b = anchors[rng() % anchors.size()];
        int s = b / cols, d = b % cols;
        if (r == s || c == d || used_high[b]) continue;
        // Explicit opposite corners. x must coincide with the appropriate one.
        int cross_r = r * cols + d;
        int cross_s = s * cols + c;
        int expected_x = row ? cross_r : cross_s;
        int y = row ? cross_s : cross_r;
        if (x != expected_x || stratum[y] != stratum[b] ||
            used_control[x] || used_control[y]) continue;
        double ea = energy[a], eb = energy[b], ex = energy[x], ey = energy[y];
        double ha = activity[a] + activity[b], ca = activity[x] + activity[y];
        double hm = std::sqrt(ea) + std::sqrt(eb), cm = std::sqrt(ex) + std::sqrt(ey);
        if (!balanced(high_activity + ha, control_activity + ca, .09) ||
            !balanced(high_magnitude + hm, control_magnitude + cm, .04) ||
            !balanced(high_energy + ea + eb, control_energy + ex + ey, .015) ||
            !balanced(high_row[r] + ea, control_row[r] + energy[cross_r], .12) ||
            !balanced(high_row[s] + eb, control_row[s] + energy[cross_s], .12) ||
            !balanced(high_col[c] + ea, control_col[c] + energy[cross_s], .12) ||
            !balanced(high_col[d] + eb, control_col[d] + energy[cross_r], .12)) continue;
        used_high[a] = used_high[b] = true;
        used_control[x] = used_control[y] = true;
        output_high[count] = a; output_control[count++] = x;
        output_high[count] = b; output_control[count++] = y;
        high_row[r] += ea; high_row[s] += eb;
        high_col[c] += ea; high_col[d] += eb;
        control_row[r] += energy[cross_r]; control_row[s] += energy[cross_s];
        control_col[c] += energy[cross_s]; control_col[d] += energy[cross_r];
        high_activity += ha; control_activity += ca;
        high_magnitude += hm; control_magnitude += cm;
        high_energy += ea + eb; control_energy += ex + ey;
    }
    return count;
}
