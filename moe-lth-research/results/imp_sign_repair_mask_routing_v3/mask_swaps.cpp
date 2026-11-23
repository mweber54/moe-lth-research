#include <cstdint>
#include <random>
#include <utility>

// Run the symmetric double-edge exchange chain on each bipartite matrix.
extern "C" int randomize(int columns, uint8_t *mask, int *edges, int count,
                         uint64_t seed, int trials) {
    if (count < 2) return 0;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> sample(0, count - 1);
    int accepted = 0;
    for (int t = 0; t < trials; ++t) {
        int i = sample(rng), j = sample(rng);
        int a = edges[i], b = edges[j];
        int x = a / columns * columns + b % columns;
        int y = b / columns * columns + a % columns;
        if (mask[x] || mask[y]) continue;
        mask[a] = mask[b] = 0;
        mask[x] = mask[y] = 1;
        edges[i] = x;
        edges[j] = y;
        ++accepted;
    }
    return accepted;
}
