// Binary bipartite edge-swap chain. Fixed proposal count includes self transitions.
#include <cstdint>
#include <random>
#include <vector>
extern "C" uint64_t binary_swaps(uint8_t* mask, int64_t nr, int64_t nc,
                                 uint64_t seed, uint64_t attempts) {
    std::vector<int64_t> edge;
    for (int64_t i=0;i<nr*nc;++i) if (mask[i]) edge.push_back(i);
    if (edge.size()<2) return 0;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> pick(0,edge.size()-1);
    uint64_t accepted=0;
    for(uint64_t k=0;k<attempts;++k) {
        uint64_t a=pick(rng), b=pick(rng);
        int64_t x=edge[a], y=edge[b], r=x/nc, s=y/nc, c=x%nc, d=y%nc;
        if(r==s || c==d) continue;
        int64_t u=r*nc+d, v=s*nc+c;
        if(mask[u] || mask[v]) continue;
        mask[x]=mask[y]=0; mask[u]=mask[v]=1;
        edge[a]=u; edge[b]=v; ++accepted;
    }
    return accepted;
}
