#include <cmath>
#include <cstdint>

static inline bool balanced(double a, double b, double limit) {
    return std::abs(a-b) <= limit * std::max(a,b);
}

// Exhaustively test ordered high-corner pairs. Return 1 with a witness, 0 if
// the full space is exhausted, and -1 if max_pairs is reached.
extern "C" int exact_rectangle(int rows, int cols, const int *stratum,
                                const double *energy, const double *activity,
                                const int *high, int n_high, uint64_t max_pairs,
                                int *witness, uint64_t *pairs_tested,
                                uint64_t *locally_eligible) {
    uint64_t tested=0, eligible=0;
    for (int ia=0; ia<n_high; ++ia) {
        int a=high[ia], r=a/cols, c=a%cols;
        double ea=energy[a];
        for (int ib=0; ib<n_high; ++ib) {
            if (ia==ib) continue;
            if (tested++ >= max_pairs) {
                *pairs_tested=tested-1; *locally_eligible=eligible; return -1;
            }
            int b=high[ib], s=b/cols, d=b%cols;
            if (r==s || c==d) continue;
            int cross_r=r*cols+d, cross_s=s*cols+c;
            if (stratum[cross_r] < 0 || stratum[cross_s] < 0) continue;
            bool row_orientation = stratum[cross_r]==stratum[a] && stratum[cross_s]==stratum[b];
            bool col_orientation = stratum[cross_s]==stratum[a] && stratum[cross_r]==stratum[b];
            if (!row_orientation && !col_orientation) continue;
            ++eligible;
            double eb=energy[b], ex=energy[cross_r], ey=energy[cross_s];
            if (!balanced(activity[a]+activity[b],activity[cross_r]+activity[cross_s],.09) ||
                !balanced(std::sqrt(ea)+std::sqrt(eb),std::sqrt(ex)+std::sqrt(ey),.04) ||
                !balanced(ea+eb,ex+ey,.015) ||
                !balanced(ea,ex,.12) || !balanced(eb,ey,.12) ||
                !balanced(ea,ey,.12) || !balanced(eb,ex,.12)) continue;
            witness[0]=a; witness[1]=b; witness[2]=cross_r; witness[3]=cross_s;
            witness[4]=row_orientation ? 1 : 0; witness[5]=col_orientation ? 1 : 0;
            *pairs_tested=tested; *locally_eligible=eligible; return 1;
        }
    }
    *pairs_tested=tested; *locally_eligible=eligible; return 0;
}
