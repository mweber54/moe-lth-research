#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
struct Node {double lo[4],hi[4];int begin,end,left=-1,right=-1;};
struct Tree {
 const double* p;std::vector<int> order;std::vector<Node> nodes;double scale[4];
 Tree(const double* x,int n):p(x),order(n){std::iota(order.begin(),order.end(),0);nodes.reserve(2*n+1);for(int d=0;d<4;d++){double lo=x[d],hi=x[d];for(int i=0;i<n;i++){lo=std::min(lo,x[4*i+d]);hi=std::max(hi,x[4*i+d]);}scale[d]=std::max(hi-lo,1e-300);}build(0,n);}
 int build(int a,int b){int id=nodes.size();nodes.emplace_back();nodes[id].begin=a;nodes[id].end=b;for(int d=0;d<4;d++){nodes[id].lo[d]=nodes[id].hi[d]=p[4*order[a]+d];for(int j=a+1;j<b;j++){double z=p[4*order[j]+d];nodes[id].lo[d]=std::min(nodes[id].lo[d],z);nodes[id].hi[d]=std::max(nodes[id].hi[d],z);}}
 if(b-a>16){int axis=0;double best=-1;for(int d=0;d<4;d++){double r=(nodes[id].hi[d]-nodes[id].lo[d])/scale[d];if(r>best){best=r;axis=d;}}int mid=(a+b)/2;std::nth_element(order.begin()+a,order.begin()+mid,order.begin()+b,[&](int i,int j){return p[4*i+axis]<p[4*j+axis];});int l=build(a,mid),r=build(mid,b);nodes[id].left=l;nodes[id].right=r;}return id;}
 uint64_t query(int id,const double* c,int dimensions,bool reverse){const auto& nd=nodes[id];double windows[3]={.005,.1,.1};double threshold=reverse?2*c[3]:c[3]/2;
 if(!reverse && c[3]<=0)return 0;
 if(reverse ? (c[3]==0?nd.hi[3]<=0:nd.hi[3]<threshold) : nd.lo[3]>threshold)return 0;
 bool inside=reverse ? (c[3]==0?nd.lo[3]>0:nd.lo[3]>=threshold):nd.hi[3]<=threshold;
 for(int d=0;d<dimensions;d++){if(nd.lo[d]-c[d]>windows[d] || c[d]-nd.hi[d]>windows[d])return 0;if(std::max(std::abs(nd.lo[d]-c[d]),std::abs(nd.hi[d]-c[d]))>windows[d])inside=false;}
 if(inside)return reverse?1:nd.end-nd.begin;
 if(nd.left<0){uint64_t count=0;for(int j=nd.begin;j<nd.end;j++){const double* q=p+4*order[j];bool ok=reverse?(c[3]==0?q[3]>0:q[3]>=threshold):q[3]<=threshold;for(int d=0;d<dimensions && ok;d++)ok=std::abs(q[d]-c[d])<=windows[d];if(ok){count++;if(reverse)return 1;}}return count;}
 uint64_t a=query(nd.left,c,dimensions,reverse);if(reverse && a)return 1;return a+query(nd.right,c,dimensions,reverse);}
};
extern "C" void count_ranges(const double* points,int n,uint64_t* high,uint8_t* low){if(!n)return;Tree tree(points,n);for(int i=0;i<n;i++)for(int d=1;d<=3;d++){high[3*i+d-1]=tree.query(0,points+4*i,d,false);low[3*i+d-1]=tree.query(0,points+4*i,d,true)>0;}}
