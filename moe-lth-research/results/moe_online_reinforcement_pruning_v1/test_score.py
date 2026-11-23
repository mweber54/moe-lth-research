import math

def score(w0,updates):
 denominator=sum(abs(x) for x in updates)
 return math.copysign(1,w0)*sum(updates)/denominator if w0 and denominator else None

assert score(2.,[.1,.2])==1.
assert score(-2.,[-.1,-.2])==1.
assert abs(score(-2.,[-.1,.2])+1/3)<1e-15
assert score(2.,[-.1,-.2])==-1.
assert score(0.,[.1]) is None
assert score(2.,[0.,0.]) is None
print('hand-computed sign convention PASS')
