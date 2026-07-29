# M1: never discharge the deferred refresh -- the walk stops at the Refresh ply
# and the children fold from its STALE accumulator.
p='src/eval/nnue/sota_threats.rs'; s=open(p).read()
a="            if self.stack[*a].fold[p] == Fold::Refresh {"
b="            if false && self.stack[*a].fold[p] == Fold::Refresh {"
assert s.count(a)==1, s.count(a); open(p,'w').write(s.replace(a,b))
