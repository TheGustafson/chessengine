# M3: let the ancestor walk step PAST a Refresh ply (treat only Done as a stop).
p='src/eval/nnue/sota_threats.rs'; s=open(p).read()
a="            while *a > 0 && self.stack[*a].fold[p] == Fold::FromParent {"
b="            while *a > 0 && self.stack[*a].fold[p] != Fold::Done {"
assert s.count(a)==1; open(p,'w').write(s.replace(a,b))
