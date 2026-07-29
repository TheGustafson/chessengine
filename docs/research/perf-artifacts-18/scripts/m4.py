# M4: the sibling overwrite leaves a stale Fold::Done (the dangerous direction).
p='src/eval/nnue/sota_threats.rs'; s=open(p).read()
a="""            self.stack[child].fold[pi] = if king_crossed(dp, persp) {
                Fold::Refresh
            } else {
                Fold::FromParent
            };"""
b="""            if self.stack[child].fold[pi] != Fold::Done {
                self.stack[child].fold[pi] = if king_crossed(dp, persp) {
                    Fold::Refresh
                } else {
                    Fold::FromParent
                };
            }"""
assert s.count(a)==1; open(p,'w').write(s.replace(a,b))
