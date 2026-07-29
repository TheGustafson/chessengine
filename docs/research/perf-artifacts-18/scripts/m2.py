# M2: never RECORD the obligation -- a crossing ply folds from its parent, i.e.
# mixes two mirror orientations.
p='src/eval/nnue/sota_threats.rs'; s=open(p).read()
a="""            self.stack[child].fold[pi] = if king_crossed(dp, persp) {
                Fold::Refresh
            } else {
                Fold::FromParent
            };"""
b="""            let _ = persp;
            self.stack[child].fold[pi] = Fold::FromParent;"""
assert s.count(a)==1; open(p,'w').write(s.replace(a,b))
