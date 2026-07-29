# M5: the deferred rebuild reads the WRONG perspective's king out of the
# snapshot -- the exact class the snapshot form makes newly possible.
p='src/eval/nnue/sota_features.rs'; s=open(p).read()
a="    let ksq = pos.kings[perspective as usize];"
b="    let ksq = pos.kings[perspective.flip() as usize];"
assert s.count(a)==1; open(p,'w').write(s.replace(a,b))
