p='src/movegen.rs'; s=open(p).read()
a = """        if m.kind() != MoveKind::Normal {
            let mut list = MoveList::new();
            if self.checkers().is_empty() {
                generate::<{ gen_type::NON_EVASIONS }>(self, &mut list);
            } else {
                generate::<{ gen_type::EVASIONS }>(self, &mut list);
            }
            return list.contains(&m);
        }"""
b = """        if m.kind() != MoveKind::Normal {
            return self.pseudo_legal_special(m);
        }"""
assert s.count(a)==1, "arm not found"
s = s.replace(a, b)
anchor = "    pub fn pseudo_legal(&self, m: Move) -> bool {"
helper = '''    /// [`pseudo_legal`](Self::pseudo_legal)'s non-Normal arm, out of line.
    ///
    /// Study 17 §5's rule: a whole arm of the function's own top-level
    /// `MoveKind` dispatch, where the caller already holds every argument
    /// (`self` and `m` — two, well under the six-register limit). This arm holds
    /// the ONLY `MoveList` in the function — 520 bytes of stack frame plus its
    /// initialisation — on a path taken by well under 5% of TT moves, and
    /// `pseudo_legal` is inlined into the movepick TT-move validation, so that
    /// frame was being reserved on every `MovePicker` construction.
    #[cold]
    #[inline(never)]
    fn pseudo_legal_special(&self, m: Move) -> bool {
        let mut list = MoveList::new();
        if self.checkers().is_empty() {
            generate::<{ gen_type::NON_EVASIONS }>(self, &mut list);
        } else {
            generate::<{ gen_type::EVASIONS }>(self, &mut list);
        }
        list.contains(&m)
    }

'''
assert s.count(anchor)==1
open(p,'w').write(s.replace(anchor, helper + anchor))
print("ok")
