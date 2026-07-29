p='src/eval/nnue/sota_features.rs'; s=open(p).read()
start = s.index("pub(super) fn for_each_threat_at(")
end = s.index("// ===========================================================================\n// From-scratch accumulator refresh")
b = '''pub(super) fn for_each_threat_at(
    pos: &ThreatPosition,
    perspective: Color,
    mut f: impl FnMut(usize),
) {
    let ksq = pos.kings[perspective as usize];
    let occ = pos.occ;
    let all_pawns = pos.typed(PieceType::Pawn);

    for c in Color::ALL {
        // --- Pawns: two diagonal captures + a straight push-threat ---------
        let attacker = Piece::new(c, PieceType::Pawn);
        for from in pos.pieces(c, PieceType::Pawn) {
            emit_targets(pos, perspective, ksq, attacker, from, pawn_attacks(c, from), &mut f);
            // Push threat: a pawn directly in front (own or enemy) is
            // threatened. NOT over-generatable: `MAP` would wrongly keep a
            // pawn push onto a knight or a rook, so the victim mask here is
            // pawns and only pawns.
            if let Some(to) = from.shifted(c.pawn_push())
                && all_pawns.contains(to)
            {
                let attacked = Piece::new(pos.color_on(to), PieceType::Pawn);
                if let Some(idx) = threat_index(perspective, attacker, from, to, attacked, ksq) {
                    f(idx);
                }
            }
        }

        // --- Knight, bishop, rook, queen -----------------------------------
        for pt in [
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let attacker = Piece::new(c, pt);
            for from in pos.pieces(c, pt) {
                emit_targets(pos, perspective, ksq, attacker, from, attacks(pt, from, occ), &mut f);
            }
        }
        // Kings are never attackers (whole row of `map` is -1), so they are
        // deliberately not looped.
    }
}

/// Emit every kept relation from `attacker` on `from` into its attack set
/// `att`, **iterating one victim type at a time** so the victim's type is a
/// loop constant.
///
/// This is [`super::sota_threats`]'s `record_targets` shape, brought to the
/// from-scratch side. Two things change, and the second is the point:
///
/// * the victim's identity is `(color_on(to), vt)` — ONE bitboard test — where
///   the coarse-mask form had to ask `piece_on(to)`, five type tests against a
///   snapshot (or a mailbox byte plus an `Option` unwrap with a live panic edge
///   against a `Board`); and
/// * the victim mask now comes from [`KEPT_VICTIMS`], i.e. from `MAP`, instead
///   of three hand-written unions (`pawn_targets` / `minor_slider_targets` /
///   `queen_targets`) that had to be kept in agreement with `MAP` by eye. The
///   incremental observer was already driven off `KEPT_VICTIMS`; now both sides
///   of the "incremental == refresh" oracle read the same table, and
///   `kept_victim_masks_match_predicate` pins it.
#[inline]
fn emit_targets(
    pos: &ThreatPosition,
    perspective: Color,
    ksq: Square,
    attacker: Piece,
    from: Square,
    att: Bitboard,
    f: &mut impl FnMut(usize),
) {
    let kept = KEPT_VICTIMS[attacker.piece_type as usize];
    for vt in PieceType::ALL {
        if kept & (1 << vt as u8) == 0 {
            continue;
        }
        for to in att & pos.typed(vt) {
            let attacked = Piece::new(pos.color_on(to), vt);
            if let Some(idx) = threat_index(perspective, attacker, from, to, attacked, ksq) {
                f(idx);
            }
        }
    }
}

'''
open(p,'w').write(s[:start] + b + s[end:])
print("ok")
