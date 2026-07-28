use std::{collections::BTreeMap, env, fs::File, io::BufReader, process::ExitCode};

use sfbinpack::{TrainingDataEntry, parse_chunk, read_chunk_into};

fn increment(map: &mut BTreeMap<i64, u64>, key: i64) {
    *map.entry(key).or_default() += 1;
}

fn render_map(map: &BTreeMap<i64, u64>) -> String {
    map.iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: archon-sfbinpack-audit FILE [MAX_ENTRIES]");
        return ExitCode::from(2);
    };
    let max_entries = args
        .next()
        .and_then(|text| text.parse::<u64>().ok())
        .unwrap_or(u64::MAX);
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("open {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = file.metadata().map(|value| value.len()).unwrap_or(0);
    let mut reader = BufReader::new(file);
    let mut chunk = Vec::new();
    let mut chunks = 0_u64;
    let mut entries = 0_u64;
    let mut read_error = None;
    let mut results = BTreeMap::new();
    let mut score_abs_100 = BTreeMap::new();
    let mut ply_16 = BTreeMap::new();
    let mut rule50_10 = BTreeMap::new();
    let mut white_to_move = 0_u64;
    let mut in_check = 0_u64;
    let mut continuation_pairs = 0_u64;
    let mut previous: Option<TrainingDataEntry> = None;
    let mut samples = Vec::new();

    loop {
        match read_chunk_into(&mut reader, &mut chunk) {
            Ok(true) => chunks += 1,
            Ok(false) => break,
            Err(error) => {
                read_error = Some(error.to_string());
                break;
            }
        }
        for entry in parse_chunk(&chunk) {
            entries += 1;
            increment(&mut results, i64::from(entry.result));
            increment(
                &mut score_abs_100,
                i64::from(entry.score.unsigned_abs() / 100),
            );
            increment(&mut ply_16, i64::from(entry.ply / 16));
            increment(&mut rule50_10, i64::from(entry.pos.rule50_counter() / 10));
            white_to_move += u64::from(entry.pos.side_to_move().ordinal() == 0);
            in_check += u64::from(entry.pos.is_checked(entry.pos.side_to_move()));
            if let Some(prior) = previous {
                continuation_pairs += u64::from(prior.is_continuation(&entry));
            }
            previous = Some(entry);
            if samples.len() < 5 {
                samples.push(format!(
                    "{{\"fen\":\"{}\",\"move\":\"{}\",\"score_stm\":{},\"ply\":{},\"result_stm\":{},\"rule50\":{}}}",
                    entry.pos.fen().unwrap(),
                    entry.mv.as_uci(),
                    entry.score,
                    entry.ply,
                    entry.result,
                    entry.pos.rule50_counter()
                ));
            }
            if entries >= max_entries {
                break;
            }
        }
        if entries >= max_entries {
            break;
        }
    }

    let error_json = read_error
        .as_ref()
        .map(|value| format!("\"{}\"", value.replace('"', "\\\"")))
        .unwrap_or_else(|| "null".to_string());
    println!(
        "{{\n  \"schema\": \"archon.sfbinpack-audit.v1\",\n  \"input\": \"{}\",\n  \"input_bytes\": {},\n  \"complete_chunks\": {},\n  \"entries\": {},\n  \"truncated_tail_error\": {},\n  \"result_stm_counts\": {{{}}},\n  \"abs_score_100cp_bucket_counts\": {{{}}},\n  \"ply_16_bucket_counts\": {{{}}},\n  \"rule50_10_bucket_counts\": {{{}}},\n  \"white_to_move\": {},\n  \"in_check\": {},\n  \"continuation_pairs\": {},\n  \"samples\": [{}]\n}}",
        path.replace('"', "\\\""),
        bytes,
        chunks,
        entries,
        error_json,
        render_map(&results),
        render_map(&score_abs_100),
        render_map(&ply_16),
        render_map(&rule50_10),
        white_to_move,
        in_check,
        continuation_pairs,
        samples.join(",")
    );
    ExitCode::SUCCESS
}
