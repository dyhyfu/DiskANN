/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use std::fmt::Debug;

use diskann::{graph::index::QueryLabelProvider, utils::VectorId};
use diskann_benchmark_runner::files::InputFile;
use diskann_label_filter::{
    kv_index::GenericIndex,
    stores::bftree_store::BfTreeStore,
    traits::{
        posting_list_trait::{PostingList, RoaringPostingList},
        query_evaluator::QueryEvaluator,
    },
    ASTExpr, DefaultKeyCodec,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;
use std::sync::Arc;

pub struct QueryBitmapEvaluator {
    pub ast_expr: ASTExpr,
    evaluated_bitmap: RoaringPostingList,
}

impl QueryBitmapEvaluator {
    /// Create a new filter and evaluate the bitmap immediately (existing behavior).
    pub fn new(
        ast_expr: ASTExpr,
        inverted_index: &GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>,
    ) -> Self {
        let evaluated_bitmap = inverted_index.evaluate_query(&ast_expr).unwrap();
        Self {
            ast_expr,
            evaluated_bitmap,
        }
    }

    /// Ensure evaluated and return a reference to the bitmap (convenience).
    fn get_bitmap(&self) -> &RoaringPostingList {
        &self.evaluated_bitmap
    }

    /// Number of matching labels in this filter's evaluated bitmap.
    pub fn count(&self) -> usize {
        self.get_bitmap().len()
    }
}

impl Debug for QueryBitmapEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapFilter")
            .field("ast_expr", &self.ast_expr)
            .field("evaluated_bitmap", &self.evaluated_bitmap)
            .finish()
    }
}

impl<T> QueryLabelProvider<T> for QueryBitmapEvaluator
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.get_bitmap().contains(vec_id.into_usize())
    }
}

#[derive(Debug)]
pub struct BitmapFilter(pub BitSet);

impl<T> QueryLabelProvider<T> for BitmapFilter
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.0.contains(vec_id.into_usize())
    }
}

pub(crate) fn generate_bitmaps(
    query_predicates: &InputFile,
    data_labels: &InputFile,
) -> anyhow::Result<Vec<BitSet>> {
    let bit_maps = match read_labels_and_compute_bitmap(
        data_labels.to_str().unwrap(),
        query_predicates.to_str().unwrap(),
    ) {
        Ok(bit_maps) => bit_maps,
        Err(e) => {
            return Err(e.into());
        }
    };
    Ok(bit_maps)
}

/// Fast bitmap generation for predicates that contain only `{"vector_id": {"$eq": "N"}}` leaves.
///
/// Scans each line for the literal pattern `"$eq":"N"` (or `"$eq": "N"` with spaces) and
/// parses N directly, bypassing full JSON parsing entirely. This is O(line_length) per query
/// line instead of O(total_vectors) for the old inverted-index approach.
pub(crate) fn generate_bitmaps_from_eq_predicates(
    query_predicates: &InputFile,
) -> anyhow::Result<Vec<BitSet>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open(query_predicates.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", query_predicates.display()))?;

    BufReader::new(file)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.map_err(|e| anyhow::anyhow!("line {i}: {e}"))?;
            Ok(extract_eq_ids_from_line(&line))
        })
        .collect()
}

/// Scan a raw JSONL line for all `"$eq":"N"` occurrences and insert N into a BitSet.
/// Does not parse the full JSON — just finds the literal `"$eq"` key and reads the
/// numeric string that follows, which is safe given the known predicate schema.
fn extract_eq_ids_from_line(line: &str) -> BitSet {
    let mut bitset = BitSet::new();
    // Match both `"$eq":"N"` (no space) and `"$eq": "N"` (space after colon).
    let needle = "\"$eq\":";
    let bytes = line.as_bytes();
    let needle_bytes = needle.as_bytes();
    let nlen = needle_bytes.len();

    let mut pos = 0;
    while pos + nlen < bytes.len() {
        // Find next occurrence of `"$eq":"`
        if bytes[pos..pos + nlen] == *needle_bytes {
            pos += nlen;
            // Skip optional whitespace between `:` and `"`
            while pos < bytes.len() && bytes[pos] == b' ' {
                pos += 1;
            }
            // Expect opening `"`
            if pos >= bytes.len() || bytes[pos] != b'"' {
                continue;
            }
            pos += 1;
            // Collect digits until closing `"`
            let start = pos;
            while pos < bytes.len() && bytes[pos] != b'"' {
                pos += 1;
            }
            if let Ok(s) = std::str::from_utf8(&bytes[start..pos]) {
                if let Ok(id) = s.parse::<usize>() {
                    bitset.insert(id);
                }
            }
        } else {
            pos += 1;
        }
    }
    bitset
}

pub(crate) fn setup_filter_strategies<I, S>(
    beta: f32,
    bit_maps: I,
    search_strategy: S,
) -> Vec<BetaFilter<S, u32>>
where
    I: IntoIterator<Item = Arc<dyn QueryLabelProvider<u32>>>,
    S: Clone,
{
    bit_maps
        .into_iter()
        .map(|bit_map| BetaFilter::<S, u32>::new(search_strategy.clone(), bit_map, beta))
        .collect::<Vec<_>>()
}

pub(crate) fn as_query_label_provider(set: BitSet) -> Arc<dyn QueryLabelProvider<u32>> {
    Arc::new(BitmapFilter(set))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_filter_match() {
        let mut bitset = BitSet::new();
        bitset.insert(1);
        bitset.insert(3);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1u32));
        assert!(filter.is_match(3u32));
        assert!(!filter.is_match(2u32));
        assert!(!filter.is_match(0u32));
    }

    #[test]
    fn test_bitmap_filter_empty() {
        let bitset = BitSet::new();
        let filter = BitmapFilter(bitset);

        assert!(!filter.is_match(0u32));
        assert!(!filter.is_match(10u32));
    }

    #[test]
    fn test_bitmap_filter_large_id() {
        let mut bitset = BitSet::new();
        bitset.insert(1000);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1000u32));
        assert!(!filter.is_match(999u32));
    }
}
