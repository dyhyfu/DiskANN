/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-filtered search using adaptive-L greedy expansion.
//!
//! All nodes (matched and unmatched) guide navigation so the graph can route
//! toward matching regions. After sampling [`ADAPTIVE_L_SAMPLE_COUNT`] neighbors
//! the algorithm estimates the filter selectivity and enlarges the search list
//! if needed, so that there are enough matching candidates in the final result.

use diskann_utils::Reborrow;
use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;

use super::{Knn, Search, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    graph::{
        glue::{self, ExpandBeam, SearchExt, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, InternalSearchStats, QueryLabelProvider, SearchStats},
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::VectorId,
};

/// Number of neighbors to observe before computing the adaptive L multiplier.
const ADAPTIVE_L_SAMPLE_COUNT: u32 = 1000;

/// Maximum multiplier applied to the base L value.
const ADAPTIVE_L_MAX_MULTIPLIER: f64 = 16.0;

/// Parameters for label-filtered search with adaptive search-list expansion.
///
/// All nodes — matched and unmatched — guide graph navigation. After observing
/// [`ADAPTIVE_L_SAMPLE_COUNT`] neighbors the search list is enlarged based on
/// the observed match rate, trading latency for recall on highly selective filters.
#[derive(Debug)]
pub struct AdaptiveLGreedySearch<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Label evaluator for determining whether a node is a result candidate.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
}

impl<'q, InternalId> AdaptiveLGreedySearch<'q, InternalId> {
    /// Create new adaptive-L greedy search parameters.
    pub fn new(inner: Knn, label_evaluator: &'q dyn QueryLabelProvider<InternalId>) -> Self {
        Self {
            inner,
            label_evaluator,
        }
    }
}

impl<'q, DP, S, T> Search<DP, S, T> for AdaptiveLGreedySearch<'q, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<DP, T>,
    T: Copy + Send + Sync,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        processor: PP,
        context: &DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: for<'a> SearchPostProcess<S::SearchAccessor<'a>, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l_value().get(), start_ids.len());

            let stats = greedy_filter_search_internal(
                index.max_degree_with_slack(),
                &self.inner,
                &mut accessor,
                &computer,
                &mut scratch,
                &mut NoopSearchRecord::new(),
                self.label_evaluator,
            )
            .await?;

            // Only pass matching candidates to the post-processor.
            // scratch.best contains all nodes (matched + unmatched) since unmatched
            // nodes guide navigation, so we must filter here before output.
            let label_evaluator = self.label_evaluator;
            let matching_candidates = scratch
                .best
                .iter()
                .take(scratch.best.capacity())
                .filter(|n| label_evaluator.is_match(n.id));

            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    matching_candidates,
                    output,
                )
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

/// Computes the adaptive L value based on the observed match rate.
///
/// Uses a logarithmic scale for very selective filters so that low match rates
/// cause a proportionally larger expansion without growing unboundedly.
fn compute_adaptive_l(base_l: usize, sample_visited: u32, sample_matched: u32) -> usize {
    if sample_visited == 0 {
        return (base_l as f64 * ADAPTIVE_L_MAX_MULTIPLIER) as usize;
    }

    let match_rate = sample_matched as f64 / sample_visited as f64;

    let multiplier = if match_rate >= 0.5 {
        1.0
    } else if match_rate >= 0.1 {
        2.0
    } else if match_rate == 0.0 {
        ADAPTIVE_L_MAX_MULTIPLIER
    } else {
        // Logarithmic: 2^(-log10(match_rate)).
        // At match_rate=0.1  → 2^1 = 2×
        // At match_rate=0.01 → 2^2 = 4×
        // At match_rate=0.001 → 2^3 = 8×
        let neg_log10 = -match_rate.log10();
        2.0_f64.powf(neg_log10)
    };

    let clamped = multiplier.min(ADAPTIVE_L_MAX_MULTIPLIER);
    (base_l as f64 * clamped).ceil() as usize
}

/// Internal greedy filter search.
///
/// All neighbors are inserted into `scratch.best` regardless of label match, so
/// unmatched nodes continue to guide navigation. The match rate is sampled over
/// the first [`ADAPTIVE_L_SAMPLE_COUNT`] neighbors; if the filter is selective the
/// search list is enlarged so that the post-processor has enough candidates to fill
/// the requested top-k result set.
pub(crate) async fn greedy_filter_search_internal<I, A, T, SR>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    label_evaluator: &dyn QueryLabelProvider<I>,
) -> ANNResult<InternalSearchStats>
where
    I: VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();
    let base_l = search_params.l_value().get();

    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Initialise with starting points.
    if scratch.visited.is_empty() {
        let start_ids = accessor.starting_points().await?;
        for id in start_ids {
            scratch.visited.insert(id);
            let element = accessor
                .get_element(id)
                .await
                .escalate("start point retrieval must succeed")?;
            let dist = computer.evaluate_similarity(element.reborrow());
            scratch.best.insert(Neighbor::new(id, dist));
        }
    }

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);

    // Adaptive-L state.
    let mut sample_visited: u32 = 0;
    let mut sample_matched: u32 = 0;
    let mut adaptive_l_computed = false;

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();
        neighbors.clear();

        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
        }

        // Expand beam — all neighbors go into scratch.best regardless of label.
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        for &neighbor in &neighbors {
            scratch.best.insert(neighbor);
            // Count matches for adaptive-L estimation.
            if !adaptive_l_computed {
                sample_visited += 1;
                if label_evaluator.is_match(neighbor.id) {
                    sample_matched += 1;
                }
            }
        }

        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // After enough samples, compute adaptive L and resize if needed.
        if !adaptive_l_computed && sample_visited >= ADAPTIVE_L_SAMPLE_COUNT {
            adaptive_l_computed = true;
            let new_l = compute_adaptive_l(base_l, sample_visited, sample_matched);
            if new_l > scratch.best.capacity() {
                scratch.resize(new_l);
            }
        }
    }

    Ok(make_stats(scratch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_l_high_match_rate_no_expansion() {
        // ≥50% match → 1× multiplier → same L
        assert_eq!(compute_adaptive_l(100, 1000, 600), 100);
        assert_eq!(compute_adaptive_l(100, 1000, 500), 100);
    }

    #[test]
    fn adaptive_l_medium_match_rate_doubles() {
        // 10–50% → 2×
        assert_eq!(compute_adaptive_l(100, 1000, 200), 200); // 20%
        assert_eq!(compute_adaptive_l(100, 1000, 100), 200); // 10%
    }

    #[test]
    fn adaptive_l_low_match_rate_log_scale() {
        // ~1% match → 4×
        let l = compute_adaptive_l(100, 1000, 10);
        assert!(l >= 400 && l <= 401, "expected ~400, got {l}");
        // ~0.1% match → 8×
        let l = compute_adaptive_l(100, 1000, 1);
        assert!(l >= 800 && l <= 801, "expected ~800, got {l}");
    }

    #[test]
    fn adaptive_l_zero_matches_uses_max_multiplier() {
        assert_eq!(compute_adaptive_l(100, 1000, 0), 1600);
    }

    #[test]
    fn adaptive_l_zero_visited_uses_max_multiplier() {
        assert_eq!(compute_adaptive_l(100, 0, 0), 1600);
    }

    #[test]
    fn adaptive_l_capped_at_max_multiplier() {
        // Even with 0 matches, should not exceed base_l * 16
        let l = compute_adaptive_l(50, 1000, 0);
        assert_eq!(l, 800);
    }
}
