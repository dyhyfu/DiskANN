/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::search::{self, Search, graph::Strategy};

/// A built-in helper for benchmarking filtered K-nearest neighbors search
/// using the adaptive-L greedy search method.
///
/// All nodes guide graph navigation; after sampling 1000 neighbors the search
/// list is enlarged based on the observed filter match rate.
#[derive(Debug)]
pub struct AdaptiveL<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
}

impl<DP, T, S> AdaptiveL<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`AdaptiveL`] searcher.
    ///
    /// `labels` length must match the number of rows in `queries`.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        if labels.len() != queries.nrows() {
            Err(anyhow::anyhow!(
                "Number of label providers ({}) must be equal to the number of queries ({})",
                labels.len(),
                queries.nrows()
            ))
        } else {
            Ok(Arc::new(Self {
                index,
                queries,
                strategy,
                labels,
            }))
        }
    }
}

impl<DP, T, S> Search for AdaptiveL<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [T], DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::Knn;
    type Output = super::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let adaptive_search =
            graph::search::AdaptiveLGreedySearch::new(*parameters, &*self.labels[index]);
        let processor = self.strategy.get(index)?.default_post_processor();
        let stats = self
            .index
            .search_with(
                adaptive_search,
                self.strategy.get(index)?,
                processor,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}
