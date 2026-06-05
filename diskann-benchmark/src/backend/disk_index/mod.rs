/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "disk-index")] {
        mod benchmarks;
        mod build;
        mod search;
        mod json_spancollector;

        /// Register disk index benchmarks when the `disk-index` feature is enabled.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            benchmarks::register_benchmarks(registry)
        }
    } else {
        crate::utils::stub_impl!(
            "disk-index",
            inputs::disk::DiskIndexOperation
        );

        mod imp2 {
            use diskann_benchmark_runner::{
                benchmark::{FailureScore, MatchScore},
                output::Output,
                Benchmark, Checkpoint, Registry,
            };
            use crate::inputs;

            pub(super) fn register(name: &str, registry: &mut Registry) -> anyhow::Result<()> {
                Ok(registry.register(name, Stub)?)
            }

            pub(super) struct Stub;

            impl Benchmark for Stub {
                type Input = inputs::disk::DiskFilterIndexOperation;
                type Output = serde_json::Value;

                fn try_match(&self, _input: &inputs::disk::DiskFilterIndexOperation) -> Result<MatchScore, FailureScore> {
                    Err(FailureScore(0))
                }

                fn description(&self, f: &mut std::fmt::Formatter<'_>, _input: Option<&inputs::disk::DiskFilterIndexOperation>) -> std::fmt::Result {
                    writeln!(f, "Requires the \"disk-index\" feature")
                }

                fn run(&self, _input: &inputs::disk::DiskFilterIndexOperation, _checkpoint: Checkpoint<'_>, _output: &mut dyn Output) -> anyhow::Result<serde_json::Value> {
                    panic!("this function should not be called!");
                }
            }
        }

        /// Register stubs that guide users to enable the `disk-index` feature.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            imp::register("disk-index", registry)?;
            imp2::register("disk-index-filter", registry)
        }
    }
}
