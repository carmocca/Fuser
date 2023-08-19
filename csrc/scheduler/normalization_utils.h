// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <executor_params.h>
#include <ir/all_nodes.h>
#include <scheduler/utils.h>
#include <cmath>
#include <optional>
#include <ostream>
#include <vector>

namespace nvfuser {
class SchedulerRuntimeInfo;
namespace normalization_scheduler_utils {

//! Utility class to iterate candidates of launch configurations in a
//! preferred order. The iteration order is defined as:
//!
//!   for bdimx in all valid bdimx in an decreasing order
//!     for gdimy in valid gdimy values in an increasing order
//!
//! Each of bdimx and gdimy determines bdimy and gdimx, respecitively,
//! such that the number of threads per block is always 256 and the
//! number of blocks is always equal to the number of SMs.
class PreferredLaunchConfig {
 public:
  //! Minimum blockDim.x.
  static constexpr int kMinBdimx = 8;
  //! Maximum blockDim.x.
  static constexpr int kMaxBdimx = 16;

  PreferredLaunchConfig();

  int bdimx() const {
    return bdimx_;
  }

  int bdimy() const {
    return bdimy_;
  }

  int gdimx() const {
    return gdimxAt(grid_dims_pos_);
  }

  int gdimy() const {
    return gdimyAt(grid_dims_pos_);
  }

  //! Peek the next gdimx. -1 is returned if no further gdimx is available.
  int peekNextGdimx() const;

  //! Peek the next gdimy. -1 is returned if no further gdimy is available.
  int peekNextGdimy() const;

  //! Move to the next launch configuration. Will be marked as invalid
  //! if no valid configuration exists. Return true if successfully moved.
  bool moveToNextConfig();

  //! Try setting blockDim to the next valid config if
  //! available. Return false if no valid config exists. gridDim is
  //! reset.
  bool moveToNextBdim();

  //! Query if the next configuration will cause blockDim.x to become
  //! smaller.
  bool isNextSmallerBdimx() const;

  //! Query if blockDim.x can be further lowered
  bool canLowerBdimx() const;

  //! Query if no valid configuration is found
  bool isInvalid() const {
    return !valid_;
  }

 private:
  //! Populate the list of valid gridDim configurations
  void initValidGdims();

  int gdimxAt(int pos) const {
    return valid_grid_dims_.at(pos).first;
  }

  int gdimyAt(int pos) const {
    return valid_grid_dims_.at(pos).second;
  }

  //! Set blockDim.x and in turn blockDim.y. Return true if the
  //! specified blockDim.x is successfully set. If dry_run is true,
  //! just check if the given config is valid but do not modify the
  //! current config.
  bool setBdimx(int bdimx, bool dry_run = false);

  void resetGdim() {
    grid_dims_pos_ = 0;
  }

  void resetBdim() {
    // Start with the maximum bdimx and lower it until satisfactory
    // config is found
    setBdimx(kMaxBdimx);
  }

  //! Try setting gridDim to the next valid config if
  //! available. Return false if no valid config exists
  bool moveToNextGdim();

  int getNextGdimsPos() const;

  void invalidate() {
    valid_ = false;
  }

  friend std::ostream& operator<<(std::ostream& os, PreferredLaunchConfig cfg) {
    os << "{gdimx: " << cfg.gdimx() << ", gdimy: " << cfg.gdimy()
       << ", bdimx: " << cfg.bdimx() << ", bdimy: " << cfg.bdimy() << "}";
    return os;
  }

 private:
  //! Remember if it is still a valid configuration
  bool valid_ = false;

  //! List of valid gridDims ordered by the dimension of
  //! gridDim.x. Larger gridDim.x is preferred as it would promote
  //! larger independent parallelism
  std::vector<std::pair<int, int>> valid_grid_dims_;
  //! The offset of the Current gridDim in valid_grid_dims_
  int grid_dims_pos_ = 0;

  //! Current blockDim.x
  int bdimx_ = 0;
  //! Current blockDim.y
  int bdimy_ = 0;
};

//! Scheduling parameters for grid outer normalization
struct GridOuterNormalizationParams {
  LaunchParams launch_params;
  int64_t persistent_buffer_factor = -1;
  int64_t unswitch_factor = -1;
};

std::optional<GridOuterNormalizationParams> getGridOuterNormalizationParams(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t persistent_buffer_size);

//! Parameters store memory space of persistent buffers.
//! By default, the persistent buffers are stored in registers, however, if it
//! exists in smem_tvs, it will be allocated in shared memory.
//! This happens when the persistent buffer size is larger
//! than the available registers.
struct PersistentBufferStorageParams {
  std::vector<TensorView*> smem_tvs;
  int64_t smem_buffer_size = -1;
  int64_t regs_buffer_size = -1;
  int64_t smem_overhead = -1;
  bool has_enough_regs_and_smem = false;
  bool project_to_input = false;
  bool combined_reduction = false;
};

//! check iter type of each domain in inner and outer reduction tvs
//! inner reduction must be [I,I,...R,R]
//! outer reduction must be [R,R,...I,I]
bool checkIfReductionsAreInnerOuter(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

//! check if the inner reduction has shared input with outer reduction
bool hasSharedInput(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

//! The first part of outer reduction is computed with inner reduction and the
//! second part is scheduled separately. So, (1) the outer reduction tvs can
//! only be connected with inner reduction tvs through their producers. (2)
//! Outer reduction tvs are also scheduled separately and they can only be
//! connected through their producers.
bool isConnectedOnlyThroughReductionProducer(
    const std::vector<TensorView*>& inner_reduction_tvs,
    const std::vector<TensorView*>& outer_reduction_tvs);

//! in combined_inner_outer_reduction, the partial results of outer
//! reductions must be persistent, calculate the size of these buffers when
//! estimate register usage
int64_t partialReductionBufferSize(
    const std::vector<TensorView*>& outer_reduction_tvs,
    SchedulerRuntimeInfo& runtime_info);

//! Calculate the persistent buffer batches and threads per block.
//! Start from a large value of inner_dim_numel / (inner_vect * warpSize/4),
//! gradually reduce to small values but not smaller than a threshold determined
//! by inner_dim_numel and outer_dim_numel.
std::pair<int64_t, int64_t> getInnerOuterPersistentBufferBatches(
    const int64_t inner_dim_numel,
    const int64_t outer_dim_numel,
    const int64_t regs_buffer_size,
    const int64_t smem_buffer_size,
    const int64_t vectorize_factor,
    const int64_t warp_size);

//! Check if there are enough registers and shared memories to keep the
//! persistent buffers on chip. Return regs_buffer_size,
//! smem_buffer_size, available_register_buffer_size,
//! has_enough_regs_and_smem
PersistentBufferStorageParams getPersistentBufferStorageParams(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const int64_t vectorize_factor);

//! Return the shared memory overhead per block includes reserved by the CUDA
//! driver and the space for the reduction broadcast workspace.
int64_t getSharedMemoryOverheadPerBlock(
    Fusion* fusion,
    const std::vector<TensorView*>& persistent_buffer_tvs,
    const int64_t max_threads_per_block);

//! Use the first inner reduction tv as the reference tv if the fusion has both
//! inner and outer reductions, otherwise use the first reduction tv.
TensorView* getReferenceReductionTv(
    const std::vector<TensorView*>& reduction_tvs);
} // namespace normalization_scheduler_utils
} // namespace nvfuser
