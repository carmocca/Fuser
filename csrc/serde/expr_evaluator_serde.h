// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ir/all_nodes.h>
#include <kernel.h>
#include <vector>

namespace nvfuser::serde {

class ExpressionSerializer {
 public:
  ExpressionSerializer() = default;

  flatbuffers::Offset<serde::NaiveValueGenerator> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      kir::Kernel* kernel);

  std::vector<flatbuffers::Offset<AllocateBuffer>> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const std::vector<const kir::Allocate*>& allocations);

 private:
  flatbuffers::Offset<Instruction> serializeUnaryOp(
      flatbuffers::FlatBufferBuilder& builder,
      UnaryOp* uop) const;

  flatbuffers::Offset<Instruction> serializeBinaryOp(
      flatbuffers::FlatBufferBuilder& builder,
      BinaryOp* bop) const;

  flatbuffers::Offset<SymbolicTensor> serialize(
      flatbuffers::FlatBufferBuilder& builder,
      const nvfuser::TensorView* tv);

  std::unordered_map<Val*, long> operation_stack_;
};

} // namespace nvfuser::serde
