// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>

#include <expr_evaluator.h>
#include <fusion.h>
#include <id_model/id_graph.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ops/all_ops.h>

namespace nvfuser {

class IdGraphTest : public NVFuserTest {};

namespace {

auto buildIterDomainDefinitionsAndUses(
    const std::vector<TensorView*>& all_tvs) {
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_uses;
  std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>> id_definitions;

  for (auto tv : all_tvs) {
    VectorOfUniqueEntries<IterDomain*> root_domain_ids{
        tv->getRootDomain().begin(), tv->getRootDomain().end()};

    auto all_ids = ir_utils::allIDsOf(tv);

    // Check is this domain is a consumer of a view-like operation
    bool view_like_domain = tv->domain()->hasViewLikeRFactor();

    for (auto id : all_ids) {
      // Check if this id is a view like rfactor id
      if (view_like_domain && id->isRFactorProduct()) {
        // If the tensor domain is a view like domain, and the iteration
        // domain is marked as an rfactor product and is in the rfactor
        // domain, it's a view like rfactor iteration domain
        const auto& rfactor_domain = tv->domain()->maybeRFactor();
        if (std::find(rfactor_domain.begin(), rfactor_domain.end(), id) !=
            rfactor_domain.end()) {
          // view_rfactor_ids_.emplace(id);
        }
      }

      if (id_definitions.find(id) == id_definitions.end()) {
        id_definitions[id] = {};
      }

      if (id_uses.find(id) == id_uses.end()) {
        id_uses[id] = {};
      }

      auto def = id->definition();

      if (def == nullptr || root_domain_ids.has(id)) {
        continue;
      }

      if (id_definitions.find(id) == id_definitions.end()) {
        id_definitions[id] = {};
      }
      id_definitions.at(id).pushBack(def);

      auto inp_ids = ir_utils::filterByType<IterDomain>(def->inputs());
      for (auto inp_id : inp_ids) {
        if (id_uses.find(inp_id) == id_uses.end()) {
          id_uses[inp_id] = {};
        }
        id_uses.at(inp_id).pushBack(def);
      }
    }
  }

  return std::make_pair(id_uses, id_definitions);
}

IdGraph initializeIdGraph(
    bool propagate_exprs,
    const std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>>&
        id_uses,
    const std::unordered_map<IterDomain*, VectorOfUniqueEntries<Expr*>>&
        id_definitions) {
  IdGraph id_graph(propagate_exprs);

  for (const auto& definition_entry : id_definitions) {
    auto id = definition_entry.first;
    auto defs = definition_entry.second;
    auto uses_it = id_uses.find(id);
    TORCH_INTERNAL_ASSERT(
        uses_it != id_uses.end(),
        "Failed to initialize id: ",
        id->toString(),
        " as it's missing a definition entry.");
    id_graph.initializeId(id, defs, uses_it->second);
  }

  return id_graph;
}

void buildExactMap(const std::vector<Expr*>& exprs, IdGraph& id_graph) {
  for (auto expr : exprs) {
    TensorView* c_tv = ir_utils::getTvOutput(expr);

    auto all_tv_outputs = ir_utils::filterByType<TensorView>(expr->outputs());

    // Map siblings, as all other tv output domains must match the first tv
    // outputs domain.
    std::deque<TensorView*> other_tv_outputs(
        all_tv_outputs.begin(), all_tv_outputs.end());
    other_tv_outputs.pop_front();

    for (auto other_tv_output : other_tv_outputs) {
      // Sibling tv's must be exactly mapped with eachother so simply zip
      // their leaf iter domains.

      TORCH_INTERNAL_ASSERT(
          other_tv_output->getRootDomain().size() ==
              c_tv->getRootDomain().size(),
          "Multiple outputs with mismatched TV domains is not supported.");

      for (auto domain_i : c10::irange(c_tv->getRootDomain().size())) {
        auto c_id = c_tv->getRootDomain()[domain_i];
        auto o_id = other_tv_output->getRootDomain()[domain_i];
        id_graph.mapIds(o_id, c_id);
      }
    }

    // Map producer-consumer relationships based on the root domain map
    auto tv_inputs = ir_utils::filterByType<TensorView>(expr->inputs());
    for (auto p_tv : tv_inputs) {
      // For exact mapings do not map any broadcast dimensions to
      // non-broadcast dimensions. Prevent any broadcasted axes being mapped
      // to non-broadcasted axes.
      auto exact_c2p_root_map =
          PairwiseRootDomainMap(p_tv, c_tv)
              .mapBroadcast(false)
              .mapConsumerToProducer(c_tv->domain(), p_tv->domain());

      for (auto c_id : getSortedKeys(exact_c2p_root_map, Statement::lessThan)) {
        auto p_id = exact_c2p_root_map.at(c_id);
        std::cerr << "Map: " << c_id->toString() << ", " << p_id->toString()
                  << std::endl;
        id_graph.mapIds(c_id, p_id);
      }
    }

    id_graph.mapThroughLoopSwizzles();
  }
}

// Partially copied from IterDomainGraphs::build for testing IdGraph only
IdGraph buildExactMap(Fusion* fusion) {
  FusionGuard fg(fusion);

  auto exprs = fusion->exprs();

  std::vector<Expr*> tv_exprs;

  std::copy_if(
      exprs.begin(), exprs.end(), std::back_inserter(tv_exprs), [](Expr* expr) {
        TORCH_INTERNAL_ASSERT(expr != nullptr);
        return ir_utils::isTvOp(expr);
      });

  auto all_tvs = ir_utils::allTvsOfExprs(tv_exprs);

  std::unordered_set<TensorView*> all_added_tvs(all_tvs.begin(), all_tvs.end());
  for (auto additional_tv :
       ir_utils::filterByType<TensorView>(fusion->inputs())) {
    if (all_added_tvs.insert(additional_tv).second) {
      all_tvs.push_back(additional_tv);
    }
  }
  for (auto additional_tv :
       ir_utils::filterByType<TensorView>(fusion->outputs())) {
    if (all_added_tvs.insert(additional_tv).second) {
      all_tvs.push_back(additional_tv);
    }
  }

  if (all_tvs.empty()) {
    return IdGraph();
  }

  // Add uses and definitions to all iter domains.
  auto [id_uses, id_definitions] = buildIterDomainDefinitionsAndUses(all_tvs);

  auto id_graph = initializeIdGraph(true, id_uses, id_definitions);

  buildExactMap(tv_exprs, id_graph);

  return id_graph;
}

} // namespace

// Test the exact map with a multi-promotion fusion pattern. Promotion
// should not matter as the exact map is concerned
TEST_F(IdGraphTest, MultiPromotionExactMap) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [y]
  auto tv0 = makeSymbolicTensor(1);
  // [w, x, y, z]
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // y
  auto tv2 = broadcast(tv0, {true, false});
  // w, y
  auto tv3 = broadcast(tv2, {false, false, true});
  // w, y, z
  auto tv4 = broadcast(tv3, {false, true, false, false});
  // w, x, y, z
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(1)->merge(1)->merge(0)->split(0, 11);

  TransformPropagator propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  inlineAllAt(tv5, 1);

  auto exact_map = buildExactMap(&fusion);

  // Make sure the non-root IDs should not be mapped at
  // all, except for tv1 and tv5, which should be mapped with each
  // other, so their non-root domain groups should have size 2.
  for (auto tv : {tv0, tv1, tv2, tv3, tv4, tv5}) {
    for (auto id : ir_utils::allIDsOf(tv)) {
      if (std::find(
              tv->getRootDomain().begin(), tv->getRootDomain().end(), id) !=
          tv->getRootDomain().end()) {
        continue;
      }

      size_t expected_size = 0;
      if (tv->name() == 1 || tv->name() == 5) {
        expected_size = 2;
      } else {
        expected_size = 1;
      }
      const auto& idg = exact_map.toGroup(id);
      ASSERT_EQ(idg->size(), expected_size) << "Unexpected IdGroup size: " << toString(idg)
                                            << ", tensor: " << tv->toString();
    }
  }
}

} // namespace nvfuser
