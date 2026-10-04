// Batched AlphaZero-style MCTS.
//
// Mirrors mcts/gpu_mcts.py (GPUMCTS) semantics:
//   score(child) = -Q_eff + c_puct * P * sqrt(N_parent_eff) / (1 + N_child_eff)
//   virtual loss = +1 visit and +1.0 value on every path node during selection
//   backup signs alternate from the leaf (+value at leaf, -value at parent...)
//   leaf values: NN value, or exact terminal values (mate -1, draws 0)
//   third-repetition leaves become terminal with draw value
//   Dirichlet noise mixed into root priors after the root expansion
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

#include "rules.hpp"

namespace azcpp {

// planes [n, 18*8*8] row-major float32 in; logits [n*4544] and values [n] out.
using EvalFn = std::function<void(const float*, int, float*, float*)>;

struct SearchParams {
  int num_simulations = 400;
  float c_puct = 1.5f;
  float dirichlet_alpha = -1.0f;  // negative: no noise
  float dirichlet_epsilon = 0.25f;
  int batch_size = 32;
};

struct Edge {
  int child;
  int action;
  float prior;
};

struct Node {
  Position state;
  int parent = -1;
  int visit = 0;
  float value_sum = 0.0f;
  int vvisit = 0;       // virtual statistics (selection batches only)
  float vvalue = 0.0f;
  bool expanded = false;
  bool terminal = false;  // draw-by-rule marker or no legal moves
  int edge_start = -1;
  int edge_count = 0;
};

class BatchMcts {
 public:
  explicit BatchMcts(std::uint64_t seed = 42) : rng_(seed) {}

  void set_seed(std::uint64_t seed) { rng_.seed(seed); }

  // history: [games * history_len] flat repetition hashes through each root.
  // history_len == 0 disables in-tree repetition detection (mirrors GPUMCTS).
  void search(const std::vector<Position>& roots,
              const std::int64_t* history, int history_len,
              const SearchParams& params, const EvalFn& eval);

  // Normalized root visit policy over all 4544 actions: [games * 4544].
  const std::vector<float>& root_policy() const { return root_policy_; }

  // One action per root by temperature (0 => argmax; >0 => sampling).
  std::vector<int> select_actions(float temperature);

  // Child positions reached by one action per root.
  std::vector<Position> advance(const std::vector<int>& actions) const;

  const std::vector<Position>& roots() const { return roots_; }
  int games() const { return static_cast<int>(roots_.size()); }
  long total_duplicate_leaves() const { return total_duplicates_; }

 private:
  void ensure_capacity(int games, int sims);
  void expand(const std::vector<int>& node_ids, const float* logits);
  void eval_states(const std::vector<int>& node_ids, const EvalFn& eval,
                   std::vector<float>& logits_out, std::vector<float>& values_out);
  int select_leaf(int root, std::vector<int>& path);
  void apply_virtual(const std::vector<int>& path, float delta);
  void add_dirichlet(int games, float alpha, float epsilon);
  void compute_root_policy(int games);
  std::vector<float> node_planes(const std::vector<int>& node_ids) const;

  float c_puct_ = 1.5f;

  std::uint64_t rng_u64() { return rng_(); }
  float rng_uniform() {
    return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_);
  }

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<Position> roots_;
  std::vector<int> root_ids_;
  std::vector<float> root_policy_;
  std::mt19937_64 rng_;

  std::vector<float> planes_buf_;
  std::vector<float> logits_buf_;
  std::vector<float> values_buf_;

  long total_duplicates_ = 0;
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline std::vector<float> BatchMcts::node_planes(const std::vector<int>& node_ids) const {
  std::vector<float> out(node_ids.size() * kNumModelPlanes * 64);
  for (size_t i = 0; i < node_ids.size(); ++i) {
    nodes_[node_ids[i]].state.encode_planes(out.data() + i * kNumModelPlanes * 64);
  }
  return out;
}

inline void BatchMcts::eval_states(const std::vector<int>& node_ids,
                                   const EvalFn& eval,
                                   std::vector<float>& logits_out,
                                   std::vector<float>& values_out) {
  const int n = static_cast<int>(node_ids.size());
  planes_buf_.resize(static_cast<size_t>(n) * kNumModelPlanes * 64);
  for (int i = 0; i < n; ++i) {
    nodes_[node_ids[i]].state.encode_planes(
        planes_buf_.data() + static_cast<size_t>(i) * kNumModelPlanes * 64);
  }
  logits_buf_.assign(static_cast<size_t>(n) * kActionSpace, 0.0f);
  values_buf_.assign(n, 0.0f);
  eval(planes_buf_.data(), n, logits_buf_.data(), values_buf_.data());
  logits_out = std::move(logits_buf_);
  values_out = std::move(values_buf_);
}

inline void BatchMcts::ensure_capacity(int games, int sims) {
  // Typical legal-move counts are ~30-35; leave headroom so vector growth
  // stays rare.  Indices remain valid across reallocation regardless.
  const size_t expected =
      static_cast<size_t>(games) * (1 + static_cast<size_t>(sims) * 40);
  nodes_.clear();
  edges_.clear();
  nodes_.reserve(expected);
  edges_.reserve(expected);
}

inline void BatchMcts::expand(const std::vector<int>& node_ids, const float* logits) {
  for (size_t idx = 0; idx < node_ids.size(); ++idx) {
    const int id = node_ids[idx];
    assert(!nodes_[id].expanded);  // unique, non-terminal leaves only

    std::vector<int> legal;
    nodes_[id].state.legal_actions(legal);
    std::sort(legal.begin(), legal.end());
    const int count = static_cast<int>(legal.size());

    // Priors: softmax over the legal subset of the 4544 logits.
    std::vector<float> priors(count, 0.0f);
    if (count > 0) {
      if (logits != nullptr) {
        const float* row = logits + static_cast<size_t>(idx) * kActionSpace;
        float mx = -1.0e30f;
        for (int a : legal) mx = std::max(mx, row[a]);
        float sum = 0.0f;
        for (int k = 0; k < count; ++k) {
          const float e = std::exp(row[legal[k]] - mx);
          priors[k] = e;
          sum += e;
        }
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int k = 0; k < count; ++k) priors[k] *= inv;
      } else {
        const float inv = 1.0f / static_cast<float>(count);
        for (int k = 0; k < count; ++k) priors[k] = inv;
      }
    }

    const int edge_start = static_cast<int>(edges_.size());
    Position parent_state = nodes_[id].state;

    for (int k = 0; k < count; ++k) {
      Position child = parent_state;
      child.apply(legal[k]);
      Node cn;
      cn.state = std::move(child);
      cn.parent = id;
      cn.terminal = (cn.state.halfmove >= 100 || cn.state.insufficient_material());
      const int cid = static_cast<int>(nodes_.size());
      nodes_.push_back(cn);
      edges_.push_back(Edge{cid, legal[k], priors[k]});
    }

    // Re-fetch by index: nodes_ may have reallocated above.
    Node& n = nodes_[id];
    n.edge_start = edge_start;
    n.edge_count = count;
    n.terminal = (count == 0);
    n.expanded = true;
  }
}

inline int BatchMcts::select_leaf(int root, std::vector<int>& path) {
  path.clear();
  path.push_back(root);
  int cur = root;
  while (true) {
    const Node& n = nodes_[cur];
    if (!n.expanded || n.terminal || n.edge_count <= 0) break;
    const float parent_eff =
        static_cast<float>(n.visit + n.vvisit);
    const float sqrt_parent = std::sqrt(std::max(parent_eff, 1.0f));
    float best_score = -1.0e30f;
    int best_edge = -1;
    for (int e = n.edge_start; e < n.edge_start + n.edge_count; ++e) {
      const Edge& edge = edges_[e];
      const Node& child = nodes_[edge.child];
      const float eff_v = static_cast<float>(child.visit + child.vvisit);
      const float q = eff_v > 0.0f ? (child.value_sum + child.vvalue) / eff_v : 0.0f;
      const float score =
          -q + c_puct_ * edge.prior * sqrt_parent / (1.0f + eff_v);
      if (score > best_score) {  // strict >: first (lowest action id) wins ties
        best_score = score;
        best_edge = e;
      }
    }
    if (best_edge < 0) break;
    cur = edges_[best_edge].child;
    path.push_back(cur);
  }
  return cur;
}

inline void BatchMcts::apply_virtual(const std::vector<int>& path, float delta) {
  for (int id : path) {
    nodes_[id].vvisit += (delta > 0 ? 1 : -1);
    nodes_[id].vvalue += delta;
  }
}

inline void BatchMcts::add_dirichlet(int games, float alpha, float epsilon) {
  std::gamma_distribution<float> gamma(alpha, 1.0f);
  for (int g = 0; g < games; ++g) {
    const int id = root_ids_[g];
    Node& n = nodes_[id];
    if (n.edge_count <= 0) continue;
    float noise_sum = 0.0f;
    std::vector<float> noise(n.edge_count);
    for (int e = 0; e < n.edge_count; ++e) {
      noise[e] = gamma(rng_);
      noise_sum += noise[e];
    }
    const float inv = noise_sum > 0.0f ? 1.0f / noise_sum : 0.0f;
    for (int e = 0; e < n.edge_count; ++e) {
      Edge& edge = edges_[n.edge_start + e];
      edge.prior = (1.0f - epsilon) * edge.prior +
                   epsilon * (noise[e] * inv);
    }
  }
}

inline void BatchMcts::compute_root_policy(int games) {
  root_policy_.assign(static_cast<size_t>(games) * kActionSpace, 0.0f);
  for (int g = 0; g < games; ++g) {
    const Node& n = nodes_[root_ids_[g]];
    float total = 0.0f;
    std::vector<float> probs(n.edge_count);
    for (int e = 0; e < n.edge_count; ++e) {
      const Node& child = nodes_[edges_[n.edge_start + e].child];
      probs[e] = static_cast<float>(child.visit);
      total += probs[e];
    }
    if (total <= 0.0f) continue;
    const float inv = 1.0f / total;
    float* row = root_policy_.data() + static_cast<size_t>(g) * kActionSpace;
    for (int e = 0; e < n.edge_count; ++e) {
      row[edges_[n.edge_start + e].action] += probs[e] * inv;
    }
  }
}

inline void BatchMcts::search(const std::vector<Position>& roots,
                              const std::int64_t* history, int history_len,
                              const SearchParams& params, const EvalFn& eval) {
  const int games = static_cast<int>(roots.size());
  roots_ = roots;
  c_puct_ = params.c_puct;
  total_duplicates_ = 0;
  if (games <= 0) {
    root_ids_.clear();
    root_policy_.clear();
    return;
  }

  const int sims = std::max(0, params.num_simulations);
  ensure_capacity(games, std::max(1, sims));

  root_ids_.resize(games);
  for (int g = 0; g < games; ++g) {
    Node n;
    n.state = roots[g];
    root_ids_[g] = static_cast<int>(nodes_.size());
    nodes_.push_back(n);
  }

  // Evaluate + expand the roots.
  std::vector<float> logits, values;
  eval_states(root_ids_, eval, logits, values);
  expand(root_ids_, logits.data());

  if (params.dirichlet_alpha > 0.0f) {
    add_dirichlet(games, params.dirichlet_alpha, params.dirichlet_epsilon);
  }
  if (sims > 0) {

  const int bsz_cap = std::max(1, params.batch_size);
  int remaining = sims;

  std::vector<int> flat_leaves;
  std::vector<std::vector<int>> paths;
  std::vector<int> terminal_leaf;
  std::vector<float> leaf_values;

  while (remaining > 0) {
    const int bsz = std::min(bsz_cap, remaining);
    paths.assign(static_cast<size_t>(bsz) * games, {});

    // Select bsz leaves per game, applying virtual loss immediately so later
    // selections in this round diversify (same order as GPUMCTS batches).
    for (int s = 0; s < bsz; ++s) {
      for (int g = 0; g < games; ++g) {
        std::vector<int>& path = paths[static_cast<size_t>(s) * games + g];
        select_leaf(root_ids_[g], path);
        apply_virtual(path, 1.0f);
      }
    }
    // Remove virtual statistics before real backups.
    for (auto& path : paths) apply_virtual(path, -1.0f);
#ifndef NDEBUG
    for (const Node& n : nodes_) {
      assert(n.vvisit == 0 && n.vvalue == 0.0f);
    }
#endif

    // Flatten leaves and detect duplicates (diagnostic only).
    flat_leaves.clear();
    flat_leaves.reserve(static_cast<size_t>(bsz) * games);
    for (auto& path : paths) flat_leaves.push_back(path.back());
    {
      std::unordered_map<int, int> seen;
      seen.reserve(flat_leaves.size());
      for (int leaf : flat_leaves) {
        auto it = seen.find(leaf);
        if (it == seen.end()) seen.emplace(leaf, 1);
        else { ++it->second; ++total_duplicates_; }
      }
    }

    // Third-repetition terminal marking (needs game history through the root).
    if (history != nullptr && history_len > 0) {
      for (int s = 0; s < bsz; ++s) {
        for (int g = 0; g < games; ++g) {
          std::vector<int>& path = paths[static_cast<size_t>(s) * games + g];
          const int leaf = path.back();
          const U64 leaf_hash = nodes_[leaf].state.state_hash();
          std::int64_t historical = 0;
          const std::int64_t* row = history + static_cast<size_t>(g) * history_len;
          for (int i = 0; i < history_len; ++i) {
            if (static_cast<std::uint64_t>(row[i]) == leaf_hash) ++historical;
          }
          std::int64_t branch = 0;
          for (size_t k = 1; k < path.size(); ++k) {  // skip the root (in history)
            if (nodes_[path[k]].state.state_hash() == leaf_hash) ++branch;
          }
          if (historical + branch >= 3) nodes_[leaf].terminal = true;
        }
      }
    }

    // Partition terminal vs evaluable leaves; evaluate uniques in one batch.
    const int total = bsz * games;
    terminal_leaf.assign(total, 0);
    leaf_values.assign(total, 0.0f);
    std::vector<int> unique;
    unique.reserve(total);
    std::vector<int> uniq_of(total, -1);  // per leaf: row in `unique` or -1
    std::unordered_map<int, int> uniq_index;  // node id -> unique row
    for (int i = 0; i < total; ++i) {
      const int leaf = flat_leaves[i];
      if (nodes_[leaf].terminal) {
        terminal_leaf[i] = 1;
        leaf_values[i] = nodes_[leaf].state.leaf_terminal_value();
      } else {
        auto it = uniq_index.find(leaf);
        if (it == uniq_index.end()) {
          const int row = static_cast<int>(unique.size());
          uniq_index.emplace(leaf, row);
          unique.push_back(leaf);
          uniq_of[i] = row;
        } else {
          uniq_of[i] = it->second;
        }
      }
    }

    if (!unique.empty()) {
      std::vector<float> u_logits, u_values;
      eval_states(unique, eval, u_logits, u_values);
      expand(unique, u_logits.data());
      std::vector<float> value_of_unique(unique.size(), 0.0f);
      for (size_t u = 0; u < unique.size(); ++u) {
        const Position& st = nodes_[unique[u]].state;
        const int count = st.legal_action_count();
        const bool newly_terminal =
            count == 0 || st.halfmove >= 100 || st.insufficient_material();
        value_of_unique[u] =
            newly_terminal ? st.leaf_terminal_value() : u_values[u];
      }
      for (int i = 0; i < total; ++i) {
        if (uniq_of[i] >= 0) leaf_values[i] = value_of_unique[uniq_of[i]];
      }
    }

    // Backup every path with its leaf value, alternating signs.
    for (int s = 0; s < bsz; ++s) {
      for (int g = 0; g < games; ++g) {
        const size_t i = static_cast<size_t>(s) * games + g;
        const float v = leaf_values[i];
        const std::vector<int>& path = paths[i];
        int dist = 0;
        for (auto it = path.rbegin(); it != path.rend(); ++it, ++dist) {
          Node& n = nodes_[*it];
          n.visit += 1;
          n.value_sum += (dist % 2 == 0) ? v : -v;
        }
      }
    }

    remaining -= bsz;
  }  // while (remaining > 0)
  }  // if (sims > 0)

  compute_root_policy(games);
}

inline std::vector<int> BatchMcts::select_actions(float temperature) {
  const int ng = games();
  std::vector<int> actions(ng, 0);
  for (int g = 0; g < ng; ++g) {
    const Node& n = nodes_[root_ids_[g]];
    if (n.edge_count <= 0) { actions[g] = 0; continue; }
    if (temperature <= 0.0f) {
      float best = -1.0f;
      int best_e = 0;
      for (int e = 0; e < n.edge_count; ++e) {
        const float v =
            static_cast<float>(nodes_[edges_[n.edge_start + e].child].visit);
        if (v > best) { best = v; best_e = e; }
      }
      actions[g] = edges_[n.edge_start + best_e].action;
    } else {
      const float inv_t = 1.0f / temperature;
      float total = 0.0f;
      std::vector<float> weights(n.edge_count);
      for (int e = 0; e < n.edge_count; ++e) {
        const float v =
            static_cast<float>(nodes_[edges_[n.edge_start + e].child].visit);
        weights[e] = v > 0.0f ? std::pow(v, inv_t) : 0.0f;
        total += weights[e];
      }
      int chosen = n.edge_count - 1;
      if (total > 0.0f) {
        float r = rng_uniform() * total;
        for (int e = 0; e < n.edge_count; ++e) {
          r -= weights[e];
          if (r <= 0.0f) { chosen = e; break; }
        }
      }
      actions[g] = edges_[n.edge_start + chosen].action;
    }
  }
  return actions;
}

inline std::vector<Position> BatchMcts::advance(
    const std::vector<int>& actions) const {
  const int ng = games();
  std::vector<Position> out(ng);
  for (int g = 0; g < ng; ++g) {
    const Node& n = nodes_[root_ids_[g]];
    int child = -1;
    for (int e = 0; e < n.edge_count; ++e) {
      const Edge& edge = edges_[n.edge_start + e];
      if (edge.action == actions[g]) { child = edge.child; break; }
    }
    out[g] = (child >= 0) ? nodes_[child].state : n.state;
  }
  return out;
}

}  // namespace azcpp
