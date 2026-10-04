// pybind11 bindings: rules-level Position API + batched MCTS with a
// Python-provided neural-network evaluation callback.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "azcpp/mcts.hpp"
#include "azcpp/rules.hpp"

namespace py = pybind11;
using namespace azcpp;

namespace {

constexpr int kPlanesFlat = kNumModelPlanes * 8 * 8;

// c_style | forcecast: strided views / other dtypes (e.g. numpy bool_ from
// torch) are converted to contiguous canonical arrays before we touch memory.
template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

void check_state_shapes(const CArray<std::int64_t>& pieces,
                        const CArray<std::int8_t>& turn,
                        const CArray<std::int16_t>& castling,
                        const CArray<std::int16_t>& ep,
                        const CArray<std::int16_t>& halfmove,
                        const CArray<std::int16_t>& fullmove) {
  const py::buffer_info pc = pieces.request();
  if (pc.ndim != 2 || pc.shape[1] != 12) {
    throw std::invalid_argument("pieces must have shape [games, 12]");
  }
  const py::ssize_t games = pc.shape[0];
  const py::buffer_info reqs[] = {turn.request(), castling.request(),
                                  ep.request(), halfmove.request(),
                                  fullmove.request()};
  for (const auto& r : reqs) {
    if (r.ndim != 1 || r.shape[0] != games) {
      throw std::invalid_argument("state arrays must all have shape [games]");
    }
  }
}

std::vector<Position> positions_from_arrays(const CArray<std::int64_t>& pieces,
                                            const CArray<std::int8_t>& turn,
                                            const CArray<std::int16_t>& castling,
                                            const CArray<std::int16_t>& ep,
                                            const CArray<std::int16_t>& halfmove,
                                            const CArray<std::int16_t>& fullmove) {
  check_state_shapes(pieces, turn, castling, ep, halfmove, fullmove);
  const size_t games = static_cast<size_t>(pieces.request().shape[0]);
  std::vector<Position> out;
  out.reserve(games);
  for (size_t g = 0; g < games; ++g) {
    Position p;
    for (int i = 0; i < 12; ++i) {
      p.occ[i] = static_cast<U64>(
          static_cast<std::uint64_t>(*pieces.data(g, i)));
    }
    p.turn = static_cast<std::int16_t>(turn.data(g)[0] ? 1 : 0);
    p.castling = castling.data(g)[0];
    p.ep = ep.data(g)[0];
    p.halfmove = halfmove.data(g)[0];
    p.fullmove = fullmove.data(g)[0];
    out.push_back(p);
  }
  return out;
}

py::tuple arrays_from_positions(const std::vector<Position>& positions) {
  const size_t games = positions.size();
  py::array_t<std::int64_t> pieces({static_cast<py::ssize_t>(games),
                                    static_cast<py::ssize_t>(12)});
  py::array_t<std::int8_t> turn(static_cast<py::ssize_t>(games));
  py::array_t<std::int16_t> castling(static_cast<py::ssize_t>(games));
  py::array_t<std::int16_t> ep(static_cast<py::ssize_t>(games));
  py::array_t<std::int16_t> halfmove(static_cast<py::ssize_t>(games));
  py::array_t<std::int16_t> fullmove(static_cast<py::ssize_t>(games));
  auto pc = pieces.mutable_unchecked<2>();
  auto tu = turn.mutable_unchecked<1>();
  auto ca = castling.mutable_unchecked<1>();
  auto epa = ep.mutable_unchecked<1>();
  auto hm = halfmove.mutable_unchecked<1>();
  auto fm = fullmove.mutable_unchecked<1>();
  for (size_t g = 0; g < games; ++g) {
    const Position& p = positions[g];
    for (int i = 0; i < 12; ++i) {
      pc(g, i) = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(p.occ[i]));
    }
    tu(g) = static_cast<std::int8_t>(p.turn);
    ca(g) = p.castling;
    epa(g) = p.ep;
    hm(g) = p.halfmove;
    fm(g) = p.fullmove;
  }
  return py::make_tuple(pieces, turn, castling, ep, halfmove, fullmove);
}

using ContigF32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

EvalFn make_eval_fn(py::function fn) {
  return [fn = std::move(fn)](const float* planes, int n, float* logits,
                              float* values) {
    // Non-owning view valid for the duration of the synchronous call; the
    // callback must not retain it.
    py::array_t<float> arr({static_cast<py::ssize_t>(n),
                            static_cast<py::ssize_t>(kPlanesFlat)},
                           planes);
    py::object result = fn(arr);
    py::tuple pair = result.cast<py::tuple>();
    if (pair.size() != 2) {
      throw std::invalid_argument("eval callback must return (logits, values)");
    }
    ContigF32 lg = pair[0].cast<ContigF32>();
    ContigF32 vl = pair[1].cast<ContigF32>();
    auto lgb = lg.request();
    auto vlb = vl.request();
    if (lgb.size != static_cast<py::ssize_t>(n) * kActionSpace) {
      throw std::invalid_argument("logits must have shape [n, 4544]");
    }
    if (vlb.size != static_cast<py::ssize_t>(n)) {
      throw std::invalid_argument("values must have shape [n]");
    }
    std::memcpy(logits, lgb.ptr,
                sizeof(float) * static_cast<size_t>(n) * kActionSpace);
    std::memcpy(values, vlb.ptr, sizeof(float) * static_cast<size_t>(n));
  };
}

class PyBatchMcts {
 public:
  void set_seed(std::uint64_t seed) { inner_.set_seed(seed); }
  void set_c_puct(float c) { c_puct_ = c; }

  void search(CArray<std::int64_t> pieces, CArray<std::int8_t> turn,
              CArray<std::int16_t> castling, CArray<std::int16_t> ep,
              CArray<std::int16_t> halfmove, CArray<std::int16_t> fullmove,
              CArray<std::int64_t> history, int num_simulations,
              float dirichlet_alpha, float dirichlet_epsilon, int batch_size,
              py::function eval_fn) {
    const auto roots = positions_from_arrays(pieces, turn, castling, ep,
                                             halfmove, fullmove);
    auto hb = history.request();
    if (hb.ndim != 2) {
      throw std::invalid_argument(
          "repetition history must have shape [games, positions]");
    }
    if (hb.shape[0] != static_cast<py::ssize_t>(roots.size())) {
      throw std::invalid_argument("history rows must equal game count");
    }
    const int hist_len = static_cast<int>(hb.shape[1]);
    const std::int64_t* hist_ptr =
        hist_len > 0 ? static_cast<const std::int64_t*>(hb.ptr) : nullptr;

    SearchParams params;
    params.num_simulations = num_simulations;
    params.c_puct = c_puct_;
    params.dirichlet_alpha = dirichlet_alpha;
    params.dirichlet_epsilon = dirichlet_epsilon;
    params.batch_size = batch_size;

    inner_.search(roots, hist_ptr, hist_len, params, make_eval_fn(eval_fn));
  }

  py::array_t<float> root_visit_policy() const {
    const auto& policy = inner_.root_policy();
    const py::ssize_t games = static_cast<py::ssize_t>(inner_.games());
    py::array_t<float> out({games, static_cast<py::ssize_t>(kActionSpace)});
    if (!policy.empty()) {
      std::memcpy(out.mutable_data(), policy.data(),
                  sizeof(float) * policy.size());
    }
    return out;
  }

  py::array_t<std::int64_t> select_actions(float temperature) {
    const auto actions = inner_.select_actions(temperature);
    py::array_t<std::int64_t> out(static_cast<py::ssize_t>(actions.size()));
    auto u = out.mutable_unchecked<1>();
    for (size_t i = 0; i < actions.size(); ++i) {
      u(static_cast<py::ssize_t>(i)) = actions[i];
    }
    return out;
  }

  py::tuple advance(CArray<std::int64_t> actions) {
    auto ab = actions.request();
    if (ab.ndim != 1 ||
        ab.shape[0] != static_cast<py::ssize_t>(inner_.games())) {
      throw std::invalid_argument("actions must have shape [games]");
    }
    const std::int64_t* ap = static_cast<const std::int64_t*>(ab.ptr);
    std::vector<int> acts(inner_.games());
    for (size_t g = 0; g < acts.size(); ++g) {
      acts[g] = static_cast<int>(ap[g]);
    }
    return arrays_from_positions(inner_.advance(acts));
  }

  py::tuple root_positions() const {
    std::vector<Position> roots;
    roots.reserve(inner_.games());
    for (const auto& p : inner_.roots()) roots.push_back(p);
    return arrays_from_positions(roots);
  }

  long duplicate_leaves() const { return inner_.total_duplicate_leaves(); }

 private:
  BatchMcts inner_{42};
  float c_puct_ = 1.5f;
};

}  // namespace

PYBIND11_MODULE(az_cpp_mcts, m) {
  m.doc() = "C++ chess rules + batched MCTS for the clean AlphaZero pipeline.";

  m.def(
      "action_tables",
      []() {
        py::array_t<std::int64_t> from(kActionSpace);
        py::array_t<std::int64_t> to(kActionSpace);
        py::array_t<std::int64_t> promo(kActionSpace);
        auto f = from.mutable_unchecked<1>();
        auto t = to.mutable_unchecked<1>();
        auto p = promo.mutable_unchecked<1>();
        for (int a = 0; a < kActionSpace; ++a) {
          int fr, to_sq, pr;
          decode_action(a, fr, to_sq, pr);
          f(a) = fr;
          t(a) = to_sq;
          p(a) = pr;
        }
        return py::make_tuple(from, to, promo);
      },
      "The canonical 4544 (from, to, promo_code) tables.");

  m.def(
      "encode_action",
      [](int from, int to, int promo) { return encode_action(from, to, promo); });
  m.def(
      "decode_action",
      [](int action) {
        int f, t, p;
        decode_action(action, f, t, p);
        return py::make_tuple(f, t, p);
      });

  py::class_<Position>(m, "Position")
      .def(py::init<>())
      .def_static(
          "start",
          []() {
            return Position::from_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
          })
      .def_static("from_fen", &Position::from_fen)
      .def("to_fen", &Position::to_fen)
      .def(
          "legal_actions",
          [](const Position& p) {
            std::vector<int> legal;
            p.legal_actions(legal);
            py::array_t<std::int64_t> out(
                static_cast<py::ssize_t>(legal.size()));
            auto u = out.mutable_unchecked<1>();
            for (size_t i = 0; i < legal.size(); ++i) {
              u(static_cast<py::ssize_t>(i)) = legal[i];
            }
            return out;
          })
      .def(
          "apply",
          [](const Position& p, int action) {
            Position q = p;
            q.apply(action);
            return q;
          })
      .def("in_check", &Position::in_check)
      .def("insufficient_material", &Position::insufficient_material)
      .def(
          "terminal_info",
          [](const Position& p) {
            bool terminal = false;
            int value = 0;
            p.terminal_info(terminal, value);
            return py::make_tuple(terminal, value);
          })
      .def("state_hash",
           [](const Position& p) {
             return static_cast<std::int64_t>(p.state_hash());
           })
      .def(
          "to_model_input",
          [](const Position& p) {
            py::array_t<float> out({static_cast<py::ssize_t>(kNumModelPlanes),
                                    static_cast<py::ssize_t>(8),
                                    static_cast<py::ssize_t>(8)});
            p.encode_planes(out.mutable_data());
            return out;
          })
      .def_property_readonly("turn", [](const Position& p) { return int(p.turn); })
      .def_property_readonly("castling",
                             [](const Position& p) { return int(p.castling); })
      .def_property_readonly("ep", [](const Position& p) { return int(p.ep); })
      .def_property_readonly("halfmove",
                             [](const Position& p) { return int(p.halfmove); })
      .def_property_readonly("fullmove",
                             [](const Position& p) { return int(p.fullmove); })
      .def("pieces",
           [](const Position& p) {
             py::array_t<std::int64_t> out(12);
             auto u = out.mutable_unchecked<1>();
             for (int i = 0; i < 12; ++i) {
               u(i) = static_cast<std::int64_t>(
                   static_cast<std::uint64_t>(p.occ[i]));
             }
             return out;
           });

  py::class_<PyBatchMcts>(m, "BatchMcts")
      .def(py::init<>())
      .def("set_seed", &PyBatchMcts::set_seed)
      .def("set_c_puct", &PyBatchMcts::set_c_puct)
      .def("search", &PyBatchMcts::search, py::arg("pieces"), py::arg("turn"),
           py::arg("castling"), py::arg("ep_square"), py::arg("halfmove_clock"),
           py::arg("fullmove_number"), py::arg("repetition_history"),
           py::arg("num_simulations"), py::arg("dirichlet_alpha"),
           py::arg("dirichlet_epsilon"), py::arg("batch_size"),
           py::arg("eval_fn"))
      .def("root_visit_policy", &PyBatchMcts::root_visit_policy)
      .def("select_actions", &PyBatchMcts::select_actions,
           py::arg("temperature"))
      .def("advance", &PyBatchMcts::advance, py::arg("actions"))
      .def("root_positions", &PyBatchMcts::root_positions)
      .def("duplicate_leaves", &PyBatchMcts::duplicate_leaves);
}
