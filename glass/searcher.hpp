#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "glass/common.hpp"
#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/quant.hpp"
#include "glass/utils.hpp"

namespace glass {

struct SearcherBase {
  virtual void SetData(const float *data, int n, int dim) = 0;
  virtual void Optimize(int num_threads = 0) = 0;
  virtual void Search(const float *q, int k, std::pair<int,float> *dst) const = 0;
  virtual void SetEf(int ef) = 0;
  virtual ~SearcherBase() = default;
};

template <typename Quantizer> struct Searcher : public SearcherBase {

  int d;
  int nb;
  Graph<int> graph;
  Quantizer quant;

  // Search parameters
  int ef = 32;

  // Memory prefetch parameters
  int po = 1; // 预取邻居数
  int pl = 1; // 预取cacheline

  // Optimization parameters
  constexpr static int kOptimizePoints = 1000;
  constexpr static int kTryPos = 10;
  constexpr static int kTryPls = 5;
  constexpr static int kTryK = 10;
  int sample_points_num;
  std::vector<float> optimize_queries;
  const int graph_po;

  Searcher(const Graph<int> &graph) : graph(graph), graph_po(graph.K / 16) {}

  // 
  void SetData(const float *data, int n, int dim) override {
    this->nb = n;
    this->d = dim;

    // 量化
    quant = Quantizer(d);
    quant.train(data, n);

    // 随机采样
    sample_points_num = std::min(kOptimizePoints, nb - 1);
    std::vector<int> sample_points(sample_points_num);
    std::mt19937 rng;
    GenRandom(rng, sample_points.data(), sample_points_num, nb);
    optimize_queries.resize(sample_points_num * d);
    for (int i = 0; i < sample_points_num; ++i) {
      memcpy(optimize_queries.data() + i * d, data + sample_points[i] * d,
             d * sizeof(float));
    }
  }

  void SetEf(int ef) override { this->ef = ef; }

  // 优化预取参数,加快查询时间
  void Optimize(int num_threads = 0) override {
    if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
    }
    std::vector<int> try_pos(std::min(kTryPos, graph.K));
    std::vector<int> try_pls(
        std::min(kTryPls, (int)upper_div(quant.code_size, 64)));
    std::iota(try_pos.begin(), try_pos.end(), 1);
    std::iota(try_pls.begin(), try_pls.end(), 1);

    std::vector<std::pair<int, float>> dummy_dst(kTryK);
    printf("=============Start optimization=============\n");
    { // warmup
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
      for (int i = 0; i < sample_points_num; ++i) {
        Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
      }
    }

    float min_ela = std::numeric_limits<float>::max();
    int best_po = 0, best_pl = 0;
    for (auto try_po : try_pos) {
      for (auto try_pl : try_pls) {
        this->po = try_po;
        this->pl = try_pl;
        auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < sample_points_num; ++i) {
          Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
        }

        auto ed = std::chrono::high_resolution_clock::now();
        auto ela = std::chrono::duration<double>(ed - st).count();
        if (ela < min_ela) {
          min_ela = ela;
          best_po = try_po;
          best_pl = try_pl;
        }
      }
    }
    this->po = 1;
    this->pl = 1;
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < sample_points_num; ++i) {
      Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
    }
    auto ed = std::chrono::high_resolution_clock::now();
    float baseline_ela = std::chrono::duration<double>(ed - st).count();
    printf("settint best po = %d, best pl = %d\n"
           "gaining %.2f%% performance improvement\n============="
           "Done optimization=============\n",
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
    this->po = best_po;
    this->pl = best_pl;
  }


  /**
  q:查询向量
  k:topk
  dst:存储结果
   */
  void Search(const float *q, int k, std::pair<int,float> *dst) const override {
    auto computer = quant.get_computer(q);
    // 构建顺序数组
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type>
        pool(nb, std::max(k, ef), k);
      
    graph.initialize_search(pool, computer);
    SearchImpl(pool, computer); // 注意id为什么要& mask?
    quant.reorder(pool, q, dst, k); // 这里可以把距离存下来,同时只取前k个
  }

  template <typename Pool, typename Computer>
  void SearchImpl(Pool &pool, const Computer &computer) const {
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) { // 这边是预取到cache中
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == -1) {
          break;
        }
        if (i + po < graph.K && graph.at(u, i + po) != -1) { // 又预取
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        if (pool.vis.get(v)) { // 访问过了，跳过；
          continue;
        }
        pool.vis.set(v);
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist); // 加入pool中
      }
    }
  }
};

inline std::unique_ptr<SearcherBase> create_searcher(const Graph<int> &graph,
                                                     const std::string &metric,
                                                     int level = 1) {
  auto m = metric_map[metric];
  // level 控制量化程度
  if (level == 0) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<FP32Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<FP32Quantizer<Metric::IP>>>(graph);
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (level == 1) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ8Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<SQ8Quantizer<Metric::IP>>>(graph);
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (level == 2) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ4Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<SQ4Quantizer<Metric::IP>>>(graph);
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else {
    printf("Quantizer type not supported\n");
    return nullptr;
  }
}

} // namespace glass
