#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/memory.hpp"

namespace glass {

constexpr int EMPTY_ID = -1;

/***********************************************************
 * Graph structure to store a graph.
 *
 * It is represented by an adjacency matrix `data`, where
 * data[i, j] is the j-th neighbor of node i.
 ***********************************************************/
template <typename node_t> 
struct Graph {

  int N; // total number of nodes
  int K; // number of neighbors per node

  node_t *data = nullptr; // the flattened adjacency matrix, size N-by-K

  std::unique_ptr<HNSWInitializer> initializer = nullptr; // 为了适配HNSW算法,NSG中用不到

  std::vector<int> eps; // HNSW需要多个eps,NSG只需要一个(一般来说)

  Graph() = default;
  
  Graph(node_t *edges, int N, int K) : N(N), K(K), data(edges) {}

  // construct an empty graph
  // NOTE: the newly allocated data needs to be destroyed at destruction time
  Graph(int N, int K)
      : N(N), K(K), data((node_t *)alloc2M((size_t)N * K * sizeof(node_t))) {}

  // copy constructor
  Graph(const Graph &g) : Graph(g.N, g.K) {
    this->eps = g.eps;

    // 为啥不用memcpy ?
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        at(i, j) = g.at(i, j);
      }
    }
    if (g.initializer) {
      initializer = std::make_unique<HNSWInitializer>(*g.initializer);
    }
  }

  // 跟前面构造一个空图是一样的？
  // 区别是这里memset成了-1
  void init(int N, int K) {
    data = (node_t *)alloc2M((size_t)N * K * sizeof(node_t));
    std::memset(data, -1, N * K * sizeof(node_t));
    this->K = K;
    this->N = N;
  }

  ~Graph() { free(data); }

  const int *edges(int u) const { return data + K * u; }

  int *edges(int u) { return data + K * u; }

  node_t at(int i, int j) const { return data[i * K + j]; }

  node_t &at(int i, int j) { return data[i * K + j]; }

  void prefetch(int u, int lines) const {
    // edges(u)返回u邻居数组指针
    mem_prefetch((char *)edges(u), lines);
  }

  template <typename Pool, typename Computer>
  void initialize_search(Pool &pool, const Computer &computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    } else {
      for (auto ep : eps) {
        pool.insert(ep, computer(ep));
      }
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same_v<node_t, int32_t>);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, N * K * 4);
    if (initializer) {
      initializer->save(writer);
    }
    printf("Graph Saving done\n");
  }

  void load(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)alloc2M((size_t)N * K * 4);
    reader.read((char *)data, N * K * 4);
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
    printf("Graph Loding done\n");
  }
};

} // namespace glass