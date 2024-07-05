#pragma once

#include "glass/common.hpp"
#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/fp32_quant.hpp"
#include "glass/simd/distance.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace glass {

// 8bit 0-255
template <Metric metric, int DIM = 0> struct SQ8Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 16; // 保证是16的倍数。
  // 数据的纬度,对齐纬度？ myabe。。。。。
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr; // 指向存储量化后数据的指针
  std::vector<float> mx, mi, dif; // 每个纬度的最大值,每个纬度的最小值，以及两者之间的差值

  SQ8Quantizer() = default;

  explicit SQ8Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        mx(d_align, -HUGE_VALF), mi(d_align, HUGE_VALF), dif(d_align) {}

  ~SQ8Quantizer() { free(codes); }

  void train(const float *data, int n) { 
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < d; ++j) {
        mx[j] = std::max(mx[j], data[i * d + j]); // 计算每一维的最大值
        mi[j] = std::min(mi[j], data[i * d + j]); // 计算每一维的最小值
      }
    }
    for (int64_t j = 0; j < d; ++j) {
      dif[j] = mx[j] - mi[j]; // 记录每一位最大值与最小值的差距
    }
    for (int64_t j = d; j < d_align; ++j) {
      dif[j] = mx[j] = mi[j] = 0; // 后面 d-d_align为0?
    }
    // code_size是对齐后的纬度
    codes = (char *)alloc2M((size_t)n * code_size); // 分配内存
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i)); // 每一维量化成8bit
    }
  }

  char *get_data(int u) const { return codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) { // 每一维量化成8bit表示
      float x = (from[j] - mi[j]) / dif[j];
      if (x < 0) {
        x = 0.0;
      }
      if (x > 1.0) {
        x = 1.0;
      }
      uint8_t y = x * 255;
      to[j] = y;
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float * /**q*/, std::pair<int, float> *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i].first = pool.id(i);
      dst[i].second = pool.get_dist(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func =
        metric == Metric::L2 ? L2SqrSQ8_ext : IPSQ8_ext;
    const SQ8Quantizer &quant;
    float *q;
    const float *mi, *dif;
    Computer(const SQ8Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)),
          mi(quant.mi.data()), dif(quant.dif.data()) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align, mi,
                       dif);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass
