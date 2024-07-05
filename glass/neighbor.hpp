#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

#include "glass/memory.hpp"

namespace glass {

namespace searcher {

template <typename Block = uint64_t> struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int nbytes;
  Block *data;
  explicit Bitset(int n)
      : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((uint64_t *)alloc64B(nbytes)) {
    memset(data, 0, nbytes);
  }
  ~Bitset() { free(data); }
  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }
  bool get(int i) {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }

  void *block_address(int i) { return data + i / block_size; }
};

// 邻居结构体定义
template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};


// 大根堆实现
template <typename dist_t> struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (size < capacity) {
      pool[size] = {u, dist};
      // 插入
      std::push_heap(pool.begin(), pool.begin() + ++size); // 默认大根堆
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  std::pair<int,dist_t> pop() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    return {pool[size].id,pool[size].distance};
  }
  // 下沉
  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < size;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < size && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int size = 0, capacity;

  // 实现了align_alloc的vector
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

// 最小最大堆
// O(1) 查询
// O(logn) 插入
// 节省空间
template <typename dist_t> struct MinMaxHeap {
  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id >= 0) {
        size--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    size++;
    return true;
  }
  dist_t max() { return pool[0].distance; }
  void clear() { size = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i)
      ;
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --size;
    return ret;
  }

  int size = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

// 有序数组, 支持二分查找
// 存的是原始id . 为什么需要&上一个mask呢? 已解决
// cur? pop以及insert会更改cur. 明白了,cur是当前pool里面最小元素的下标吧？
template <typename dist_t> struct LinearPool {
  LinearPool(int n, int capacity, int = 0)
      : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) { // 右边都是大于的
        hi = mid;
      } else {
        lo = mid + 1; // 左边都是小于等于的
      }
    }
    return lo; // 返回小于等于dist的最后一个位置
  }

  bool insert(int u, dist_t dist) {
    // 满了，且比pool内所有元素的距离不小
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    // data[lo] ~ data[size] 全都往后移一位
    // 然后data[lo] 赋值为新插入的
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>)); //
    data_[lo] = {u, dist};
    if (size_ < capacity_) { // 小于capacity才更新size
      size_++;
    }
    if (lo < cur_) { // 这个cur_具体作用？不太理解它的实现方式。插入的时候要更新当前指针？
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id); // 这里原始id会变。
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647; //32位最大整数
  int get_id(int id) const { return id & kMask; } // 知道了为什么&kMask,因为有些set_checked的时候变了，需要恢复
  void set_checked(int &id) { id |= 1 << 31; } // pop后最高位至1;tomb标志?(1<<31)-1是int最大值哦
  bool is_checked(int id) { return id >> 31 & 1; } // 然后检查这个tomb tag?
  dist_t get_dist(int i)  const {return data_[i].distance;}
  int nb, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> data_;
  Bitset<uint64_t> vis;
};

template <typename dist_t> struct HeapPool {
  HeapPool(int n, int capacity, int topk)
      : nb(n), capacity_(capacity), candidates(capacity), retset(topk), vis(n) {
  }
  bool insert(int u, dist_t dist) {
    retset.push(u, dist);
    return candidates.push(u, dist);
  }
  int pop() { return candidates.pop_min(); }
  bool has_next() const { return candidates.size > 0; }
  int id(int i) const { return retset.pool[i].id; }
  int capacity() const { return capacity_; }
  int nb, size_ = 0, capacity_;
  MinMaxHeap<dist_t> candidates;
  MaxHeap<dist_t> retset;
  Bitset<uint64_t> vis;
};

} // namespace searcher

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct Node {
  int id;
  float distance;

  Node() = default;
  Node(int id, float distance) : id(id), distance(distance) {}

  inline bool operator<(const Node &other) const {
    return distance < other.distance;
  }
};

inline int insert_into_pool(Neighbor *addr, int K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance) {
      right = mid;
    } else {
      left = mid;
    }
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) {
      break;
    }
    if (addr[left].id == nn.id) {
      return K + 1;
    }
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) {
    return K + 1;
  }
  memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

} // namespace glass
