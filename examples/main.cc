#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <boost/program_options.hpp>

#include "glass/hnsw/hnsw.hpp"
#include "glass/nsg/nndescent.hpp"
#include "glass/nsg/nsg.hpp"
#include "glass/searcher.hpp"
namespace po = boost::program_options;

/**
 * @brief read data form file in fvecs format
 * @param filename 文件名
 * @param p 数据指针
 * @param n 数据行数
 * @param dim 数据维度
 */
template <typename T>
void load_fvecs(const char *filename, T *&p, int64_t &n, int64_t &dim) {
  std::cout << filename <<std::endl;
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  int dim_32;
  in.read((char *)&dim_32, 4); // 读4字节, dim
  dim = dim_32;
  // 移动到末尾
  in.seekg(0, std::ios::end);
  // 计算有多少 总字节除以每个vector的大小
  n = in.tellg() / (4 + dim * sizeof(T));
  in.seekg(0, std::ios::beg);
  std::cout << "Read path: " << filename << ", nx: " << n << ", dim: " << dim
            << std::endl;

  // 分配对齐内存,为后续使用simd指令准备
  p = reinterpret_cast<T *>(aligned_alloc(64, n * dim * sizeof(T)));

  // read vector line by line
  for (int i = 0; i < n; ++i) {
    in.seekg(4, std::ios::cur); // 移动4字节,是因为每行有4字节记录维度
    in.read((char *)&p[i * dim], dim * sizeof(T)); // 读数据,每次读一个vector
  }
  in.close();
}
int main(int argc, char **argv) {
  
  std::string base_path;
  std::string query_path;
  std::string gt_path;
  std::string graph_path;
  int level;
  int topk;
  int search_ef;
  int num_threads;
  po::options_description desc{"Arguments"};
  try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("base_path", po::value<std::string>(&base_path)->required(), "base vector file path");
        desc.add_options()("query_path", po::value<std::string>(&query_path)->required(), "query vector file path");
        desc.add_options()("gt_path", po::value<std::string>(&gt_path)->required(),"ground_truth file path");
        desc.add_options()("graph_path", po::value<std::string>(&graph_path)->required(),"index path");
        desc.add_options()("level", po::value<int>(&level)->required(),"optimization level");
        desc.add_options()("topk", po::value<int>(&topk)->required(),"K-ANN");
        desc.add_options()("ef", po::value<int>(&search_ef)->default_value(32),"search ef for HNSW (cannot be set lower than topk)");
        desc.add_options()("num_thread", po::value<int>(&num_threads)->default_value(1),"search thread num");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        std::cout << "error" << std::endl;
        return -1;
    }
  float *base = nullptr;
  float *query = nullptr;
  int *gt = nullptr;
  int64_t N, dim, nq, gt_k;
  // load base vector
  load_fvecs(base_path.c_str(), base, N, dim);
  // load query vector
  load_fvecs(query_path.c_str(), query, nq, dim);
  // load ground_truth
  load_fvecs(gt_path.c_str(), gt, nq, gt_k);
  if (!std::filesystem::exists(graph_path)) {
    glass::HNSW hnsw(dim, "L2", 32, 200); // L->候选(构建ef) R->连接数
    hnsw.Build(base, N);
    hnsw.final_graph.save(graph_path);
  }
  glass::Graph<int> graph;
  graph.load(graph_path);
  auto searcher = glass::create_searcher(graph, "L2", level);
  searcher->SetData(base, N, dim);
  searcher->Optimize(num_threads);
  // search_ef the size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k.
  // The value ef of can be anything between k and the size of the dataset.
  searcher->SetEf(search_ef);
  double recall;
  double best_qps = 0.0;
  int iters = 1;
  for (int iter = 1; iter <= iters; ++iter) {
    printf("iter : [%d/%d]\n", iter, iters);
    std::vector<int> pred(nq * topk);
    auto st = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < nq; ++i) {
      searcher->Search(query + i * dim, topk, pred.data() + i * topk);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    double qps = nq / ela;
    best_qps = std::max(qps, best_qps);
    int cnt = 0;
    for (int i = 0; i < nq; ++i) {
      std::unordered_set<int> st(gt + i * gt_k, gt + i * gt_k + topk);
      for (int j = 0; j < topk; ++j) {
        if (st.count(pred[i * topk + j])) {
          cnt++;
        }
      }
    }
    recall = (double)cnt / nq / topk;
    printf("\tRecall@%d = %.4lf, QPS = %.2lf\n", topk, recall, qps);
  }
  printf("Best QPS = %.2lf\n", best_qps);
  free(base);
  free(query);
  free(gt);
}