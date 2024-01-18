#ifndef HPC_DEMO_COMMON_H_
#define HPC_DEMO_COMMON_H_

#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <iostream>
#include <vector>
#include <mpi.h>

template <typename F, typename... Args> struct invoke_result {
  using type = decltype(std::declval<F>()(std::declval<Args>()...));
};

template <typename F, typename... Args>
using invoke_result_t = typename invoke_result<F, Args...>::type;

static void mpi_init(int &mpi_comm_sz, int &mpi_rank) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
}

static void mpi_finalize() { MPI_Finalize(); }

inline static int &rank() {
  static int rank;
  return rank;
}

template <typename F, typename... Args>
static void TEST_FUNC_BENCHMARK(const int &epoch, F &&f, Args &&...args) {
  auto func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
  auto begin_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < epoch; i++) {
    func();
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        end_time - begin_time)
                        .count() *
                    1000;

  if (rank() == 0) {
    std::cout << "cost time: " << duration << " ms" << std::endl;
  }
}

static void print_sum(std::vector<int> &data) {
  int sum = 0;
  for (auto &v : data) {
    sum += v;
  }
  std::cout << sum << std::endl;
}
#endif /* ifndef HPC_DEMO_COMMON_H_ */
