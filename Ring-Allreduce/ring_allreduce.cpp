#include "../common.h"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <vector>

std::vector<int> gen_rank_data(const int &mpi_rank, const int &data_size) {
  std::srand(mpi_rank);
  std::vector<int> cur_rank_data(data_size);
  for (int i = 0; i < data_size; i++) {
    cur_rank_data[i] = std::rand() % (mpi_rank + 1);
  }
  return cur_rank_data;
}

void reset_rank_data(std::vector<int> &cur_rank_data,
                     std::vector<int> &all_rank_data, const int &mpi_rank,
                     const int &data_size) {
  cur_rank_data = gen_rank_data(mpi_rank, data_size);
  for (int i = 0; i < data_size; i++) {
    all_rank_data[i] = 0;
  }
}

void Ring_Allreduce(void *sendbuf, void *recvbuf, int n, const MPI_Comm &comm,
                    int comm_sz, int mpi_rank) {
  memcpy(recvbuf, sendbuf, n * sizeof(int));
  int block_size = n / comm_sz;
  int recv_data[block_size];
  MPI_Request request[2];

  for (int i = 0; i < comm_sz - 1; i++) {
    /* rank 0
     *            recv
     *        <------------
     *       0      1      2
     *         send
     *         --->
     *
     * rank 1
     *       0      1      2
     *         recv  send
     *         --->  --->
     *
     * rank 2
     *            send
     *        <------------
     *       0      1      2
     *                recv
     *                --->
     **/
    int offset_send = (mpi_rank - i + comm_sz) % comm_sz * block_size;
    int offset_recv = (mpi_rank - i - 1 + comm_sz) % comm_sz * block_size;

    MPI_Isend(static_cast<int *>(recvbuf) + offset_send, block_size, MPI_INT,
              (mpi_rank + 1 + comm_sz) % comm_sz, i, comm, &request[0]);
    MPI_Irecv(recv_data, block_size, MPI_INT,
              (mpi_rank - 1 + comm_sz) % comm_sz, i, comm, &request[1]);
    MPI_Wait(&request[1], nullptr);
    for (int j = 0; j < block_size; j++) {
      static_cast<int *>(recvbuf)[offset_recv + j] += recv_data[j];
    }
    MPI_Wait(&request[0], nullptr);
  }

  for (int i = 0; i < comm_sz - 1; i++) {
    int offset_send = (mpi_rank - i + 1 + comm_sz) % comm_sz * block_size;
    int offset_recv = (mpi_rank - i + comm_sz) % comm_sz * block_size;

    MPI_Isend(static_cast<int *>(recvbuf) + offset_send, block_size, MPI_INT,
              (mpi_rank + 1 + comm_sz) % comm_sz, i, comm, &request[0]);
    MPI_Irecv(static_cast<int *>(recvbuf) + offset_recv, block_size, MPI_INT,
              (mpi_rank - 1 + comm_sz) % comm_sz, i, comm, &request[1]);
    MPI_Wait(&request[1], nullptr);
    MPI_Wait(&request[0], nullptr);
  }
}

void Native_Allreduce(void *sendbuf, void *recvbuf, int n, MPI_Comm comm,
                      int comm_sz, int mpi_rank) {
  MPI_Reduce(sendbuf, recvbuf, n, MPI_INT, MPI_SUM, 0, comm);
  MPI_Bcast(recvbuf, n, MPI_INT, 0, comm);
}

int main(int argc, char *argv[]) {
  int epoch = std::atoi(argv[1]);
  int data_size = std::atoi(argv[2]);
  int mpi_comm_sz;
  int &mpi_rank = rank();

  mpi_init(mpi_comm_sz, mpi_rank);

  std::vector<int> cur_rank_data = gen_rank_data(mpi_rank, data_size);
  std::vector<int> all_rank_data(data_size, 0);
  if (mpi_rank == 0) {
    std::cout << "Ring_Allreduce: " << std::endl;
  }
  TEST_FUNC_BENCHMARK(epoch, Ring_Allreduce, &cur_rank_data[0],
                      &all_rank_data[0], data_size, MPI_COMM_WORLD, mpi_comm_sz,
                      mpi_rank);
  if (mpi_rank == 0) {
    print_sum(all_rank_data);
  }

  if (mpi_rank == 0) {
    std::cout << "Native_Allreduce: " << std::endl;
  }
  reset_rank_data(cur_rank_data, all_rank_data, mpi_rank, data_size);
  TEST_FUNC_BENCHMARK(epoch, Native_Allreduce, &cur_rank_data[0],
                      &all_rank_data[0], data_size, MPI_COMM_WORLD, mpi_comm_sz,
                      mpi_rank);
  if (mpi_rank == 0) {
    print_sum(all_rank_data);
  }

  if (mpi_rank == 0) {
    std::cout << "MPI_Allreduce: " << std::endl;
  }
  reset_rank_data(cur_rank_data, all_rank_data, mpi_rank, data_size);
  TEST_FUNC_BENCHMARK(epoch, MPI_Allreduce, &cur_rank_data[0],
                      &all_rank_data[0], data_size, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    print_sum(all_rank_data);
  }

  mpi_finalize();

  // Print(cur_rank_data);
  return 0;
}
