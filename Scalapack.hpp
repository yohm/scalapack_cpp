//
// Created by Yohsuke Murase on 2021/10/17.
//
#include <iostream>
#include <vector>
#include <array>
#include <cassert>
#include "mpi.h"

#ifndef CPP_SCALAPACK_HPP
#define CPP_SCALAPACK_HPP



extern "C" {
  void sl_init_(int *icontext, int *nprow, int *npcolumn);
  // SL_INIT initializes an NPROW x NPCOL process grid using a row-major ordering
  // of the processes. This routine retrieves a default system context which will
  // include all available processes. (out) ictxt, (in) nprow, npcolumn

  void blacs_gridinfo_(int *icontext, int *nprow, int *npcolumn, int *myrow,
                       int *mycolumn);
  // (in) icontext: BLACS context
  // (out) nprow, npcolumn: the numbers of rows and columns in this process grid
  // (out) myrow, mycolumn: the process grid row- and column-index

  void blacs_exit_(int *cont);
  // (in) continue: if 0, all the resources are released. If nonzero, MPI
  // resources are not released.

  void blacs_gridexit_(int *icontext);
  // (in) icontext: BLACS context

  void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc,
                 int *icsrc, int *icontext, int *lld, int *info);
  // (out) descriptor for the global matrix. `desc` must be an array of int of
  //   length 9. int[9]
  // (in) m, n: rows and columns of the matrix (in) mb, nb: row,
  //   column block sizes
  // (in) irsrc, icsrc: the process row (column) over which the
  // first row of the global matrix is distributed.
  // (in) icontext: BLACS context
  // (in) lld: leading dimension of the local array
  // (out) info: 0 => completed successfully

  void dgesd2d_(int *icontext, int *m, int *n, double *A, int *lda, int *r_dest,
                int *c_dest);
  // Takes a general rectangular matrix and sends it to the destination process.
  // (in) icontext: BLACS context
  // (in) m,n: matrix sizes
  // (in) A: matrix
  // (in) lda: leading dimension (m)
  // (in) r_dest, c_dest: the process corrdinate of the process to send the
  // message to

  void dgerv2d_(int *icontext, int *m, int *n, double *A, int *lda, int *row_src,
                int *col_src);
  // Receives a message from the process into the general rectangular matrix.
  // (in) icontext: BLACS context
  // (in) m,n,lda: sizes of the matrix
  // (out) A: matrix
  // (in) row_src, col_src: the process coordinate of the source of the message

  void pdgesv_(int *n, int *nrhs, double *A, int *ia, int *ja, int desc_a[9],
               int *ipvt, double *B, int *ib, int *jb, int desc_b[9], int *info);
  // These subroutines solve the following systems of equations for multiple
  // right-hand sides: AX = B
  // (in) n: order of the submatrix = the number of rows of B
  // (in) nrhs: the number of columns of B
  // (in/out) A: the local part of the global general matrix A.
  // (in) ia, ja: the row and the column indices of the
  //   global matrix A, identifying the first row and column of the submatrix A.
  // (in) desc_a: descriptor of A matrix
  // (out) ipvt: the local part of the global vector ipvt, containing the pivot
  // indices.
  // (in/out) B: the local part of the global general matrix B,
  //   containing the right-hand sides of the system.
  // (in) ib, jb: the row and the column indices of the global matrix B,
  //   identifying the first row and column of the submatrix B.
  // (in) desc_b: descriptor of B matrix (out) info: error code

  void pdgemm_( char *TRANSA, char *TRANSB,
                int *M, int *N, int *K,
                double *ALPHA,
                double *A, int *IA, int *JA, int DESCA[9],
                double *B, int *IB, int *JB, int DESCB[9],
                double *BETA,
                double *C, int *IC, int *JC, int DESCC[9] );
  // calculate alpha*AB + beta*C
  // A: M x K, B: K x N, C: M x N
  // (in) TRANSA, TRANSB: 'n' or 't'  (normal or transpose)
  // (in) M, N, K: sizes of the matrix
  // (in) ALPHA, BETA: coefficient
  // (in) A,B,C: input matrix
  // (out) C: output matrix
}


class Scalapack {
  public:
  static int ICTXT, NPROW, NPCOL, MYROW, MYCOL;
  static void Initialize(const std::array<int,2>& proc_grid_size) {
    NPROW = proc_grid_size[0];
    NPCOL = proc_grid_size[1];
    int num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    if (num_proc != NPROW * NPCOL) {
      std::cerr << "Error: invalid number of procs" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    sl_init_(&ICTXT, &NPROW, &NPCOL);
    blacs_gridinfo_(&ICTXT, &NPROW, &NPCOL, &MYROW, &MYCOL);
  }
  static void Finalize() {
    blacs_gridexit_(&ICTXT);
    int blacs_exitcode = 0;
    blacs_exit_(&blacs_exitcode);
  }

  // global matrix
  class GMatrix {
    public:
    GMatrix(size_t N, size_t M) : N(N), M(M) {
      A.resize(N * M, 0.0);
    }
    size_t N, M;
    std::vector<double> A;
    double At(size_t I, size_t J) const { return A.at(I*M+J); }
    void Set(size_t I, size_t J, double val) { A[I*M+J] = val; }
    double* Data() { return A.data(); }
    size_t Size() { return A.size(); }
    friend std::ostream& operator<<(std::ostream& os, const GMatrix& gm) {
      for (size_t i = 0; i < gm.N; i++) {
        for (size_t j = 0; j < gm.M; j++) {
          os << gm.At(i, j) << ' ';
        }
        os << "\n";
      }
      return os;
    }
    void BcastFrom(int root_rank) {
      int my_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
      std::array<uint64_t,2> sizes = {N, M};
      MPI_Bcast(sizes.data(), 2, MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
      if (my_rank != root_rank) { A.resize(N*M); }
      MPI_Bcast(A.data(), N*M, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
    }
  };

  // local matrix for scalapack
  class LMatrix {
    public:
    LMatrix(int N, int M, int NB, int MB) : N(N), M(M), NB(NB), MB(MB) {  // matrix N x M with block NB x MB
      SUB_ROWS = (N / (NB * NPROW)) * NB + std::min(N % (NB * NPROW), NB);
      SUB_COLS = (M / (MB * NPCOL)) * MB + std::min(M % (MB * NPCOL), MB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit_(DESC, &N, &M, &NB, &MB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
    }
    LMatrix(const GMatrix& gm, int NB, int MB) : N(gm.N), M(gm.M), NB(NB), MB(MB) {
      SUB_ROWS = (N / (NB * NPROW)) * NB + std::min(N % (NB * NPROW), NB);
      SUB_COLS = (M / (MB * NPCOL)) * MB + std::min(M % (MB * NPCOL), MB);
      int RSRC = 0, CSRC = 0, INFO;
      descinit_(DESC, &N, &M, &NB, &MB, &RSRC, &CSRC, &ICTXT, &SUB_ROWS, &INFO);
      assert(INFO == 0);
      SUB.resize(SUB_ROWS * SUB_COLS, 0.0);
      for (int i = 0; i < SUB_ROWS; i++) {
        for (int j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < N && J < M) {
            Set(i, j, gm.At(I, J));
          }
        }
      }
    };
    int N, M;  // size of the global matrix
    int NB, MB; // block sizes
    int SUB_ROWS, SUB_COLS;  // size of the local matrix
    int DESC[9];
    std::vector<double> SUB;

    // convert submatrix index (i,j) at process (p_row, p_col) into global coordinate (I,J)
    std::array<size_t,2> ToGlobalCoordinate(size_t i, size_t j, int p_row = MYROW, int p_col = MYCOL) const {
      // block coordinate (bi, bj)
      size_t bi = i / NB;
      size_t bj = j / MB;
      // local coordinate inside the block
      size_t ii = i % NB;
      size_t jj = j % MB;
      // calculate global coordinate
      size_t I = bi * (NB * NPROW) + p_row * NB + ii;
      size_t J = bj * (MB * NPCOL) + p_col * MB + jj;
      return {I, J};
    }
    // convert global matrix index (I,J) to local coordinate (i,j),(p_row,p_col)
    std::pair<std::array<size_t,2>, std::array<int,2>> ToLocalCoordinate(size_t I, size_t J) const {
      // global block coordinate (BI, BJ)
      size_t BI = I / NB;
      size_t BJ = J / MB;
      // process coordinate (bi, bj)
      int p_row = BI % NPROW;
      int p_col = BJ % NPCOL;
      // local block coordinate (bi, bj)
      size_t bi = BI / NPROW;
      size_t bj = BJ / NPCOL;
      // local coordinate inside the block
      size_t ii = I % NB;
      size_t jj = J % MB;
      // calculate global coordinate
      size_t i = bi * NB + ii;
      size_t j = bj * MB + jj;
      return {{i, j}, {p_row, p_col}};
    }
    double At(size_t i, size_t j) const {  // get an element at SUB[ (i,j) ]
      return SUB[i + j * SUB_ROWS];
    }
    void Set(size_t i, size_t j, double val) {
      SUB[i + j * SUB_ROWS] = val;
    }
    void SetByGlobalCoordinate(size_t I, size_t J, double val) {
      auto local_pos = ToLocalCoordinate(I, J);
      auto ij = local_pos.first;
      auto proc_grid = local_pos.second;
      if (proc_grid[0] == MYROW && proc_grid[1] == MYCOL) {
        Set(ij[0], ij[1], val);
      }
    }
    void SetAll(double val) {
      for (size_t i = 0; i < SUB_ROWS; i++) {
        for (size_t j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          if (IJ[0] < N && IJ[1] < M) Set(i, j, val);
        }
      }
    }
    double* Data() { return SUB.data(); }
    friend std::ostream& operator<<(std::ostream& os, const LMatrix& lm) {
      for (size_t i = 0; i < lm.SUB_ROWS; i++) {
        for (size_t j = 0; j < lm.SUB_COLS; j++) {
          os << lm.At(i, j) << ' ';
        }
        os << "\n";
      }
      return os;
    }

    GMatrix ConstructGlobalMatrix() const {
      GMatrix A(N, M);
      for (size_t i = 0; i < SUB_ROWS; i++) {
        for (size_t j = 0; j < SUB_COLS; j++) {
          auto IJ = ToGlobalCoordinate(i, j);
          size_t I = IJ[0], J = IJ[1];
          if (I < N && J < M) {
            A.Set(I, J, At(i, j));
          }
        }
      }
      GMatrix AA(N, M);
      MPI_Allreduce(A.Data(), AA.Data(), N*M, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return AA;
    }
    void DebugPrintAtRoot(std::ostream& out) const {
      MPI_Barrier(MPI_COMM_WORLD);
      GMatrix g = ConstructGlobalMatrix();
      if (Scalapack::MYROW == 0 && Scalapack::MYCOL == 0) {
        out << g;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    static LMatrix Identity(int N, int M, int NB, int MB) {
      LMatrix lm(N, M, NB, MB);
      for (size_t i = 0; i < lm.SUB_ROWS; i++) {
        for (size_t j = 0; j < lm.SUB_COLS; j++) {
          auto IJ = lm.ToGlobalCoordinate(i, j);
          if (IJ[0] < N && IJ[1] < M && IJ[0] == IJ[1]) lm.Set(i, j, 1.0);
        }
      }
      return lm;
    }
  };

  // call PDGESV routine of Scalapack
  // The solution is stored in B
  static void CallPDGESV(LMatrix& A, LMatrix& B) {
    assert(A.N == A.M);
    assert(A.N == B.N);
    int IA = 1;
    int JA = 1;
    int IB = 1;
    int JB = 1;
    int INFO = 0;
    std::vector<int> IPIV(A.N + A.NB, 0);
    // std::cerr << A << B;
    pdgesv_(&A.N, &B.M, A.Data(), &IA, &JA, A.DESC, IPIV.data(),
            B.Data(), &IB, &JB, B.DESC, &INFO);
    if (INFO != 0) {
      std::cerr << "Error: INFO of PDGESV is not zero but " << INFO << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    assert(INFO == 0);
  }

  // computes AB+C
  // the results are stored in C
  static void CallPDGEMM(double alpha, LMatrix& A, LMatrix& B, double beta, LMatrix& C) {
    assert(A.N == C.N);
    assert(A.M == B.N);
    assert(B.M == C.M);
    int IA = 1, JA = 1, IB = 1, JB = 1, IC = 1, JC = 1;
    char trans = 'n';
    pdgemm_(&trans, &trans, &C.N, &C.M, &A.M,
            &alpha,
            A.Data(), &IA, &JA, A.DESC,
            B.Data(), &IB, &JB, B.DESC,
            &beta,
            C.Data(), &IC, &JC, C.DESC);
  }

};

#endif //CPP_SCALAPACK_HPP
