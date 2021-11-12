#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include "mpi.h"
#include "Scalapack.hpp"


#define myassert(x) do {                              \
if (!(x)) {                                           \
  printf("Assertion failed: %s, file %s, line %d\n"   \
         , #x, __FILE__, __LINE__);                   \
  MPI_Abort(MPI_COMM_WORLD, 1);                       \
  }                                                   \
} while (0)

int my_rank;

void test_Matrix() {
  const size_t N = 3, M = 5;
  Scalapack::GMatrix g(3, 5);

  if (my_rank == 0) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        g.Set(i, j, i*5+j);
      }
    }
    std::cerr << g;
  }
  g.BcastFrom(0);
  Scalapack::LMatrix lm(g, 2, 2);

  myassert(lm.SUB_ROWS == 2);
  myassert(lm.SUB_COLS == 3);

  for (int pi = 0; pi < Scalapack::NPROW; pi++) {
    for (int pj = 0; pj < Scalapack::NPCOL; pj++) {
      if (Scalapack::MYROW == pi && Scalapack::MYCOL == pj) {
        std::cerr << "(pi, pj): " << pi << ", " << pj << std::endl;
        std::cerr << lm;
        for (int i = 0; i < lm.SUB_ROWS; i++) {
          for (int j = 0; j < lm.SUB_COLS; j++) {
            auto IJ = lm.ToGlobalCoordinate(i, j);
            if (IJ[0] < lm.N && IJ[1] < lm.M) {
              auto local_pos = lm.ToLocalCoordinate(IJ[0], IJ[1]);
              myassert( local_pos.first[0] == i );
              myassert( local_pos.first[1] == j );
              myassert( local_pos.second[0] == pi );
              myassert( local_pos.second[1] == pj );
            }
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  lm.SetByGlobalCoordinate(1, 3, -100.0);
  std::ostringstream ss;
  lm.DebugPrintAtRoot(ss);
  if (my_rank == 0) {
    myassert( ss.str() == std::string("0 1 2 3 4 \n5 6 7 -100 9 \n10 11 12 13 14 \n") );
  }

  // test SetAll
  {
    Scalapack::LMatrix m1(3, 1, 2, 1);
    m1.SetAll(1.0);
    Scalapack::GMatrix g1 = m1.ConstructGlobalMatrix();
    for (size_t i = 0; i < 3; i++) {
      myassert( g1.At(i, 0) == 1.0 );
    }
  }

  // test Identity
  {
    auto g = Scalapack::LMatrix::Identity(3, 3, 2, 2).ConstructGlobalMatrix();
    if (my_rank == 0) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if (i == j) { myassert(g.At(i, j) == 1.0); }
          else { myassert(g.At(i, j) == 0.0); }
        }
      }
    }
  }
}

void test_PDGESV() {

  Scalapack::GMatrix A(4, 4);
  A.Set(0, 0, 3.0); A.Set(0, 1, 1.0); A.Set(0, 2, 1.0); A.Set(0, 3, 2.0);
  A.Set(1, 0, 5.0); A.Set(1, 1, 1.0); A.Set(1, 2, 3.0); A.Set(1, 3, 4.0);
  A.Set(2, 0, 2.0); A.Set(2, 1, 0.0); A.Set(2, 2, 1.0); A.Set(2, 3, 0.0);
  A.Set(3, 0, 1.0); A.Set(3, 1, 3.0); A.Set(3, 2, 2.0); A.Set(3, 3, 1.0);
  Scalapack::LMatrix lA(A, 2, 2);

  Scalapack::GMatrix B(4, 1);
  B.Set(0, 0, 2.0);
  B.Set(1, 0, 6.0);
  B.Set(2, 0, 1.0);
  B.Set(3, 0, 3.0);
  Scalapack::LMatrix lB(B, 2, 1);

  // if (my_rank == 0) {
  //   std::cerr << "A: \n" << A << std::endl;
  //   std::cerr << "B: \n" << B << std::endl;
  //   std::cerr << "lA: \n" << lA << std::endl;
  //   std::cerr << "lB: \n" << lB << std::endl;
  // }

  Scalapack::CallPDGESV(lA, lB);

  Scalapack::GMatrix X = lB.ConstructGlobalMatrix();
  if (my_rank == 0) {
    std::cerr << "test_PDGESV -------------" << std::endl;
    std::cerr << "A: \n" << A << std::endl;
    std::cerr << "B: \n" << B << std::endl;
    std::cerr << "X: \n" << X << std::endl;

    for (int i = 0; i < 4; i++) {
      double ans = 0.0;
      for (int j = 0; j < 4; j++) {
        ans += A.At(i, j) * X.At(j, 0);
      }
      myassert( std::abs(ans - B.At(i, 0) ) < 1.0e-6 );
    }
  }
}

void test_PDGEMM() {

  const size_t N = 4, M = 4;
  Scalapack::GMatrix A(4, 4);
  A.Set(0, 0, 3.0); A.Set(0, 1, 1.0); A.Set(0, 2, 1.0); A.Set(0, 3, 2.0);
  A.Set(1, 0, 5.0); A.Set(1, 1, 1.0); A.Set(1, 2, 3.0); A.Set(1, 3, 4.0);
  A.Set(2, 0, 2.0); A.Set(2, 1, 0.0); A.Set(2, 2, 1.0); A.Set(2, 3, 0.0);
  A.Set(3, 0, 1.0); A.Set(3, 1, 3.0); A.Set(3, 2, 2.0); A.Set(3, 3, 1.0);
  Scalapack::LMatrix lA(A, 2, 2);

  Scalapack::GMatrix B(4, 1);
  B.Set(0, 0, -0.2727);
  B.Set(1, 0, -0.1818);
  B.Set(2, 0, 1.54545);
  B.Set(3, 0, 0.727273);
  Scalapack::LMatrix lB(B, 2, 1);

  Scalapack::LMatrix lC(4, 1, 2, 1);

  Scalapack::CallPDGEMM(1.0, lA, lB, 0.0, lC);

  Scalapack::GMatrix C = lC.ConstructGlobalMatrix();
  if (my_rank == 0) {
    std::cerr << "test_PDGEMM -------------" << std::endl;
    std::cerr << C;
    auto close = [](double x, double y)->bool { return std::abs(x-y) < 0.001; };
    myassert( close(C.At(0,0), 2.0) );
    myassert( close(C.At(1,0), 6.0) );
    myassert( close(C.At(2,0), 1.0) );
    myassert( close(C.At(3,0), 3.0) );
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  Scalapack::Initialize({2, 2});

  test_Matrix();

  test_PDGESV();

  test_PDGEMM();

  MPI_Finalize();
  return 0;
}