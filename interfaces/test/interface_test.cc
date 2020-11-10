#include "interfaces/conex.h"
#include "conex/debug_macros.h"

#include "gtest/gtest.h"

TEST(TestArguments, AddLMI) {
  void* p = ConexCreateConeProgram();
  int constraint_id = 0;
  EXPECT_TRUE(CONEX_AddLinearMatrixInequality(p, 2, 2, &constraint_id) == CONEX_SUCCESS);
  EXPECT_EQ(0, constraint_id);
  EXPECT_TRUE(CONEX_AddLinearMatrixInequality(p, 2, 4, &constraint_id) == CONEX_SUCCESS);
  EXPECT_EQ(1, constraint_id);

  void* null_ptr = NULL;
  EXPECT_TRUE(CONEX_AddLinearMatrixInequality(null_ptr, 2, 2, &constraint_id) == CONEX_FAILURE);

  void* corrupted_ptr = static_cast<double*>(p) + 1;
  EXPECT_TRUE(CONEX_AddLinearMatrixInequality(corrupted_ptr, 2, 2,&constraint_id) == CONEX_FAILURE);

  int bad_complex_dim = 3;
  EXPECT_TRUE(CONEX_AddLinearMatrixInequality(p, 2, bad_complex_dim, &constraint_id) == CONEX_FAILURE);

  int bad_order = 0;
  EXPECT_TRUE(CONEX_AddLinearMatrixInequality(p, bad_order, 2, &constraint_id) == CONEX_FAILURE);
  ConexDeleteConeProgram(p);
}
