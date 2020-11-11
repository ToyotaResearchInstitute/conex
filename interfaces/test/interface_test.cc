#include "interfaces/conex.h"
#include "conex/debug_macros.h"

#include "gtest/gtest.h"

TEST(TestArguments, AddLMI) {
  void* p = ConexCreateConeProgram();
  int constraint_id = 0;
  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(p, 2, 2, &constraint_id) == CONEX_SUCCESS);
  EXPECT_EQ(0, constraint_id);
  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(p, 2, 4, &constraint_id) == CONEX_SUCCESS);
  EXPECT_EQ(1, constraint_id);

  void* null_ptr = NULL;
  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(null_ptr, 2, 2, &constraint_id) == CONEX_FAILURE);

  void* corrupted_ptr = static_cast<double*>(p) + 1;
  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(corrupted_ptr, 2, 2,&constraint_id) == CONEX_FAILURE);

  int bad_complex_dim = 3;
  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(p, 2, bad_complex_dim, &constraint_id) == CONEX_FAILURE);

  int bad_order = 0;
  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(p, bad_order, 2, &constraint_id) == CONEX_FAILURE);
  ConexDeleteConeProgram(p);
}

TEST(TestArguments, UpdateLMI) {
  void* p = ConexCreateConeProgram();

  int constraint_id = 0;
  int order = 2;
  int hyper_complex_dim = 2;

  EXPECT_TRUE(CONEX_NewLinearMatrixInequality(p, order, hyper_complex_dim, 
                                              &constraint_id) == CONEX_SUCCESS);
  
  EXPECT_TRUE(CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, order - 2, hyper_complex_dim -1));

  int bad_hyper_complex_dim = hyper_complex_dim;
  int status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, order - 2, 
                                       bad_hyper_complex_dim);
  EXPECT_EQ(CONEX_FAILURE, status); 

  int bad_order = order;
  EXPECT_EQ(CONEX_FAILURE, 
            CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, bad_order, order - 2, 
                                       hyper_complex_dim - 1));

  int bad_index = order;
  EXPECT_EQ(CONEX_FAILURE, CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, bad_index, hyper_complex_dim - 1));

  ConexDeleteConeProgram(p);
}
