#include "interfaces/conex.h"
#include "conex/debug_macros.h"

#include "gtest/gtest.h"

TEST(TestArguments, AddLMI) {
  void* p = CONEX_CreateConeProgram();
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
  CONEX_DeleteConeProgram(p);
}

TEST(TestArguments, UpdateLMI) {
  void* p = CONEX_CreateConeProgram();

  int status;
  int constraint_id = 0;
  int constraint_id_2 = 0;
  int order = 2;
  int hyper_complex_dim = 2;

  status = CONEX_NewLinearMatrixInequality(p, order, hyper_complex_dim, 
                                              &constraint_id);
  EXPECT_EQ(CONEX_SUCCESS, status);

  status = CONEX_NewLinearMatrixInequality(p, order, hyper_complex_dim, 
                                              &constraint_id_2);
  EXPECT_EQ(CONEX_SUCCESS, status);

  DUMP(constraint_id_2);

  DUMP(constraint_id);
  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, order - 2, hyper_complex_dim -1);
  EXPECT_EQ(CONEX_SUCCESS, status);

  int bad_hyper_complex_dim = hyper_complex_dim;
  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, order - 2, bad_hyper_complex_dim);
  EXPECT_EQ(CONEX_FAILURE, status); 

  int bad_order = order;
  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, bad_order, order - 2, hyper_complex_dim - 1);
  EXPECT_EQ(CONEX_FAILURE, status); 

  int bad_index = order;
  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, bad_index, hyper_complex_dim - 1);
  EXPECT_EQ(CONEX_FAILURE, status); 

  status = CONEX_UpdateAffineTerm(p, constraint_id, .3,  order - 1, order - 2, bad_hyper_complex_dim);
  EXPECT_EQ(CONEX_FAILURE, status); 

  status = CONEX_UpdateAffineTerm(p, constraint_id, .3,  order - 1, order - 2, hyper_complex_dim - 1);
  EXPECT_EQ(CONEX_SUCCESS, status); 

  status = CONEX_UpdateAffineTerm(p, constraint_id, .3, 0, 0, hyper_complex_dim - 1);
  EXPECT_EQ(CONEX_FAILURE, status); 
  CONEX_DeleteConeProgram(p);
}
