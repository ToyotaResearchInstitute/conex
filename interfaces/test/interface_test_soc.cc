#include "conex/debug_macros.h"
#include "interfaces/conex.h"

#include "gtest/gtest.h"

TEST(TestArguments, AddLMI) {
  void* p = CONEX_CreateConeProgram();
  int constraint_id = 0;
  EXPECT_TRUE(CONEX_NewLorentzConeConstraint(p, 2, &constraint_id) ==
              CONEX_SUCCESS);
  EXPECT_EQ(0, constraint_id);
  EXPECT_TRUE(CONEX_NewLorentzConeConstraint(p, 2, &constraint_id) ==
              CONEX_SUCCESS);
  EXPECT_EQ(1, constraint_id);

  void* null_ptr = NULL;
  EXPECT_TRUE(CONEX_NewLorentzConeConstraint(null_ptr, 2, &constraint_id) ==
              CONEX_FAILURE);

  int bad_order = 0;
  EXPECT_TRUE(CONEX_NewLorentzConeConstraint(p, bad_order, &constraint_id) ==
              CONEX_FAILURE);
  CONEX_DeleteConeProgram(p);
}

TEST(TestArguments, UpdateLMI) {
  void* p = CONEX_CreateConeProgram();

  int status;
  int constraint_id = 0;
  int order = 2;

  status = CONEX_NewLorentzConeConstraint(p, order, &constraint_id);
  EXPECT_EQ(CONEX_SUCCESS, status);

  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, 0, 0);
  EXPECT_EQ(CONEX_SUCCESS, status);

  int bad_hyper_complex_dim = 1;
  int bad_variable = -1;
  int bad_column_index = 1;

  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1, 0,
                                      bad_hyper_complex_dim);
  EXPECT_EQ(CONEX_FAILURE, status);
  status =
      CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, bad_variable, 0, 0);
  EXPECT_EQ(CONEX_FAILURE, status);
  status = CONEX_UpdateLinearOperator(p, constraint_id, .3, 2, order - 1,
                                      bad_column_index, 0);
  EXPECT_EQ(CONEX_FAILURE, status);

  // Repeat for affine term.
  status = CONEX_UpdateAffineTerm(p, constraint_id, .3, 0, 0, 0);
  EXPECT_EQ(CONEX_SUCCESS, status);
  status = CONEX_UpdateAffineTerm(p, constraint_id, .3, 2, 0, 0);
  EXPECT_EQ(CONEX_SUCCESS, status);

  status =
      CONEX_UpdateAffineTerm(p, constraint_id, .3, 2, 0, bad_hyper_complex_dim);
  EXPECT_EQ(CONEX_FAILURE, status);
  status = CONEX_UpdateAffineTerm(p, constraint_id, .3, 2, bad_column_index, 0);
  EXPECT_EQ(CONEX_FAILURE, status);

  CONEX_DeleteConeProgram(p);
}
