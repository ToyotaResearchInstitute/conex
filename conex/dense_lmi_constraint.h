#include "psd_constraint.h"
#include "newton_step.h"

//using InputRefType = Eigen::Map<const DenseMatrix>;
using InputRefType = DenseMatrix;
// using InputRefType = Eigen::Ref<const DenseMatrix>;
template<typename T>
std::vector<InputRefType> ConvertToRef(const std::vector<T>& v) {
  std::vector<InputRefType> t;
  for (const auto& vi : v) {
    // t.push_back(  InputRefType(vi.data(), vi.rows(), vi.cols()));
    t.push_back(InputRefType(vi));
  }
  return t;
}

struct DenseLMIConstraint final : public PsdConstraint {
  template<typename T>
  DenseLMIConstraint(int n, std::vector<T>* constraint_matrices,
                      T* constraint_affine) :
      PsdConstraint(n, static_cast<int>(constraint_matrices->size())),
      constraint_matrices_(ConvertToRef(*constraint_matrices)),
      //constraint_affine_(constraint_affine->data(),
      //                      constraint_affine->rows(),
      //                      constraint_affine->cols()) {
      constraint_affine_(*constraint_affine) {}

  /*
  DenseLMIConstraint(int n, const std::vector<StorageType>& constraint_matrices,
                     const StorageType& constraint_affine) : 
      PsdConstraint(n, static_cast<int>(constraint_matrices.size())),
                      constraint_matrices_data{constraint_matrices},
                      constraint_affine_data{constraint_affine},
                      constraint_matrices_{&constraint_matrices_data},
                      constraint_affine_{&constraint_affine_data} {}
  */

  const std::vector<DenseMatrix> constraint_matrices_;
  const DenseMatrix constraint_affine_;

  friend void ConstructSchurComplementSystem<DenseLMIConstraint>(
        DenseLMIConstraint* o, bool initialize, SchurComplementSystem* sys);

 private:
  void ComputeNegativeSlack(double k, const Ref& y, Ref* s);
  void ComputeAW(int i, const Ref& W, Ref* AW, Ref* WAW);
  double EvalDualConstraint(int j, const Ref& W);
  double EvalDualObjective(const Ref& W);
  void MultByA(const Ref& x, Ref* Y);
};
