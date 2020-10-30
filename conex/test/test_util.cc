#include "test_util.h"

DenseMatrix Symmetrize(const DenseMatrix& X) {
  DenseMatrix Xt = X.transpose();
  return .5 * (X + Xt); 
}


Eigen::VectorXd vec(const DenseMatrix& A) {
  return Eigen::Map<const Eigen::VectorXd>(A.data(), A.rows()*A.cols());
}

DenseMatrix RandomPSD(int n) {
  DenseMatrix c = Symmetrize(DenseMatrix::Random(n, n));
  return c * c;
}

DenseMatrix RandomSym(int n) {
  return Symmetrize(DenseMatrix::Random(n, n));
}


DenseMatrix RandomLinear(int n) {
  DenseMatrix c = DenseMatrix::Random(n, 1);
  return c.cwiseProduct(c);
}

vector<SparseMatrixTuple> Partition(DenseMatrix& m) {
  vector<SparseMatrixTuple> tuples;
  map<double, int> classes;
  int num_classes = 0;
  for (int i = 0; i < m.rows(); i++) {
    for (int j = i; j < m.rows(); j++) {
      int class_ij = num_classes;
      if (classes.find(m(i, j)) != classes.end()) {
        class_ij = classes.find(m(i, j))->second;
      } else {
        classes[m(i, j)] = num_classes;
        tuples.push_back({});
        num_classes++;
      }
      tuples.at(class_ij).push_back(std::pair<int, int>{i, j});
    }
  }
  return tuples;
}

vector<SparseMatrixTuple> GetRandomTuples(int n, int m) {
  vector<SparseMatrixTuple> myVect;

  if (m > .5*n*n + .5*n) {
    bool valid_inputs = false;
    assert(valid_inputs);
  }

  Eigen::Matrix<double, -1, -1> Mask = Eigen::Matrix<double, -1, -1>::Zero(n, n);
  while(myVect.size() < static_cast<unsigned int>(m)) {
    Eigen::MatrixXd Mi = Eigen::Matrix<double, -1, -1>::Random(n, n);
    Mask += Symmetrize((Mi*10).array().round().abs());
    myVect = Partition(Mask);
  }
  
  vector<SparseMatrixTuple>::const_iterator first = myVect.begin();
  vector<SparseMatrixTuple>::const_iterator last = myVect.begin() + m;
  return vector<SparseMatrixTuple>(first, last);
}

vector<DenseMatrix> GetRandomDenseMatrices(int n, int m) {
  vector<DenseMatrix> y(m);
  for (int i = 0; i < m; i++) {
    y.at(i) = RandomSym(n); 
  }
  return y;
}

double NormFro(const DenseMatrix& x) {
  return std::sqrt((x*x.transpose()).trace());
}


double min(double x, double y) {
  double z = x;
  if (z > y) {
    z = y;
  }
  return z;
}


DenseMatrix QuadRep(const DenseMatrix& x, const DenseMatrix& y) {
  return x*y*x;
}
