#pragma once
#include <iostream>
#include <memory>
#include <vector>

namespace conex {


template <typename T>
int SizeOf(const std::vector<T>& x) {
  int y = 0;
  for (const auto& xi : x) {
    y += SizeOf(xi);
  }
  return y;
}

// template <typename T>
// void print(T const& t){ std::cout << t << "\n"; }

template <typename T>
void print(const std::vector<T>& x) {
  for (const auto& xi : x) {
    print(xi);
  }
}

template <typename T>
void Initialize(std::vector<T>* x, double* ptr) {
  double* data = ptr;
  for (auto& xi : *x) {
    Initialize(&xi, data);
    data += SizeOf(xi);
  }
}

// Workspaces are actually smart pointers to type-erased data:
class Workspace {
 public:
  template <typename T>
  Workspace(T* t) : model(std::make_unique<Model<T>>(t)) {}

  friend void Initialize(Workspace* o, double* y) {
    o->model->do_initialize(y);
  }

  friend int SizeOf(const Workspace& o) { return o.model->do_sizeof(); }

  friend void print(const Workspace& o) { o.model->do_print(); }

 private:
  struct Concept {
    virtual void do_print() = 0;
    virtual void do_initialize(double* ptr) = 0;
    virtual int do_sizeof() = 0;
    virtual ~Concept() = default;
  };

  template <typename T>
  struct Model final : Concept {
    Model(T* t) : data(t) {}
    void do_print() override { print(*data); }

    void do_initialize(double* ptr) override { Initialize(data, ptr); }

    int do_sizeof() override { return SizeOf(*data); }

    T* data;
  };
  std::unique_ptr<Concept> model;
};

// Workspaces are actually smart pointers to type-erased data:
struct WorkspaceStats {
  static constexpr int max_iter = 1000;
  static constexpr int num_items = 2;
  friend void Initialize(WorkspaceStats* o, double* y) {
    o->sqrt_inv_mu = y;
    o->norm_inf_d = y + max_iter;
    o->initialized = true;
  }

  friend int SizeOf(const WorkspaceStats& o) { return max_iter * num_items; }

  friend void print(const WorkspaceStats& o) {
    for (int i = 0; i < max_iter; i++) {
      std::cout << "sqrt_inv_mu, norm_inf_d" << o.sqrt_inv_mu[i] << ", "
                << o.norm_inf_d[i] << "\n";
    }
  }

  bool IsInitialized() { return initialized; }

  double* sqrt_inv_mu;
  double* norm_inf_d;
  int num_iter;
  bool initialized = false;
};

} // namespace conex
