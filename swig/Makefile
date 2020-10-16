COPTS=-O0 -g -std=c++17 -fno-omit-frame-pointer -I/home/frank/lintpaths/
LD=gold

EIGEN=$(EIGEN_DIR)
BAZEL_EXTERNAL=../../../bazel-codegen_examples/external
CONEX="/home/frank/research/collison/swig"
CXX=clang++

conex.o: conex.cc conex.h ../constraint.h ../linear_constraint.h
	$(CXX) -c -std=c++17 $(COPTS) -fpic -I $(EIGEN) -I $(CONEX)  conex.cc

libconex.so: 
	bazel build :libconex.so
	cp ../bazel-bin/swig/libconex.so .

a.out: libconex.so main.cc
	$(CXX) $(COPTS) -I $(EIGEN) -fuse-ld=gold main.cc -L. -lconex 

_conex.so: libconex.so
	rm -rf build
	python setup.py build
	cp build/lib.linux-x86_64-2.7/*.so .

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out
