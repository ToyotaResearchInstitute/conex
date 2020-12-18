COPTS=-O3  -fPIC -g -std=c++17 -fno-omit-frame-pointer -DEIGEN_USE_BLAS  -fvisibility=hidden -fdata-sections -ffunction-sections  -I$(MKLROOT)/include
COPTSI=-O3  -fPIC -g -std=c++17 -fno-omit-frame-pointer -DEIGEN_USE_BLAS  -fdata-sections -ffunction-sections -I$(MKLROOT)/include

EIGEN=$(EIGEN_DIR)
CONEX=../

CPPFLAGS=$(COPTS) -I $(EIGEN) -I $(CONEX)
all: libconex.so

SOURCES = approximate_eigenvalues.cc \
	  block_triangular_operations.cc \
	  clique_ordering.cc \
	  cone_program.cc \
	  dense_lmi_constraint.cc \
	  divergence.cc \
	  exponential_map.cc \
	  hermitian_psd.cc \
	  kkt_assembler.cc \
	  kkt_solver.cc \
	  linear_constraint.cc \
	  matrix_exponential.cc \
	  psd_constraint.cc \
	  quadratic_cone_constraint.cc \
	  soc_constraint.cc \
	  supernodal_solver.cc \
	  triangular_matrix_workspace.cc \
	  vect_jordan_matrix_algebra.cc \

OBJECTS := $(SOURCES:.cc=.o)

%.o : ../conex/%.cc
	$(CXX) -c $(CPPFLAGS) $< 

libsource: $(SOURCES:.cc=.o)

conex.o: conex.h conex.cc
	g++ -c $(COPTSI)  -I${EIGEN_DIR} -I $(CONEX) conex.cc

libconex.so:  conex.o libsource
	g++ -static-libstdc++ -Wl,--exclude-libs,ALL -Wl,--gc-sections -shared -o libconex.so conex.o $(OBJECTS) -lblas 

main: libconex.so
	g++ test/test_app.cc libconex.so

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out
