EIGEN=$(EIGEN_DIR)
CONEX=../

CPPFLAGS =-O3 -fPIC -std=c++17 -fdata-sections -ffunction-sections  

# BLAS 
#CPPFLAGS_LIBRARY=$(CPPFLAGS) -I $(EIGEN) -I $(CONEX) -DEIGEN_USE_BLAS -fvisibility=hidden
#LFLAGS=-lblas 

# MKL
CPPFLAGS_LIBRARY=$(CPPFLAGS) -I $(EIGEN) -I $(CONEX) -DEIGEN_USE_MKL_ALL -I$(MKLROOT)/include -fvisibility=hidden
LFLAGS= -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -fopenmp

all: libconex.so

SOURCES = approximate_eigenvalues.cc \
	  block_triangular_operations.cc \
	  clique_ordering.cc \
	  cone_program.cc \
	  dense_lmi_constraint.cc \
	  divergence.cc \
	  exponential_map.cc \
	  exponential_map_pade.cc \
	  hermitian_psd.cc \
	  kkt_assembler.cc \
	  kkt_solver.cc \
	  linear_constraint.cc \
	  psd_constraint.cc \
	  quadratic_cone_constraint.cc \
	  soc_constraint.cc \
	  supernodal_solver.cc \
	  triangular_matrix_workspace.cc \
	  tree_utils.cc \
	  jordan_matrix_algebra.cc \

OBJECTS := $(SOURCES:.cc=.o)

%.o : ../conex/%.cc
	$(CXX) -c $(CPPFLAGS_LIBRARY) $< 

libsource: $(SOURCES:.cc=.o)

conex.o: conex.h conex.cc
	g++ -c $(CPPFLAGS) -I${EIGEN_DIR} -I $(CONEX) conex.cc

libconex.so:  conex.o libsource
	g++ -static-libstdc++ -Wl,--exclude-libs,ALL -Wl,--gc-sections -shared -o libconex.so conex.o $(OBJECTS) $(LFLAGS)

main: libconex.so
	g++ test/test_app.cc libconex.so

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out
