COPTS=-O3  -fPIC -g -std=c++17 -fno-omit-frame-pointer 
EIGEN=$(EIGEN_DIR)
CONEX=../

all: libconex.so

cone_program.o: ../conex/cone_program.cc 
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/cone_program.cc 

linear_constraint.o: ../conex/linear_constraint.cc  
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/linear_constraint.cc 

divergence.o: ../conex/divergence.cc 
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/divergence.cc

dense_lmi_constraint.o: ../conex/dense_lmi_constraint.cc  ../conex/psd_constraint.h 
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/dense_lmi_constraint.cc 

psd_constraint.o: ../conex/psd_constraint.cc ../conex/psd_constraint.h 
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/psd_constraint.cc 

hermitian_psd.o: ../conex/hermitian_psd.cc
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/hermitian_psd.cc

soc_constraint.o: ../conex/soc_constraint.cc
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/soc_constraint.cc

eigen_decomp.o: ../conex/eigen_decomp.cc 
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/eigen_decomp.cc

matrix_exponential.o: ../conex/matrix_exponential.cc
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/matrix_exponential.cc

exponential_map.o: ../conex/exponential_map.cc
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/exponential_map.cc

approximate_eigenvalues.o: ../conex/approximate_eigenvalues.cc
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/approximate_eigenvalues.cc

vect_jordan_matrix_algebra.o: ../conex/vect_jordan_matrix_algebra.cc
	$(CXX) -c  $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/vect_jordan_matrix_algebra.cc

conex.o: conex.h conex.cc
	g++ -c $(COPTS)  -I${EIGEN_DIR} -I $(CONEX) conex.cc

libconex.so:  conex.o dense_lmi_constraint.o linear_constraint.o psd_constraint.o eigen_decomp.o cone_program.o divergence.o approximate_eigenvalues.o matrix_exponential.o hermitian_psd.o exponential_map.o vect_jordan_matrix_algebra.o soc_constraint.o
	g++ -static-libstdc++ -shared -o libconex.so conex.o dense_lmi_constraint.o linear_constraint.o psd_constraint.o eigen_decomp.o cone_program.o divergence.o approximate_eigenvalues.o matrix_exponential.o hermitian_psd.o exponential_map.o vect_jordan_matrix_algebra.o soc_constraint.o

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out
