#COPTS=-O0 -fPIC -g -std=c++17 -fno-omit-frame-pointer -static-libstdc++
COPTS=-O0 -fPIC -g -std=c++17 -fno-omit-frame-pointer 
EIGEN=$(EIGEN_DIR)
CONEX="../"


cone_program.o: 
	$(CXX) -c -std=c++17 $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/cone_program.cc 

linear_constraint.o: 
	$(CXX) -c -std=c++17 $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/linear_constraint.cc 

dense_lmi_constraint.o: 
	$(CXX) -c -std=c++17 $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/dense_lmi_constraint.cc 

psd_constraint.o: 
	$(CXX) -c -std=c++17 $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/psd_constraint.cc 

eigen_decomp.o: 
	$(CXX) -c -std=c++17 $(COPTS) -I $(EIGEN) -I $(CONEX) ../conex/eigen_decomp.cc

matrix_exponential.o: matrix_exponential.cc matrix_exponential.h 
	$(CXX) -c -std=c++17 $(COPTS) -I $(EIGEN)  ../conex/matrix_exponential.cc


libconex.so:  conex.h conex.cc dense_lmi_constraint.o linear_constraint.o psd_constraint.o eigen_decomp.o cone_program.o
	g++ -c -std=c++17 -O3 -fpic -I${EIGEN_DIR} -static-libstdc++ -I ../ conex.cc
	g++ -static-libstdc++ -shared -o libconex.so conex.o dense_lmi_constraint.o linear_constraint.o psd_constraint.o eigen_decomp.o cone_program.o

all: libconex.so

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out

