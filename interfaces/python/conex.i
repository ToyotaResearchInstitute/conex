%module conex

%{
    #define SWIG_FILE_WITH_INIT
    #include "../conex.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* IN_FARRAY2, int DIM1, int DIM2) {(const double* A, int Ar, int Ac)}
%apply (double* IN_FARRAY2, int DIM1, int DIM2) {(const double* cmat, int cr, int cc)}
%apply (double* IN_FARRAY3, int DIM1, int DIM2, int DIM3) {(const double* Aarray, int Aarrayr, int Aarrayc, int m)}
%apply (double* IN_ARRAY1, int DIM1) {(const double* b, int br)}
%apply (double* IN_ARRAY1, int DIM1) {(const double* c, int cr)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* y, int yr)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* x, int xr)}
%apply (long* INPLACE_ARRAY1, int DIM1) {(const long* vars, int vars_c)}
%apply (double* INPLACE_FARRAY2, int DIM1, int DIM2) {(double* x, int xr, int xc)}

%include "../conex.h"
