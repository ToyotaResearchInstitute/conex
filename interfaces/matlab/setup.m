function setup()
    %TODO(FrankPermenter): Remove this. 
    setenv('BLAS_VERSION','/usr/lib/x86_64-linux-gnu/libblas.so.3')

    display('Updating path...')
    directory = fileparts(which('setup.m'));
    addpath([directory,'/util'])
    addpath([directory,'/test'])
    addpath(directory)

    addpath('~/Downloads/SeDuMi_1_3/')

    cd '~/Downloads/SDPT3-4.0'
    startup
    Installmex
    cd ~/conexnew/conex/interfaces/matlab/

    %Checking dependencies
    display('Checking dependencies...')
 %   if ~exist(ConexLibraryName(), 'file') 
 %      error('Cannot find Conex solver library. Please compile and add it to path.') 
 %   end
    display('Done!')
    display('Type run_conex_tests to check installation.')
