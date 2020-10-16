function setup()

    display('Updating path...')
    directory = fileparts(which('setup.m'));
    addpath([directory,'/util'])
    addpath([directory,'/test'])
    addpath(directory)

    %Checking dependencies
    display('Checking dependencies...')
    if ~exist(ConexLibraryName(), 'file') 
       error('Cannot find Conex solver library. Please compile and add it to path.') 
    end
    display('Done!')
    display('Type run_conex_tests to check installation.')
