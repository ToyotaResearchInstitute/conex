function str = ConexLibraryName()
  if ismac
    str = 'libconex.so';
  elseif isunix
    str = 'libconex.so';
  elseif ispc
    str = 'libconex.dll';
  else
    error('Platform not supported') 
  end
