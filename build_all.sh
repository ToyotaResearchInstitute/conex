set -e

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$PWD/interfaces"
export LIBRARY_PATH="${LIBRARY_PATH}:$PWD/interfaces"
export PYTHONPATH="${PYTHONPATH}:$PWD/interfaces/python"

mydir=$(dirname $(realpath $0))
[[ $PWD != $mydir ]] && { echo "Error: Script cannot be run from different directory."; exit 1; }


bazel_config=blas
bazel run //:buildifier --config=$bazel_config
find ./conex/ -iname *.h -o -iname *.cc | xargs clang-format -i
find ./interfaces/ -iname *.h -o -iname *.cc | xargs clang-format -i


## Build and test repo. 
cd conex 
bazel test --config=$bazel_config ...
cd ..

## Build and test C API. 
cd interfaces
make
bazel test --cache_test_results=no --config=$bazel_config ...
cd ..

## Build and test Python swig interface. 
cd interfaces/python
make clean
make
cd ../../
python interfaces/python/test/run_tests.py
