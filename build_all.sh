set -e

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$PWD/interfaces"
export LIBRARY_PATH="${LIBRARY_PATH}:$PWD/interfaces"
export PYTHONPATH="${PYTHONPATH}:$PWD/interfaces/python"

mydir=$(dirname $(realpath $0))
[[ $PWD != $mydir ]] && { echo "Error: Script cannot be run from different directory."; exit 1; }


bazel run //:buildifier --config=mkl
find ./conex/ -iname *.h -o -iname *.cc | xargs clang-format -i
find ./interfaces/ -iname *.h -o -iname *.cc | xargs clang-format -i

bazel_config=blas

## Build and test repo. 
bazel test ... --config=$bazel_config

## Build and test C API. 
cd interfaces
make 
bazel test --cache_test_results=no --config=$bazel_config ...
cd ..

## Build and test Python swig interface. 
cd interfaces/python
make
cd ../../
python interfaces/python/test/run_tests.py
