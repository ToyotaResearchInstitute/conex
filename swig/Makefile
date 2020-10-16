

libconex.so: 
	bazel build :libconex.so
	cp ../bazel-bin/swig/libconex.so .
	#cp ~/research/conex/swig/libconex.so libconex.so

conex.py:
	python setup.py build

build:
	python setup.py build

_conex.so: libconex.so conex.py build
	cp build/lib.linux-x86_64-2.7/*.so .

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out
	rm -rf build
	rm -f conex.py
	rm -f conex.pyc
