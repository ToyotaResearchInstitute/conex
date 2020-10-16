all: _conex.so

libconex.so:  ../conex.h ../conex.cc ../libconex.so
	cp ../libconex.so libconex.so

conex.py: ../conex.h ../conex.cc
	python setup.py build

build: ../conex.h ../conex.cc
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
