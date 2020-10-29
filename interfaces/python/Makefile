all: _conex.so

numpy.i:
	python dependencies/get_numpy_interface.py

libconex.so:  ../conex.h ../conex.cc ../libconex.so
	cp ../libconex.so libconex.so

conex.py: ../conex.h ../conex.cc numpy.i
	python dependencies/setup.py build

build: ../conex.h ../conex.cc numpy.i
	python dependencies/setup.py build

_conex.so: libconex.so conex.py build
	cp build/lib.linux-x86_64-2.7/*.so .

.PHONY: clean

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.out
	rm -f conex_wrap.c
	rm -rf build
	rm -f conex.py
	rm -f *.pyc
	rm -f numpy.i
