all: _conex.so

numpy.i:
	python dependencies/get_numpy_interface.py

_conex.so: numpy.i ../libconex.so 
	rm -rf build
	rm -f conex_wrap.c
	python dependencies/setup.py build
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
