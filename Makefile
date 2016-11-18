# Makefile for assimulo project.

compressed_sources := zips

# Vrtualenv + assimulo.
virtdir := virtual_assimulo
virtualenv_available := $(shell command -v virtualenv 2> /dev/null)
vp := $(virtdir)/bin
assimulo_src := Assimulo-2.9
assimulo_src_zip := $(compressed_sources)/Assimulo-2.9.zip

# Sundial specific.
cmake_available := $(shell command -v cmake 2> /dev/null)
sundials_src := sundials-2.7.0
sundials_src_tar := $(compressed_sources)/sundials-2.7.0.tar.gz
sundials_bulid_dir := $(sundials_src)/builddir
cmake_flags := -DCMAKE_C_FLAGS=-g -03 -fPCI

# LaTeX related variables.
texdir = TeX
output_pdf = fmnn05.pdf
source_pdf = main.pdf
source_tex = main.tex

tex_includes = $(wildcard $(texdir)/includes/*.tex)

.PHONY: all clean uncompress pdf

all: uncompress $(sundials_bulid_dir) $(virtdir)

$(sundials_bulid_dir):
ifndef cmake_available
	$(error "cmake is not available, please install cmake.")
endif
	mkdir $(sundials_bulid_dir)
	cd $(sundials_bulid_dir) &&	cmake $(cmake_flags) ../ && sudo make install

uncompress: $(assimulo_src) $(sundials_src)

$(assimulo_src):
	unzip $(assimulo_src_zip)

$(sundials_src):
	mkdir $(sundials_src)
	tar -xzvf $(sundials_src_tar)

clean:
	rm -rf $(assimulo_src)
	rm -rf $(sundials_src)
	rm -rf $(virtdir)
	rm -rf $(sundials_bulid_dir)

$(virtdir):
ifndef virtualenv_available
	$(error "virtualenv is not available, please install virtualenv, ex `pip install virtualenv`.")
endif
	virtualenv -p python3 $(virtdir)
	# Install nose.
	$(vp)/pip install nose
	# Install Cython.
	$(vp)/pip install Cython
	# Install numpy.
	$(vp)/pip install numpy
	# Install matplotlib.
	$(vp)/pip install scipy
	# Install matplotlib.
	$(vp)/pip install matplotlib
	# Install assimulo.
	cd $(assimulo_src) && ../$(vp)/python setup.py install --sundials-home=/usr/local

pdf: $(texdir)/$(output_pdf)

$(texdir)/$(output_pdf): $(texdir)/$(source_tex) $(tex_includes) BDF.py
	# Run code extractor each build.
	python code_extractor.py
	cd $(texdir) && \
	pdflatex $(source_tex) && \
	mv $(source_pdf) $(output_pdf)
