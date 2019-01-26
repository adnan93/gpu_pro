
EXECUTABLE := aesp

CU_FILES   := aesparallel.cu

CU_DEPS    :=

CC_FILES   := aesserial.c

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g 
LDFLAGS=-L/usr/local/cuda/lib64/

LIBS       :=
FRAMEWORKS := 


ifeq ($(ARCH), Darwin)
FRAMEWORKS += OpenGL GLUT
else
# Building on Linux
LIBS += GL glut cudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc
NVCCFLAGS= -m64 -arch compute_20

OBJS=$(OBJDIR)/aesserial.o $(OBJDIR)/aesparallel.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.c
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
