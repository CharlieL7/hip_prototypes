HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES_0 = conv_prefetch_add.cpp
OBJECTS_0 = $(SOURCES_0:.cpp=.o)

EXECUTABLE_0=./conv_prefetch_add.exe

.PHONY: test


all: conv_prefetch

conv_prefetch: $(EXECUTABLE_0)

CXXFLAGS =-std=c++17
CXX=$(HIPCC)

$(EXECUTABLE_0): $(OBJECTS_0)
	$(HIPCC) $(OBJECTS_0) -o $@

clean:
	rm -f $(OBJECTS_0)
	rm -f $(EXECUTABLE_0)
