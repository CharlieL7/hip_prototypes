HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES_0 = conv_prefetch_add.cpp
OBJECTS_0 = $(SOURCES_0:.cpp=.o)

SOURCES_1 = conv_add_sync.cpp
OBJECTS_1 = $(SOURCES_1:.cpp=.o)

EXECUTABLE_0=./conv_prefetch_add.exe
EXECUTABLE_1=./conv_add_sync.exe

.PHONY: test


all: conv_prefetch sync_copy

conv_prefetch: $(EXECUTABLE_0)

sync_copy: $(EXECUTABLE_1)

CXXFLAGS =-std=c++17
CXX=$(HIPCC)

$(EXECUTABLE_0): $(OBJECTS_0)
	$(HIPCC) $(OBJECTS_0) -o $@

$(EXECUTABLE_1): $(OBJECTS_1)
	$(HIPCC) $(OBJECTS_1) -o $@

clean:
	rm -f $(OBJECTS_0)
	rm -f $(EXECUTABLE_0)
	rm -f $(OBJECTS_1)
	rm -f $(EXECUTABLE_1)
