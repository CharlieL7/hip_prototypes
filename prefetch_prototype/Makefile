HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES_0 = conv_add_sync_copy.cpp
OBJECTS_0 = $(SOURCES_0:.cpp=.o)

SOURCES_1 = conv_add_async_copy.cpp
OBJECTS_1 = $(SOURCES_1:.cpp=.o)

EXECUTABLE_0=./conv_add_sync_copy
EXECUTABLE_1=./conv_add_async_copy

.PHONY: test


all: sync_copy async_copy

sync_copy: $(EXECUTABLE_0)

async_copy: $(EXECUTABLE_1)

CXXFLAGS =-g
CXX=$(HIPCC)

$(EXECUTABLE_0): $(OBJECTS_0)
	$(HIPCC) $(OBJECTS_0) -o $@

$(EXECUTABLE_1): $(OBJECTS_1)
	$(HIPCC) $(OBJECTS_1) -o $@
