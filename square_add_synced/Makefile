HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES_0 = square_add_synced.cpp
OBJECTS_0 = $(SOURCES_0:.cpp=.o)

EXECUTABLE_0=./square_add_synced.exe

.PHONY: test


all: trivial

trivial: $(EXECUTABLE_0)

CXXFLAGS =-std=c++17
CXX=$(HIPCC)

$(EXECUTABLE_0): $(OBJECTS_0)
	$(HIPCC) $(OBJECTS_0) -o $@

clean:
	rm -f $(OBJECTS_0)
	rm -f $(EXECUTABLE_0)
