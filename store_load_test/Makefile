HIP_PATH?= $(wildcard /opt/rocm)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES_0 = add_one_add_two.cpp
OBJECTS_0 = $(SOURCES_0:.cpp=.o)

SOURCES_1 = add_two_only.cpp
OBJECTS_1 = $(SOURCES_1:.cpp=.o)

EXECUTABLE_0=./add_one_add_two.exe
EXECUTABLE_1=./add_two_only.exe

.PHONY: test


all: one

one: $(EXECUTABLE_0)

two: $(EXECUTABLE_1)

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
