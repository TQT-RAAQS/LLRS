# Makefile for building libawg.so with build directory

# Compiler settings
CXX = g++
CXXFLAGS = -Wall -fPIC
NVCC = nvcc
NVCCFLAGS = -Xcompiler -fPIC
LDFLAGS			= -Llib -lyaml-cpp -lhdf5_cpp -lhdf5_serial -l:libspcm_linux.so
CUDA_LDFLAGS 	= -L/usr/local/cuda/lib64/ -lcudart -lcufft -lcuda


# Include directory for header files
INCLUDES = -Iinclude

# Source files
SOURCES = src/awg.cpp \
          src/common/ostools/spcm_ostools.cpp \
          src/common/spcm_cuda_common.cu \
          src/common/spcm_lib_card.cpp \
          src/common/spcm_lib_data.cpp

# Build directory
BUILD_DIR = lib

# Object files corresponding to source files
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
OBJECTS := $(OBJECTS:%.cu=$(BUILD_DIR)/%.o)

# Output library
LIBRARY = $(BUILD_DIR)/libawg.so

# Rule to ensure build directories are created before compiling
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Linking rule for shared library
$(LIBRARY): $(OBJECTS)
	$(CXX) -shared -o $@ $(OBJECTS) $(LDFLAGS) $(CUDA_LDFLAGS)

# Phony targets
.PHONY: all clean

all: $(LIBRARY)

clean:
	rm -rf $(BUILD_DIR)
