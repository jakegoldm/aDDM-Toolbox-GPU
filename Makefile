NVCC := /usr/local/cuda-12.2/bin/nvcc
CXX := g++

# MACROS := -DEXCLUDE_CUDA_CODE

SIM_EXECS := addm_simulate ddm_simulate
MLE_EXECS := addm_mle ddm_mle
TEST_EXECS := test benchmark

LIB_DIR := lib
OBJ_DIR := obj
INC_DIR := include
BUILD_DIR := bin
SRC_DIR := sample

CXXFLAGS := -Ofast -march=native -fPIC $(MACROS)
NVCCFLAGS := -O3 -Xcompiler -fPIC
SHAREDFLAGS = -I $(INC_DIR) -lpthread
LDFLAGS := -shared
LIB := -L lib -lpthread
INC := -I $(INC_DIR)

INSTALL_LIB_DIR := /usr/lib
INSTALL_INC_DIR := /usr/include

PY_SUFFIX := $(shell python3-config --extension-suffix)
PY_INCLUDES := $(shell python3 -m pybind11 --includes)
PY_SO_FILE := $(addsuffix $(PY_SUFFIX), addm_toolbox_gpu)

CPP_FILES := $(filter-out $(LIB_DIR)/bindings.cpp, $(wildcard $(LIB_DIR)/*.cpp))
CU_FILES := $(wildcard $(LIB_DIR)/*.cu)
CPP_OBJ_FILES := $(patsubst $(LIB_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_FILES))
ifeq ($(MACROS),)
	CU_OBJ_FILES := $(patsubst $(LIB_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_FILES)) 
endif

$(OBJ_DIR): 
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR): 
	mkdir -p $(BUILD_DIR)

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $(SHAREDFLAGS) -o $@ $<

$(OBJ_DIR)/%.o: $(LIB_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $(SHAREDFLAGS) -o $@ $<

define compile_target_cpp_nvcc
	$(CXX) $(CXXFLAGS) -c $(addprefix $(SRC_DIR)/, $1.cpp) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(NVCC) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef

define compile_target_cpp_cpu
	$(CXX) $(CXXFLAGS) -c $(addprefix $(SRC_DIR)/, $1.cpp) $(LIB) $(INC) -o $(addprefix $(OBJ_DIR)/, $1.o)
	$(CXX) $(addprefix $(OBJ_DIR)/, $1.o) $(CPP_OBJ_FILES) -o $(addprefix $(BUILD_DIR)/, $1)
endef


sim: $(OBJ_DIR) $(BUILD_DIR) $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(SIM_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(SIM_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 

mle: $(OBJ_DIR) $(BUILD_DIR) $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(MLE_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(MLE_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 

test: $(OBJ_DIR) $(BUILD_DIR) $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	$(foreach source, $(TEST_EXECS), $(call compile_target_cpp_nvcc, $(source));)
else 
	$(foreach source, $(TEST_EXECS), $(call compile_target_cpp_cpu, $(source));)
endif 


all: sim mle test


install: $(OBJ_DIR) $(BUILD_DIR) $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
ifeq ($(MACROS),)
	echo "" > $(INC_DIR)/macros.h
	$(NVCC) $(LDFLAGS) $(MACROS) -o $(INSTALL_LIB_DIR)/libaddm.so $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
else 
	echo "#define EXCLUDE_CUDA_CODE 1" > $(INC_DIR)/macros.h
	$(CXX) $(LDFLAGS) $(MACROS) -o $(INSTALL_LIB_DIR)/libaddm.so $(CPP_OBJ_FILES)
endif 
	cp -TRv $(INC_DIR) $(INSTALL_INC_DIR)/addm


pybind: 
ifeq ($(MACROS),)
	$(NVCC) $(LDFLAGS) $(NVCCFLAGS) $(PY_INCLUDES) $(INC) $(CPP_FILES) $(CU_FILES) $(LIB_DIR)/bindings.cpp -o $(PY_SO_FILE)
else 
	$(CXX) $(LDFLAGS) $(CXXFLAGS) $(PY_INCLUDES) $(INC) $(CPP_FILES) $(LIB_DIR)/bindings.cpp -o $(PY_SO_FILE)
endif


.PHONY: clean
clean:
	rm -rf $(OBJ_DIR)
	rm -rf $(BUILD_DIR)
	rm -rf docs
