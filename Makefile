CXXFLAGS := -std=c++17 -g -O3
TARGETS := bandwidthtest async transpose indicesofsetbits

indicesofsetbits: IndicesOfSetBitsCuda.o IndicesOfSetBits.o
	nvcc $(OUTPUT_OPTION) $^

IndicesOfSetBitsCuda.o: IndicesOfSetBitsCuda.cu
	nvcc $(OUTPUT_OPTION) $(CXXFLAGS) -lineinfo -c $<

IndicesOfSetBits.o: IndicesOfSetBits.cc IndicesOfSetBits.h
	gcc $(OUTPUT_OPTION) $(CXXFLAGS) -c -mavx2 $<

%: %.cu
	nvcc $(OUTPUT_OPTION) $(CXXFLAGS) $<

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	$(RM) $(TARGETS) *.o
