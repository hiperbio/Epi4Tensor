CXX = nvcc
CXXFLAGS = --default-stream per-thread -O3 -lineinfo -Xcompiler -Icutlass-2.5.0/include -Icutlass-2.5.0/tools/util/include -Iinclude -DNUM_GPUS=1
EXE_NAME = epi4tensor
SOURCES = src/cutlass-op.cu src/epistasis.cu src/epi4tensor.cu
BINDIR = bin


sm86_and:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) $(CXXFLAGS) -arch=sm_86 -DAMPERE_86_AND -Xcompiler -fopenmp -o $(BINDIR)/$(EXE_NAME)
	

sm86_xor:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) $(CXXFLAGS) -arch=sm_86 -DAMPERE_86_XOR -Xcompiler -fopenmp -o $(BINDIR)/$(EXE_NAME)



sm75:
	[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CXX) $(SOURCES) $(CXXFLAGS) -arch=sm_75 -Xcompiler -fopenmp -o $(BINDIR)/$(EXE_NAME)

clean: 
	rm -rf $(BINDIR)/$(EXE_NAME)

