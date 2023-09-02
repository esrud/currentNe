CXX_FLAGS=-Wall -Wextra -fopenmp -O2
OFNAME=currentNe

currentNe:
	$(CXX) $(CXX_FLAGS) -mcmodel=medium -o $(OFNAME) currentNe.cpp lib/progress.cpp
static:
	$(CXX) $(CXX_FLAGS) -L/opt/lib -mcmodel=medium -static -o $(OFNAME) currentNe.cpp lib/progress.cpp
clean:
	rm $(OFNAME)
