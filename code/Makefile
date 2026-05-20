CXX      = c++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -DNDEBUG
TARGET   = drm
HEADERS  = core.h mga.h bgd.h dynamic_connectivity.h euler_tour_tree.h

$(TARGET): main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)
