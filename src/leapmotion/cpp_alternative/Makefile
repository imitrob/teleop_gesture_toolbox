OS := $(shell uname)
ARCH := $(shell uname -m)

ifeq ($(OS), Linux)
  ifeq ($(ARCH), x86_64)
    LEAP_LIBRARY := /home/$(whoami)/LeapSDK/lib/x64/libLeap.so -Wl,-rpath,/home/$(whoami)/LeapSDK/lib/x64
  else
    LEAP_LIBRARY := /home/$(whoami)/LeapSDK/lib/x86/libLeap.so -Wl,-rpath,/home/$(whoami)/LeapSDK/lib/x86
  endif
else
  # OS X
  LEAP_LIBRARY := /home/$(whoami)/LeapSDK/lib/libLeap.dylib
endif

Sample: Sample.cpp
	$(CXX) -Wall -g -I/home/$(whoami)/LeapSDK/include Sample.cpp handclasses.cpp -o Sample $(LEAP_LIBRARY)
ifeq ($(OS), Darwin)
	install_name_tool -change @loader_path/libLeap.dylib /home/$(whoami)/LeapSDK/lib/libLeap.dylib Sample
endif

clean:
	rm -rf Sample Sample.dSYM
