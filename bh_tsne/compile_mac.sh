echo "Make sure to change the path to CBLAS in this file before running it!"
g++ sptree.cpp tsne.cpp -o bh_tsne -O3 -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers -lcblas
