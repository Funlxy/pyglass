#!/bin/bash

if [ -d "build" ]; then
    rm -rf "build"
    echo "Directory build has been deleted."
fi
mkdir -p build
cd build
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && make -j
prefix=../datasets/sift

ef_values=(10 15 20 30 40 60 80 100 150 200 300)
for ef in "${ef_values[@]}"
do
    ./main --base_path ${prefix}/sift_base.fvecs \
    --query_path ${prefix}/sift_query.fvecs  \
    --gt_path ${prefix}/sift_groundtruth.ivecs  \
    --graph_path ${prefix}/sift1M.glass  \
    --level 0   \
    --topk 10   \
    --ef $ef
done