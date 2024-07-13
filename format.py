import os
import sys
import struct

# trans fvecs/ivecs to .bin
# every row (dim vector....) to num dim vector.
def fit_meta_data_in_file(src_filee,dst_file):
    dim = 0
    vecs = bytearray(b"") 
    vector_size = 0
    total_num = 0
    with open(src_file, 'rb') as f:
        dim_bin = f.read(4)
        dim, = struct.unpack('i', dim_bin)
        vector_size = dim * 4
        f.seek(0,2) # move pointer to end
        total_num = int(f.tell() / (4+dim*4))
        print(f"vector dim is {dim}, total_num is {total_num}")
        cur_num = 0
        f.seek(0)
        while cur_num < total_num:
            f.read(4)
            vecs.extend(f.read(vector_size))
            cur_num += 1
        f.close()
    print(f"finish read, read size is {vecs.__len__()}")
    with open(dst_file, "wb") as f:
        f.write(struct.pack('i', total_num))
        f.write(struct.pack('i', dim))
        f.write(vecs)
        f.close()
if __name__ == '__main__':
    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    fit_meta_data_in_file(src_file,dst_file)
