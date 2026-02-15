import random

paths = { "train": (800, "train-labels", "train-inputs"), 
          "test":  (200, "test-labels", "test-inputs") }

def gen_dataset(l_file, i_file, size):
    data = [(random.getrandbits(1), random.getrandbits(1)) for _ in range(size)]
    
    i_file.write(bytearray([b for tup in data for b in tup]))
    l_file.write(bytearray([a ^ b for a, b in data]))

for size, l_path, i_path in paths.values():
    with open(l_path, 'wb') as lf, open(i_path, 'wb') as ifile:
        # Write Little-Endian Headers
        [f.write(size.to_bytes(4, 'little')) for f in (lf, ifile)]
        gen_dataset(lf, ifile, size)
