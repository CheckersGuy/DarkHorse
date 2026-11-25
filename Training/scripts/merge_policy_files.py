#!/usr/bin/env python3
"""Concatenate policy-sample files (POL1 format) into one file.

    merge_policy_files.py OUT IN1 [IN2 ...]

Each input starts with a 12-byte header: u32 magic 0x504F4C31 ("POL1") and a
u64 sample count (little endian, see Training/Rust/src/Sample.rs
write_policy_header). OUT gets one header with the summed count, followed by
the bodies of all inputs in order.
"""
import shutil
import struct
import sys

MAGIC = 0x504F4C31
HEADER = struct.Struct('<IQ')


def read_header(path):
    with open(path, 'rb') as f:
        magic, count = HEADER.unpack(f.read(HEADER.size))
    if magic != MAGIC:
        sys.exit(f"{path}: not a policy-sample file")
    return count


def main(out, inputs):
    counts = [read_header(p) for p in inputs]
    total = sum(counts)
    with open(out, 'wb') as fo:
        fo.write(HEADER.pack(MAGIC, total))
        for p in inputs:
            with open(p, 'rb') as fi:
                fi.seek(HEADER.size)
                shutil.copyfileobj(fi, fo, 64 << 20)
    for p, c in zip(inputs, counts):
        print(f"{p}: {c:,}")
    print(f"{out}: {total:,} samples")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
