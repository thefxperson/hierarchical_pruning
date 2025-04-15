# Dynamic Superblock Pruning for Fast Learned Sparse Retrieval

This repository contains the code for Superblock Pruning, as presented at SIGIR '25.

Please cite the following paper if you use this code or a modification of it:
```bibtex
@inproceedings{carlson2025sp,
  author = {Parker Carlson and Wentai Xie and Shanxiu He and Tao Yang},
  title = {Dynamic Superblock Pruning for Fast Learned Sparse Retrieval},
  booktitle = {The 48th International ACM SIGIR Conference on Research and Development in Information Retrieval ({SIGIR '25})},
  publisher = {ACM},
  year = {2025}
}
```

SP is built upon [BMP](https://github.com/pisa-engine/BMP/tree/main) and shares most of its codebase.

### Usage

#### Data
Like BMP, the CIFF files and the queries required by SP to generate an index and perform search operations can be found in the [CIFF-Hub](https://github.com/pisa-engine/ciff-hub/tree/main).

**One requirement for SP to work correctly is that the impact scores of the CIFF files have to be quantized to 8 bits. This is not always done and for this reason is highly recommended to use the CIFF files from the Hub**

#### Index
The following command creates a SP index with a superblock size of 512 and subblock size of 8 (c=512/8=64).
```
./target/release/ciff2bmp -b 512 8 -c ./bp-msmarco-passage-unicoil-quantized.ciff -o bp-msmarco-passage-unicoil-quantized.bmp
```
Currently, SP does not support compression (BMP's `--compress-range`).

#### Search
Mu controls threshold overestimation for superblocks, where eta controls threshold overestimation for subblocks. Eta also decreases the probabalistic safeness of superblock pruning. Beta can be used as in BMP for query term pruning. 
```
./target/release/search --index bp-msmarco-passage-unicoil-quantized.bmp --queries splade_queries.dev --k 1000 --mu 0.6 --eta 0.9 --beta 0.95 > res.trec
```

#### Known Issues
If superblock pruning is too aggressive and prunes every superblock, search will crash. Annecdotally, this tends to occurs with mu values below 0.4 on MS MARCO with SPLADE, though it varies by dataset and embedding. We hope to address this limitation in future work.
