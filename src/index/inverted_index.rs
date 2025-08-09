use super::posting_list::{BlockData, PostingList, PostingListIterator};
use fst::{Map, MapBuilder};
use half::f16;
use num_integer::div_ceil;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp;

#[derive(Default, Serialize, Deserialize)]
pub struct Index {
    // #[serde(skip_serializing, skip_deserializing)]
    num_documents: usize,

    posting_lists: Vec<PostingList>,
    #[serde(
        serialize_with = "serialize_fst_map",
        deserialize_with = "deserialize_fst_map"
    )]
    // #[serde(skip_serializing, skip_deserializing)]
    termmap: Map<Vec<u8>>,
    // #[serde(skip_serializing, skip_deserializing)]
    documents: Vec<String>,
    pub bsize: Vec<usize>,
}

// Serialization function for the FST Map
fn serialize_fst_map<S>(termmap: &Map<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let bytes = termmap.as_fst().to_vec();
    serializer.serialize_bytes(&bytes)
}

// Deserialization function for the FST Map
fn deserialize_fst_map<'de, D>(deserializer: D) -> Result<Map<Vec<u8>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let bytes = Vec::<u8>::deserialize(deserializer)?;
    let map = Map::new(bytes).map_err(serde::de::Error::custom)?;
    Ok(map)
}

impl Index {
    pub fn new(num_documents: usize) -> Self {
        Index {
            num_documents,
            posting_lists: Vec::new(),
            termmap: Map::default(),
            documents: Vec::new(),
            bsize: Vec::new(),
        }
    }
    pub fn documents(&self) -> &Vec<String> {
        &self.documents
    }

    pub fn posting_lists(&self) -> &Vec<PostingList> {
        &self.posting_lists
    }

    pub fn num_documents(&self) -> usize {
        self.num_documents
    }

    pub fn get_cursor(&self, term: &str, term_weight: u32) -> Option<PostingListIterator> {
        self.termmap.get(term).map(|position| {
            self.posting_lists[position as usize].iter(position as u32, term_weight)
        })
    }

    pub fn align_posting_lists(&mut self) {
        for pl in self.posting_lists.iter_mut() {
            pl.align_block_data();
        }
    }
}

#[derive(Default)]
pub struct IndexBuilder {
    num_documents: usize,
    bsize: Vec<usize>,
    posting_lists: Vec<Vec<(u32, u32)>>,
    terms: Vec<String>,
    documents: Vec<String>,
}

impl IndexBuilder {
    pub fn new(num_documents: usize, bsize: Vec<usize>) -> Self {
        IndexBuilder {
            num_documents,
            bsize,
            posting_lists: Vec::new(),
            terms: Vec::new(),
            documents: Vec::new(),
        }
    }

    pub fn insert_term(&mut self, term: &str, list: Vec<(u32, u32)>) {
        self.posting_lists.push(list);
        self.terms.push(term.to_string());
    }

    pub fn push_posting(&mut self, term_id: u32, doc_id: u32, tf: u32) {
        // Pushes the doc_id and tf to the posting assocaited with term_id. This function assumes doc_ids are added in an increasing order.
        self.posting_lists[term_id as usize].push((doc_id, tf));
    }

    pub fn insert_document(&mut self, name: &str) -> u32 {
        let doc_id = self.documents.len();
        self.documents.push(name.to_string());
        return doc_id as u32;
    }

    fn compress(data: &[u8]) -> Vec<crate::index::posting_list::CompressedBlock> {
        let mut compressed = Vec::new();

        for superblock in data.chunks(256) {
            let mut max_scores = Vec::new();

            for (offset, &value) in superblock.iter().enumerate() {
                if value > 0 {
                    max_scores.push((offset, value));
                }
            }

            compressed.push(crate::index::posting_list::CompressedBlock { max_scores });
        }

        compressed
    }

    pub fn build(self, compress_range: bool) -> Index {
        let mut num_docs = self.num_documents;
        if num_docs == 0 {
            num_docs = self.documents.len();
        }
        let posting_lists: Vec<PostingList> = self
            .posting_lists
            .into_par_iter()
            .map(|p_list| {
                // calculate each set of blockmaxes for each bsize
                let mut all_range_maxes: Vec<_> = Vec::new();
                for (i, b) in self.bsize.iter().enumerate() {
                    let range_size = *b;
                    let blocks_num = div_ceil(num_docs, range_size);
                    let mut range_maxes: Vec<u8> = vec![0; blocks_num];
                    let mut range_avgs_accum: Vec<u8>; // assume tf < u16 and bsize < u16
                    let mut range_avgs: Vec<u8>;
                    // for the first (largest) b, we also want to compute avg_s_bound to use ASC
                    if i < self.bsize.len() - 1 {
                        // compute max and averages
                        let n_segments = 8;
                        let segment_size = div_ceil(range_size, n_segments);
                        range_avgs_accum = vec![0; blocks_num * n_segments];
                        range_avgs = vec![0; 2 * blocks_num]; // treat f16 as 2xu8 for type
                                                              // compatibility
                        p_list.iter().for_each(|&(docid, score)| {
                            let current_max = &mut range_maxes[docid as usize / range_size];
                            *current_max = cmp::max(*current_max, score as u8);
                            let current_boundS =
                                &mut range_avgs_accum[docid as usize / segment_size];
                            *current_boundS = cmp::max(*current_boundS, score as u8);
                        });
                        for i in (0..(range_avgs_accum.len() / n_segments)).step_by(n_segments) {
                            let mut accum = 0;
                            let mut n_nonzero = 0;
                            for j in 0..n_segments {
                                accum += range_avgs_accum[i + j] as u32;
                                if range_avgs_accum[i + j] > 0 {
                                    n_nonzero += 1;
                                }
                            }
                            let avg = f16::from_f32(accum as f32 / n_nonzero as f32);

                            range_avgs[i * 2] = avg.to_bits().to_ne_bytes()[0];
                            range_avgs[i * 2 + 1] = avg.to_bits().to_ne_bytes()[1];
                        }

                        all_range_maxes.push(match compress_range {
                            true => panic!("Compressed block not implemented"),
                            false => BlockData::Raw(range_avgs),
                        });
                        all_range_maxes.push(match compress_range {
                            true => BlockData::Compressed(Self::compress(&range_maxes)),
                            false => BlockData::Raw(range_maxes),
                        });
                    } else {
                        // only compute range_maxes
                        p_list.iter().for_each(|&(docid, score)| {
                            let current_max = &mut range_maxes[docid as usize / range_size];
                            *current_max = cmp::max(*current_max, score as u8);
                        });
                        all_range_maxes.push(match compress_range {
                            true => BlockData::Compressed(Self::compress(&range_maxes)),
                            false => BlockData::Raw(range_maxes),
                        });
                    }
                }
                let mut sorted_scores: Vec<u32> = p_list.iter().map(|&(_, score)| score).collect();
                sorted_scores.sort_by(|a, b| b.cmp(&a));

                // Retrieve the 10th, 100th and 1000th elements
                let s10th = sorted_scores.get(9).copied().unwrap_or(0) as u8;
                let s100th = sorted_scores.get(99).copied().unwrap_or(0) as u8;
                let s1000th = sorted_scores.get(999).copied().unwrap_or(0) as u8;

                PostingList::new(all_range_maxes, vec![s10th, s100th, s1000th])
            })
            .collect();

        let mut build = MapBuilder::memory();

        let mut indexed_terms: Vec<(usize, &String)> = self.terms.iter().enumerate().collect();

        // Sort the terms lexicographically while keeping the original indices
        indexed_terms.sort_by(|a, b| a.1.cmp(b.1));

        indexed_terms.iter().for_each(|(index, term)| {
            let _ = build.insert(term, *index as u64);
        });

        Index {
            num_documents: num_docs,
            posting_lists,
            termmap: Map::new(build.into_inner().unwrap()).unwrap(),
            documents: self.documents,
            bsize: self.bsize,
        }
    }
}
