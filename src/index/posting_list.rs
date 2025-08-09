use crate::query::cursor::{Cursor, RangeMaxScore, RangeMaxScoreCursor};
use serde::{Deserialize, Serialize};
use std::mem;

#[repr(C, align(64))]
struct AlignToSixtyFour([u8; 64]);
// this allocates a Vec<u8> where the raw data is aligned to 64 bytes
unsafe fn aligned_vec<'a>(n_bytes: usize) -> Vec<u8> {
    // Lazy math to ensure we always have enough.
    let n_units = (n_bytes / mem::size_of::<AlignToSixtyFour>()) + 1;

    let mut aligned: Vec<AlignToSixtyFour> = Vec::with_capacity(n_units);

    let ptr = aligned.as_mut_ptr();
    let len_units = aligned.len();
    let cap_units = aligned.capacity();

    mem::forget(aligned);

    Vec::from_raw_parts(
        ptr as *mut u8,
        len_units * mem::size_of::<AlignToSixtyFour>(),
        cap_units * mem::size_of::<AlignToSixtyFour>(),
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub enum BlockData {
    Compressed(Vec<CompressedBlock>),
    Raw(Vec<u8>),
}

impl Default for BlockData {
    fn default() -> Self {
        BlockData::Raw(Vec::new())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedBlock {
    pub max_scores: Vec<(usize, u8)>, // pairs of offset and max score
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PostingList {
    range_maxes: Vec<BlockData>,
    kth_score: Vec<u8>,
}

impl PostingList {
    pub fn new(range_maxes: Vec<BlockData>, kth_score: Vec<u8>) -> Self {
        PostingList {
            range_maxes,
            kth_score,
        }
    }
    const fn map_kth_value(value: usize) -> usize {
        match value {
            10 => 0,
            100 => 1,
            1000 => 2,
            _ => value, // default case
        }
    }

    pub fn kth(&self, k: usize) -> u8 {
        return self.kth_score[Self::map_kth_value(k)];
    }

    pub fn iter(&self, term_id: u32, term_weight: u32) -> PostingListIterator<'_> {
        PostingListIterator::new(self, term_id, term_weight)
    }

    // this is used to re-allocate the block data such that it is aligned to 64 bytes
    // should be done during deserialization, but i don't wanna mess with the off-the-shelf module
    pub fn align_block_data(&mut self) {
        let mut aligned_range_maxes: Vec<BlockData> = Vec::with_capacity(self.range_maxes.len());
        match self.range_maxes[0] {
            BlockData::Compressed(_) => return, // not implemented
            BlockData::Raw(_) => {}
        }
        // align block data
        for block_data in self.range_maxes.iter() {
            match block_data {
                BlockData::Compressed(_) => return, // not implemented
                BlockData::Raw(upper_bounds) => unsafe {
                    let mut new_container = self.aligned_vec(upper_bounds.len());
                    for ub in upper_bounds.iter() {
                        new_container.push(*ub);
                    }
                    aligned_range_maxes.push(BlockData::Raw(new_container));
                },
            }
        }

        // replace the old range_maxes with the aligned one
        self.range_maxes = aligned_range_maxes;
    }

    // this allocates a Vec<u8> where the raw data is aligned to 64 bytes
    unsafe fn aligned_vec(&self, n_bytes: usize) -> Vec<u8> {
        // Lazy math to ensure we always have enough.
        let n_units = (n_bytes / mem::size_of::<AlignToSixtyFour>()) + 1;

        let mut aligned: Vec<AlignToSixtyFour> = Vec::with_capacity(n_units);

        let ptr = aligned.as_mut_ptr();
        let len_units = aligned.len();
        let cap_units = aligned.capacity();

        mem::forget(aligned);

        Vec::from_raw_parts(
            ptr as *mut u8,
            len_units * mem::size_of::<AlignToSixtyFour>(),
            cap_units * mem::size_of::<AlignToSixtyFour>(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PostingListIterator<'a> {
    posting_list: &'a PostingList,
    current: usize,
    term_id: u32,
    term_weight: u32,
}

impl<'a> PostingListIterator<'a> {
    pub fn new(posting_list: &'a PostingList, term_id: u32, term_weight: u32) -> Self {
        PostingListIterator {
            posting_list,
            current: usize::MAX,
            term_id,
            term_weight,
        }
    }

    pub fn kth(&self, k: usize) -> u8 {
        return self.posting_list.kth(k);
    }

    pub fn term_weight(&self) -> u8 {
        self.term_weight as u8
    }

    pub fn term_id(&self) -> u32 {
        self.term_id
    }

    pub fn position(&self) -> usize {
        self.current
    }
}

impl<'a> Cursor for PostingListIterator<'a> {
    type Value = u32;

    type Error = ();

    fn reset(&mut self) {
        self.current = usize::MAX;
    }
}

impl<'a> RangeMaxScoreCursor for PostingListIterator<'a> {
    fn range_max_scores(&self) -> Vec<RangeMaxScore> {
        let mut rms = Vec::new();
        for range_max in self.posting_list.range_maxes.iter() {
            rms.push(match range_max {
                BlockData::Compressed(compressed_block) => {
                    RangeMaxScore::Compressed(compressed_block)
                }
                BlockData::Raw(raw_bytes) => RangeMaxScore::Raw(raw_bytes),
            });

            /*BlockData::Raw(raw_bytes) => unsafe {
                    /*let mut new_container = aligned_vec(raw_bytes.len());
                    for ub in raw_bytes.iter() {
                        new_container.push(*ub);
                    }*/
                    let (_, new_container, _) = raw_bytes.align_to::<u64>();
                    RangeMaxScore::Raw(std::mem::transmute(new_container))
                },
            })*/
        }
        /*match &self.posting_list.range_maxes {
            BlockData::Compressed(compressed_block) => RangeMaxScore::Compressed(compressed_block),
            BlockData::Raw(raw_bytes) => RangeMaxScore::Raw(raw_bytes),
        }*/
        rms
    }
}
