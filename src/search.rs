use crate::index::forward_index::block_score;
use crate::index::forward_index::BlockForwardIndex;
use crate::index::posting_list::PostingListIterator;
use crate::query::cursor::DocId;
use crate::query::cursor::{RangeMaxScore, RangeMaxScoreCursor};
use crate::query::live_block;
use crate::query::topk_heap::TopKHeap;
use crate::util::progress_bar;
use num_integer::div_ceil;
use std::arch::x86_64::_mm_prefetch;
use std::arch::x86_64::*;
use std::simd::*;
use std::time::Instant;

pub fn b_search(
    queries: Vec<Vec<PostingListIterator>>,
    forward_index: &BlockForwardIndex,
    k: usize,
    alpha: f32,
    beta: f32,
    gamma: f32,
    mu: f32,
    eta: f32,
    bsize: Vec<usize>,
) -> Vec<TopKHeap<u16>> {
    b_search_verbose(
        queries,
        forward_index,
        k,
        alpha,
        beta,
        gamma,
        mu,
        eta,
        bsize,
        true,
    )
}

pub fn b_search_verbose(
    mut queries: Vec<Vec<PostingListIterator>>,
    forward_index: &BlockForwardIndex,
    k: usize,
    alpha: f32,
    beta: f32,
    gamma: f32,
    mu: f32,
    eta: f32,
    bsize: Vec<usize>,
    verbose: bool,
) -> Vec<TopKHeap<u16>> {
    let mut results: Vec<TopKHeap<u16>> = Vec::new();

    let N_SEARCHES = 5;

    let mut search_elapsed = 0;
    let mut buckets: Vec<Vec<u32>> = (0..=2usize.pow(16)).map(|_| Vec::new()).collect();
    let mut idxs_buffer = vec![0; 1 + (8841823 / 8 * 4)];
    let mut progress = None;

    let prefetch_dist = 14;
    for i_search in 0..N_SEARCHES {
        if i_search == 0 && verbose {
            eprintln!("Warming Up Index...");
        } else {
            if i_search == 1 {
                progress = if verbose {
                    Some(progress_bar(
                        "Forward index-based search",
                        (N_SEARCHES - 1) * queries.len(),
                    ))
                } else {
                    None
                };
            }
            results.clear();
        }
        results.clear();
        for query in queries.iter_mut() {
            let total_terms = query.len();
            let terms_to_keep = (total_terms as f32 * beta).ceil() as usize;
            let qv_terms_to_keep = (total_terms as f32 * beta).ceil() as usize;
            query.sort_by(|a, b| b.term_weight().partial_cmp(&a.term_weight()).unwrap());

            let mut query_vec = query[0..qv_terms_to_keep]
                .iter()
                .map(|&pl| (pl.term_id() as u16, pl.term_weight() as u8))
                .collect::<Vec<_>>();
            query_vec.sort_by_key(|e| e.0);

            let query_weights: Vec<_> = query[0..terms_to_keep]
                .iter()
                .map(|post| post.term_weight())
                .collect();

            let query_ranges: Vec<Vec<_>> = query[0..terms_to_keep]
                .iter()
                .map(|post| post.range_max_scores())
                .collect();

            let mut query_ranges_raw: Vec<Vec<_>> =
                (0..query_ranges[0].len()).map(|_| Vec::new()).collect();
            let mut query_ranges_compressed: Vec<Vec<_>> =
                (0..query_ranges[0].len()).map(|_| Vec::new()).collect();
            for term_ranges in query_ranges {
                for (range_size, range_type) in term_ranges.iter().enumerate() {
                    match range_type {
                        // iter returns unwanted reference, so deref when adding to vec
                        RangeMaxScore::Compressed(compressed) => {
                            query_ranges_compressed[range_size].push(*compressed)
                        }
                        RangeMaxScore::Raw(raw) => query_ranges_raw[range_size].push(*raw),
                    }
                }
            }

            let threshold = query[0..terms_to_keep]
                .iter()
                .map(|&pl| pl.kth(k) as u16 * pl.term_weight() as u16)
                .max()
                .unwrap_or(0);

            // SEARCH
            let start_search: Instant = Instant::now();

            // clear previous buckets, presumably faster than realloc
            buckets.iter_mut().for_each(std::vec::Vec::clear);
            let mut topk = TopKHeap::with_threshold(k, threshold as u16);

            let mut ub_iter = hierarchical_calculate_asc(
                &query_ranges_compressed,
                &query_ranges_raw,
                &query_weights,
                forward_index,
                &mut buckets,
                threshold,
                prefetch_dist,
                mu,
                eta,
                &bsize,
                &mut idxs_buffer,
            );

            let (mut current_ub, mut current_block) = ub_iter.next().unwrap();
            unsafe {
                _mm_prefetch(
                    forward_index.data.as_ptr().add(*current_block as usize) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
            for (next_ub, next_block) in ub_iter {
                unsafe {
                    _mm_prefetch(
                        forward_index.data.as_ptr().add(*next_block as usize) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }

                let offset = *current_block as usize * forward_index.block_size;

                let res = block_score(
                    &query_vec,
                    &forward_index.data[*current_block as usize],
                    forward_index.block_size,
                );

                for (doc_id, &score) in res.iter().enumerate() {
                    topk.insert(DocId(doc_id as u32 + offset as u32), score);
                }

                if topk.threshold() as f32 > current_ub as f32 * eta {
                    break;
                }
                current_block = next_block;
                current_ub = next_ub;
            }
            search_elapsed += start_search.elapsed().as_micros();
            results.push(topk.clone());
            if i_search > 0 {
                if let Some(progress_bar) = &progress {
                    progress_bar.inc(1);
                }
            }
        }

        println!("{}", search_elapsed / results.len() as u128);
        search_elapsed = 0;
    }
    if let Some(progress_bar) = &progress {
        progress_bar.finish();
    }

    if verbose {
        eprintln!(
            "search_elapsed = {}",
            search_elapsed / results.len() as u128
        );
    }

    results
}

// Original BMP
#[inline]
fn calculate_upper_bounds<'a>(
    query_ranges_compressed: &'a Vec<&[crate::index::posting_list::CompressedBlock]>,
    query_ranges_raw: &'a Vec<&[u8]>,
    query_weights: &'a Vec<u8>,
    forward_index: &'a BlockForwardIndex,
    buckets: &'a mut Vec<Vec<u32>>,
    threshold: u16,
) -> impl Iterator<Item = (usize, &'a mut u32)> {
    let run_compressed = query_ranges_compressed.len() > 0;
    let upper_bounds = match run_compressed {
        true => live_block::compute_upper_bounds(
            &query_ranges_compressed,
            &query_weights,
            forward_index.data.len(),
        ),
        false => live_block::compute_upper_bounds_raw(
            &query_ranges_raw,
            &query_weights,
            forward_index.data.len(),
        ),
    };

    //buckets.iter_mut().for_each(std::vec::Vec::clear); handled by caller
    upper_bounds.iter().enumerate().for_each(|(range_id, &ub)| {
        if ub > threshold {
            buckets[ub as usize].push(range_id as u32);
        }
    });

    let ub_iter = buckets
        .iter_mut()
        .enumerate()
        .rev()
        .flat_map(|(outer_idx, inner_vec)| inner_vec.iter_mut().map(move |val| (outer_idx, val)));

    ub_iter
}

// SIGIR 25
#[inline]
fn hierarchical_calculate_asc<'a>(
    query_ranges_compressed: &'a Vec<Vec<&[crate::index::posting_list::CompressedBlock]>>,
    query_ranges_raw: &'a Vec<Vec<&[u8]>>,
    query_weights: &'a Vec<u8>,
    forward_index: &'a BlockForwardIndex,
    buckets: &'a mut Vec<Vec<u32>>,
    threshold: u16,
    prefetch_dist: usize,
    mu: f32,
    eta: f32,
    bsize: &Vec<usize>,
    idxs_buffer: &mut Vec<usize>,
) -> impl Iterator<Item = (usize, &'a mut u32)> {
    // TODO better solution for this
    let run_compressed = query_ranges_compressed[0].len() > 0;
    let stride = bsize[0] / bsize[1];
    // prefetch first set of blocks

    // change for ASC ablation
    let filter = first_asc(
        query_ranges_compressed,
        query_ranges_raw,
        query_weights,
        threshold,
        mu,
        eta,
        prefetch_dist,
        forward_index.data.len() * forward_index.block_size,
        bsize,
    );
    let upper_bounds = match run_compressed {
        true => panic!("Unimplemented"), /*live_block::compute_upper_bounds(
        &query_ranges_compressed[0],
        &query_weights,
        forward_index.data.len(),
        ),*/
        false => live_block::optimized_compute_filtered_upper_bounds_raw(
            &query_ranges_raw[query_ranges_raw.len() - 1],
            &query_weights,
            forward_index.data.len(),
            stride,
            &filter,
            prefetch_dist,
        ),
    };

    let (idxs, j) = simd_process_upper_bounds(&upper_bounds, threshold, idxs_buffer);

    for i in 0..(j - 1) {
        for k in 0..64 {
            let range_id = idxs[i] as usize + k;
            if upper_bounds[range_id] > threshold {
                buckets[upper_bounds[range_id] as usize].push(range_id as u32);
            }
        }
    }
    for k in idxs[j - 1]..upper_bounds.len() {
        if upper_bounds[k] > threshold {
            buckets[upper_bounds[k] as usize].push(k as u32);
        }
    }

    let ub_iter = buckets
        .iter_mut()
        .enumerate()
        .rev()
        .flat_map(|(outer_idx, inner_vec)| inner_vec.iter_mut().map(move |val| (outer_idx, val)));

    ub_iter
}

#[inline]
fn first_asc<'a>(
    query_ranges_compressed: &'a Vec<Vec<&[crate::index::posting_list::CompressedBlock]>>,
    query_ranges_raw: &'a Vec<Vec<&[u8]>>,
    query_weights: &'a Vec<u8>,
    threshold: u16,
    mu: f32,
    eta: f32,
    prefetch_dist: usize,
    n_docs: usize,
    bsize: &Vec<usize>,
) -> Vec<u32> {
    // TODO better solution for this
    let run_compressed = query_ranges_compressed[0].len() > 0;
    let stride = bsize[0] / bsize[1];
    let block_len = div_ceil(n_docs, bsize[0]);

    // first level -- don't need a filter
    let (raw_maxs, raw_avgs) = match run_compressed {
        true => panic!("Unimplemented"), //live_block::compute_upper_bounds(qrc, &query_weights, forward_index.data.len()),
        false => live_block::compute_asc_raw_intrinsic(
            &query_ranges_raw[1],
            &query_ranges_raw[0],
            &query_weights,
            block_len,
        ),
    };

    let mut top_blocks = TopKHeap::with_threshold(2, threshold as u16);
    let mut filter: Vec<u32> = Vec::with_capacity(block_len);
    let unsafe_threshold: u16 = (threshold as f32 / mu) as u16;
    let eta_threshold: u16 = (threshold as f32 / eta) as u16;
    // make space for top_blocks
    filter.push(0);
    filter.push(0);
    raw_maxs
        .iter()
        .zip(raw_avgs.iter())
        .enumerate()
        .for_each(|(range_id, (&max, &avg))| {
            // segments needs to be small enough so this actually has values
            if (max > unsafe_threshold) || (avg > eta_threshold as f32) {
                filter.push(range_id as u32);
            } else {
                // no repeat scoring
                top_blocks.insert(DocId(range_id as u32), max as u16);
            }
        });
    // this means the last elements of filter  may be out of order
    // should be okay because we loop by term first
    //
    //println!("filter len before heap {}", filter.len());
    // need to add these at the front to prevent an out of bounds error later
    let mut i = 0;
    for range_id in top_blocks.to_sorted_vec().iter() {
        filter[i] = range_id.doc_id.0;
        i += 1;
    }

    filter
}

#[inline]
pub fn simd_process_upper_bounds<'a>(
    upper_bounds: &'a Vec<u16>,
    threshold: u16,
    idxs: &'a mut Vec<usize>,
) -> (&'a mut Vec<usize>, usize) {
    let mut j = 0;
    let threshold_v;
    let ones_v;
    unsafe {
        threshold_v = _mm256_set1_epi16(threshold as i16);
        ones_v = _mm256_set1_epi8(-1); // ff
    }
    for i in (0..upper_bounds.len() - 64).step_by(64) {
        unsafe {
            let ub_v0 = _mm256_loadu_si256(upper_bounds.as_ptr().add(i) as *const __m256i);
            let ub_v1 = _mm256_loadu_si256(upper_bounds.as_ptr().add(i + 16) as *const __m256i);
            let ub_v2 = _mm256_loadu_si256(upper_bounds.as_ptr().add(i + 32) as *const __m256i);
            let ub_v3 = _mm256_loadu_si256(upper_bounds.as_ptr().add(i + 48) as *const __m256i);
            let max0 = _mm256_max_epu16(ub_v0, ub_v1);
            let max1 = _mm256_max_epu16(ub_v2, ub_v3);
            let max_reduce = _mm256_max_epu16(max0, max1);
            let cmp = _mm256_cmpgt_epi16(max_reduce, threshold_v); // fails if any value >= 32767 (shouldnt
                                                                   // happen)
            let none_above_threshold = _mm256_testz_si256(cmp, ones_v);
            if !(none_above_threshold == 1) {
                idxs[j] = i;
                j += 1;
            }
        }
    }
    let lb = upper_bounds.len() - (upper_bounds.len() % 64);
    for i in lb..upper_bounds.len() {
        // only need this once per block
        if upper_bounds[i] > threshold {
            idxs[j] = i;
            j += 1;
            break;
        }
    }
    (idxs, j)
}
