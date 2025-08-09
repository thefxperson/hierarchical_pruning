use half::f16;
use half::slice::HalfFloatSliceExt;
use std::arch::x86_64::*;
use std::mem;
use std::simd::*;

#[repr(C, align(64))]
struct AlignToSixtyFour([u8; 64]);
// this allocates a Vec<u8> where the raw data is aligned to 64 bytes
unsafe fn aligned_vec<'a>(n_bytes: usize) -> Vec<u16> {
    // Lazy math to ensure we always have enough.
    let n_units = (n_bytes / mem::size_of::<AlignToSixtyFour>()) + 1;

    let mut aligned: Vec<AlignToSixtyFour> = Vec::with_capacity(n_units);

    let ptr = aligned.as_mut_ptr();
    let cap_units = aligned.capacity();

    mem::forget(aligned);

    Vec::from_raw_parts(
        ptr as *mut u16,
        // set len == cap because we zero it immediately after call
        (cap_units * mem::size_of::<AlignToSixtyFour>()) / 2,
        (cap_units * mem::size_of::<AlignToSixtyFour>()) / 2,
    )
}

pub fn compute_upper_bounds_raw(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
) -> Vec<u16> {
    let mut upper_bounds: Vec<u16> = vec![0; vector_len];

    // Iterate over each vector in scores and add its elements to the result
    for (&vec, &weight) in query_ranges.iter().zip(query_weights.iter()) {
        for (i, &score) in vec.iter().enumerate() {
            let multiplied = score as u16 * weight as u16;
            upper_bounds[i] = upper_bounds[i].saturating_add(multiplied);
        }
    }
    upper_bounds
}

pub fn compute_asc_raw<'a>(
    query_range_maxs: &'a [&[u8]],
    query_range_avgs: &'a [&[u8]],
    query_weights: &'a [u8],
    vector_len: usize,
) -> (Vec<u16>, Vec<f32>) {
    // return f32 to save high latency conversion to f16
    // let vector_len: usize = query_ranges[0].len();
    let mut upper_bounds: Vec<u16> = vec![0; vector_len];
    let mut averages: Vec<f32> = vec![0_f32; vector_len];

    // Iterate over each vector in scores and add its elements to the result
    for ((&max_vec, &avg_vec), &weight) in query_range_maxs
        .iter()
        .zip(query_range_avgs.iter())
        .zip(query_weights.iter())
    {
        for (i, &score) in max_vec.iter().enumerate() {
            let multiplied = score as u16 * weight as u16;
            upper_bounds[i] = upper_bounds[i].saturating_add(multiplied);
        }
        // need to force a slice conversion to fully use SIMD with half library
        for i in (0..avg_vec.len() - 16).step_by(16) {
            let bytes = [
                f16::from_ne_bytes([avg_vec[i], avg_vec[i + 1]]),
                f16::from_ne_bytes([avg_vec[i + 2], avg_vec[i + 3]]),
                f16::from_ne_bytes([avg_vec[i + 4], avg_vec[i + 5]]),
                f16::from_ne_bytes([avg_vec[i + 6], avg_vec[i + 7]]),
                f16::from_ne_bytes([avg_vec[i + 8], avg_vec[i + 9]]),
                f16::from_ne_bytes([avg_vec[i + 10], avg_vec[i + 11]]),
                f16::from_ne_bytes([avg_vec[i + 12], avg_vec[i + 13]]),
                f16::from_ne_bytes([avg_vec[i + 14], avg_vec[i + 15]]),
            ];
            bytes.convert_to_f32_slice(&mut averages[i / 2..i / 2 + 8]);
        }
        // and the leftovers
        for i in (avg_vec.len() - (avg_vec.len() % 16)..avg_vec.len()).step_by(2) {
            let bytes = [avg_vec[i], avg_vec[i + 1]];
            averages[i / 2] += f16::from_ne_bytes(bytes).to_f32() * weight as f32;
        }
    }
    // try to force compuler to generate faster code
    let dividend: f32 = 1.0 / query_weights.len() as f32;
    for i in 0..averages.len() {
        averages[i] *= dividend;
    }

    (upper_bounds, averages)
}

pub fn compute_filtered_upper_bounds_raw(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    stride: usize,
    filter: &Vec<usize>,
    prefetch_dist: usize,
) -> Vec<u16> {
    // let vector_len: usize = query_ranges[0].len();
    let mut upper_bounds: Vec<u16> = vec![0; vector_len];

    let stride = 32;
    let DIST = prefetch_dist;
    unsafe {
        _mm_prefetch(
            query_weights.get_unchecked(0) as *const _ as *const i8,
            _MM_HINT_T1,
        );
        _mm_prefetch(
            filter.get_unchecked(DIST) as *const _ as *const i8,
            _MM_HINT_T0,
        );
    }

    for (&vec, &weight) in query_ranges.iter().zip(query_weights.iter()) {
        for i in 0..DIST {
            unsafe {
                let next_filter_idx0 = filter.get_unchecked(i) * stride;
                let next_filter_idx1 = filter.get_unchecked(i + 1) * stride;
                _mm_prefetch(
                    vec.get_unchecked(next_filter_idx0) as *const _ as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    vec.get_unchecked(next_filter_idx1) as *const _ as *const i8,
                    _MM_HINT_NTA,
                );
            }
        }
        unsafe {
            _mm_prefetch(
                filter.get_unchecked(2 * DIST) as *const _ as *const i8,
                _MM_HINT_T0,
            );
        }
        let query_weight = u16x16::splat(weight as u16);
        for i in 0..(filter.len() - 1) / 2 {
            unsafe {
                // calc index
                let filter_idx0 = filter.get_unchecked(2 * i) * stride;
                let filter_idx1 = filter.get_unchecked(2 * i + 1) * stride;
                let next_filter_idx0 = filter.get_unchecked(2 * i + DIST) * stride;
                let next_filter_idx1 = filter.get_unchecked(2 * i + DIST + 1) * stride;
                _mm_prefetch(
                    vec.get_unchecked(next_filter_idx0) as *const _ as *const i8,
                    _MM_HINT_NTA,
                );
                _mm_prefetch(
                    vec.get_unchecked(next_filter_idx1) as *const _ as *const i8,
                    _MM_HINT_NTA,
                );

                // deal with rust type bullshit
                // these won't get compiled to ASM, will end up as vmovzxbw
                let block_weights0 =
                    u8x16::from_slice(&vec.get_unchecked(filter_idx0..filter_idx0 + 16));
                let block_weights1 =
                    u8x16::from_slice(&vec.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
                let block_weights2 =
                    u8x16::from_slice(&vec.get_unchecked(filter_idx1..filter_idx1 + 16));
                let block_weights3 =
                    u8x16::from_slice(&vec.get_unchecked(filter_idx1 + 16..filter_idx1 + 32));
                // likewise, won't get compiled
                let current_accum0 =
                    u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0..filter_idx0 + 16));
                let current_accum1 = u16x16::from_slice(
                    &upper_bounds.get_unchecked(filter_idx0 + 16..filter_idx0 + 32),
                );
                let current_accum2 =
                    u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx1..filter_idx1 + 16));
                let current_accum3 = u16x16::from_slice(
                    &upper_bounds.get_unchecked(filter_idx1 + 16..filter_idx1 + 32),
                );
                // vpmovzxbw
                let b0 = _mm256_cvtepu8_epi16(__m128i::from(block_weights0));
                let b1 = _mm256_cvtepu8_epi16(__m128i::from(block_weights1));
                let b2 = _mm256_cvtepu8_epi16(__m128i::from(block_weights2));
                let b3 = _mm256_cvtepu8_epi16(__m128i::from(block_weights3));
                // vpmullw
                let res0 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b0));
                let res1 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b1));
                let res2 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b2));
                let res3 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b3));
                // vpaddusw
                let add_res0 = _mm256_adds_epu16(__m256i::from(current_accum0), res0);
                let add_res1 = _mm256_adds_epu16(__m256i::from(current_accum1), res1);
                let add_res2 = _mm256_adds_epu16(__m256i::from(current_accum2), res2);
                let add_res3 = _mm256_adds_epu16(__m256i::from(current_accum3), res3);
                // intersperse prefetch for better performance
                _mm_prefetch(
                    filter.get_unchecked(2 * i + 2) as *const _ as *const i8,
                    _MM_HINT_NTA,
                );
                // vmovdqu
                u16x16::from(add_res0).copy_to_slice(
                    &mut upper_bounds.get_unchecked_mut(filter_idx0..filter_idx0 + 16),
                );
                u16x16::from(add_res1).copy_to_slice(
                    &mut upper_bounds.get_unchecked_mut(filter_idx0 + 16..filter_idx0 + 32),
                );
                u16x16::from(add_res2).copy_to_slice(
                    &mut upper_bounds.get_unchecked_mut(filter_idx1..filter_idx1 + 16),
                );
                u16x16::from(add_res3).copy_to_slice(
                    &mut upper_bounds.get_unchecked_mut(filter_idx1 + 16..filter_idx1 + 32),
                );
            }
        }
        unsafe {
            _mm_prefetch(
                filter.get_unchecked(0) as *const _ as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                filter.get_unchecked(DIST) as *const _ as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                query_weights.get_unchecked(0) as *const _ as *const i8, // this 0 should be the
                // right index.
                _MM_HINT_T0,
            );
        }
        // extras
        for idx in (filter[filter.len() - 1] * stride)..vector_len {
            let multiplied = vec[idx] as u16 * weight as u16;
            upper_bounds[idx] = upper_bounds[idx].saturating_add(multiplied);
        }
    }

    upper_bounds
}

pub fn optimized_compute_filtered_upper_bounds_raw<'a>(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    stride: usize,
    filter: &Vec<u32>,
    prefetch_dist: usize,
) -> Vec<u16> {
    let upper_bounds = match stride {
        16 => ocfubr_16(
            query_ranges,
            query_weights,
            vector_len,
            filter,
            prefetch_dist,
        ),
        32 => ocfubr_32(
            query_ranges,
            query_weights,
            vector_len,
            filter,
            prefetch_dist,
        ),
        64 => ocfubr_64(
            query_ranges,
            query_weights,
            vector_len,
            filter,
            prefetch_dist,
        ),
        128 => ocfubr_128(
            query_ranges,
            query_weights,
            vector_len,
            filter,
            prefetch_dist,
        ),
        256 => ocfubr_256(
            query_ranges,
            query_weights,
            vector_len,
            filter,
            prefetch_dist,
        ),
        _ => panic!("Stride not supported"),
    };
    upper_bounds
}

pub fn ocfubr_16<'a>(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    filter: &Vec<u32>,
    prefetch_dist: usize,
) -> Vec<u16> {
    let DIST = prefetch_dist;
    let stride = 16;
    let mut upper_bounds: Vec<u16>;
    unsafe {
        upper_bounds = aligned_vec(2 * vector_len); // function takes # bytes as argument
        upper_bounds.fill(0);
    }

    let mut filter_idx0;
    let n_blocks = filter.len() - 1;

    let prefetch_idx_offset = (DIST / query_ranges.len()) + 1;
    let DIST_OFFSET = DIST % query_ranges.len();
    // main loop
    for i in 0..n_blocks {
        unsafe {
            // calc index
            filter_idx0 = filter.get_unchecked(i) * stride;
        }
        for j in 0..(query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop16(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
            }
        }

        // we passed n_terms, so switch to prefetching next_filter_idx
        for j in (query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize
            ..query_ranges.len()
        {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop16(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
            }
        }
    }

    // extras
    // last filter value
    for j in 0..query_ranges.len() {
        let weight = query_weights[j];
        let query_weight = u16x16::splat(weight as u16);
        unsafe {
            cl_filtered_inner_loop32(
                query_ranges.get_unchecked(j),
                query_weight,
                &mut upper_bounds,
                filter[filter.len() - 1] as usize,
            );
        }
    }
    let lower_bound = vector_len as u32 - (vector_len as u32 % 64);
    for idx in lower_bound..vector_len as u32 {
        for j in 0..query_ranges.len() {
            let multiplied = query_ranges[j][idx as usize] as u16 * query_weights[j] as u16;
            upper_bounds[idx as usize] = upper_bounds[idx as usize].saturating_add(multiplied);
        }
    }

    upper_bounds
}
pub fn ocfubr_32<'a>(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    filter: &Vec<u32>,
    prefetch_dist: usize,
) -> Vec<u16> {
    let DIST = prefetch_dist;
    let stride = 32;
    let mut upper_bounds: Vec<u16>;
    unsafe {
        upper_bounds = aligned_vec(2 * vector_len); // function takes # bytes as argument
        upper_bounds.fill(0);
    }

    let mut filter_idx0;
    let n_blocks = filter.len() - 1;

    let prefetch_idx_offset = (DIST / query_ranges.len()) + 1;
    let DIST_OFFSET = DIST % query_ranges.len();
    // main loop
    for i in 0..n_blocks {
        unsafe {
            // calc index
            filter_idx0 = filter.get_unchecked(i) * stride;
        }
        for j in 0..(query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop32(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
            }
        }

        // we passed n_terms, so switch to prefetching next_filter_idx
        for j in (query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize
            ..query_ranges.len()
        {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop32(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
            }
        }
    }

    // extras
    // last filter value
    for j in 0..query_ranges.len() {
        let weight = query_weights[j];
        let query_weight = u16x16::splat(weight as u16);
        unsafe {
            cl_filtered_inner_loop32(
                query_ranges.get_unchecked(j),
                query_weight,
                &mut upper_bounds,
                filter[filter.len() - 1] as usize,
            );
        }
    }
    let lower_bound = vector_len as u32 - (vector_len as u32 % 64);
    for idx in lower_bound..vector_len as u32 {
        for j in 0..query_ranges.len() {
            let multiplied = query_ranges[j][idx as usize] as u16 * query_weights[j] as u16;
            upper_bounds[idx as usize] = upper_bounds[idx as usize].saturating_add(multiplied);
        }
    }

    upper_bounds
}
pub fn ocfubr_64<'a>(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    filter: &Vec<u32>,
    prefetch_dist: usize,
) -> Vec<u16> {
    let DIST = prefetch_dist;
    let stride = 64;
    let mut upper_bounds: Vec<u16>;
    unsafe {
        upper_bounds = aligned_vec(2 * vector_len); // function takes # bytes as argument
        upper_bounds.fill(0);
    }

    let mut filter_idx0;
    let mut this_filter_idx0;
    let mut next_filter_idx0;
    let n_blocks = filter.len() - 1;

    let prefetch_idx_offset = (DIST / query_ranges.len()) + 1;
    let DIST_OFFSET = DIST % query_ranges.len();
    // main loop
    for i in 0..n_blocks {
        unsafe {
            // calc index
            filter_idx0 = filter.get_unchecked(i) * stride;
            this_filter_idx0 = filter.get_unchecked(i + prefetch_idx_offset - 1) * stride;
            next_filter_idx0 = filter.get_unchecked(i + prefetch_idx_offset) * stride;
        }
        for j in 0..(query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
                _mm_prefetch(
                    query_ranges
                        .get_unchecked(j + DIST)
                        .as_ptr()
                        .offset(this_filter_idx0 as isize) as *const _
                        as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        // we passed n_terms, so switch to prefetching next_filter_idx
        for j in (query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize
            ..query_ranges.len()
        {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
                _mm_prefetch(
                    query_ranges
                        .get_unchecked((j + DIST) % query_ranges.len())
                        .as_ptr()
                        .offset(next_filter_idx0 as isize) as *const _
                        as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
    }

    // extras
    // last filter value
    for j in 0..query_ranges.len() {
        let weight = query_weights[j];
        let query_weight = u16x16::splat(weight as u16);
        unsafe {
            cl_filtered_inner_loop(
                query_ranges.get_unchecked(j),
                query_weight,
                &mut upper_bounds,
                filter[filter.len() - 1] as usize,
            );
        }
    }
    let lower_bound = vector_len as u32 - (vector_len as u32 % 64);
    for idx in lower_bound..vector_len as u32 {
        for j in 0..query_ranges.len() {
            let multiplied = query_ranges[j][idx as usize] as u16 * query_weights[j] as u16;
            upper_bounds[idx as usize] = upper_bounds[idx as usize].saturating_add(multiplied);
        }
    }

    upper_bounds
}

pub fn ocfubr_128<'a>(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    filter: &Vec<u32>,
    prefetch_dist: usize,
) -> Vec<u16> {
    let DIST = prefetch_dist;
    let stride = 128;
    let mut upper_bounds: Vec<u16>;
    unsafe {
        upper_bounds = aligned_vec(2 * vector_len); // function takes # bytes as argument
        upper_bounds.fill(0);
    }

    let mut filter_idx0;
    let mut this_filter_idx0;
    let mut next_filter_idx0;
    let n_blocks = filter.len() - 1;

    let prefetch_idx_offset = (DIST / query_ranges.len()) + 1;
    let DIST_OFFSET = DIST % query_ranges.len();
    // main loop
    for i in 0..n_blocks {
        unsafe {
            // calc index
            filter_idx0 = filter.get_unchecked(i) * stride;
            this_filter_idx0 = filter.get_unchecked(i + prefetch_idx_offset - 1) * stride;
            next_filter_idx0 = filter.get_unchecked(i + prefetch_idx_offset) * stride;
        }
        for j in 0..(query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize + 64,
                );
                _mm_prefetch(
                    query_ranges
                        .get_unchecked(j + DIST)
                        .as_ptr()
                        .offset(this_filter_idx0 as isize) as *const _
                        as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    query_ranges
                        .get_unchecked(j + DIST)
                        .as_ptr()
                        .offset((this_filter_idx0 + 64) as isize) as *const _
                        as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        // we passed n_terms, so switch to prefetching next_filter_idx
        for j in (query_ranges.len() as isize - DIST_OFFSET as isize - 1).max(0) as usize
            ..query_ranges.len()
        {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize + 64,
                );
                _mm_prefetch(
                    query_ranges
                        .get_unchecked((j + DIST) % query_ranges.len())
                        .as_ptr()
                        .offset(next_filter_idx0 as isize) as *const _
                        as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    query_ranges
                        .get_unchecked((j + DIST) % query_ranges.len())
                        .as_ptr()
                        .offset((next_filter_idx0 + 64) as isize) as *const _
                        as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
    }

    // extras
    // last filter value
    for j in 0..query_ranges.len() {
        let weight = query_weights[j];
        let query_weight = u16x16::splat(weight as u16);
        unsafe {
            cl_filtered_inner_loop(
                query_ranges.get_unchecked(j),
                query_weight,
                &mut upper_bounds,
                filter[filter.len() - 1] as usize,
            );
            cl_filtered_inner_loop(
                query_ranges.get_unchecked(j),
                query_weight,
                &mut upper_bounds,
                filter[filter.len() - 1] as usize + 64,
            );
        }
    }
    let lower_bound = vector_len as u32 - (vector_len as u32 % 64);
    for idx in lower_bound..vector_len as u32 {
        for j in 0..query_ranges.len() {
            let multiplied = query_ranges[j][idx as usize] as u16 * query_weights[j] as u16;
            upper_bounds[idx as usize] = upper_bounds[idx as usize].saturating_add(multiplied);
        }
    }

    upper_bounds
}

pub fn ocfubr_256<'a>(
    query_ranges: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
    filter: &Vec<u32>,
    prefetch_dist: usize,
) -> Vec<u16> {
    let DIST = prefetch_dist;
    let stride = 256;
    let mut upper_bounds: Vec<u16>;
    unsafe {
        upper_bounds = aligned_vec(2 * vector_len); // function takes # bytes as argument
        upper_bounds.fill(0);
    }

    let mut filter_idx0;
    let mut query_weight;
    // main loop
    for j in 0..query_ranges.len() {
        unsafe {
            query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
        }
        for i in 0..filter.len() {
            unsafe {
                // calc index
                filter_idx0 = filter.get_unchecked(i) * stride;
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize,
                );
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize + 64,
                );
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize + 128,
                );
                cl_filtered_inner_loop(
                    query_ranges.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    filter_idx0 as usize + 192,
                );
            }
        }
    }
    let lower_bound = vector_len as u32 - (vector_len as u32 % stride);
    for idx in lower_bound..(vector_len - 1) as u32 {
        for j in 0..query_ranges.len() {
            let multiplied = query_ranges[j][idx as usize] as u16 * query_weights[j] as u16;
            upper_bounds[idx as usize] = upper_bounds[idx as usize].saturating_add(multiplied);
        }
    }

    upper_bounds
}

#[inline]
pub unsafe fn filtered_inner_loop(
    vec: &[u8],
    query_weight: u16x16,
    upper_bounds: &mut Vec<u16>,
    filter_idx0: usize,
    filter_idx1: usize,
) {
    unsafe {
        // deal with rust type stuff
        // these won't get compiled to ASM, will end up as vmovzxbw
        let block_weights0 = u8x16::from_slice(&vec.get_unchecked(filter_idx0..filter_idx0 + 16));
        let block_weights1 =
            u8x16::from_slice(&vec.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
        let block_weights2 = u8x16::from_slice(&vec.get_unchecked(filter_idx1..filter_idx1 + 16));
        let block_weights3 =
            u8x16::from_slice(&vec.get_unchecked(filter_idx1 + 16..filter_idx1 + 32));
        // likewise, won't get compiled
        let current_accum0 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0..filter_idx0 + 16));
        let current_accum1 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
        let current_accum2 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx1..filter_idx1 + 16));
        let current_accum3 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx1 + 16..filter_idx1 + 32));
        // vpmovzxbw
        let b0 = _mm256_cvtepu8_epi16(__m128i::from(block_weights0));
        let b1 = _mm256_cvtepu8_epi16(__m128i::from(block_weights1));
        let b2 = _mm256_cvtepu8_epi16(__m128i::from(block_weights2));
        let b3 = _mm256_cvtepu8_epi16(__m128i::from(block_weights3));
        // vpmullw
        let res0 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b0));
        let res1 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b1));
        let res2 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b2));
        let res3 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b3));
        // vpaddusw
        let add_res0 = _mm256_adds_epu16(__m256i::from(current_accum0), res0);
        let add_res1 = _mm256_adds_epu16(__m256i::from(current_accum1), res1);
        let add_res2 = _mm256_adds_epu16(__m256i::from(current_accum2), res2);
        let add_res3 = _mm256_adds_epu16(__m256i::from(current_accum3), res3);
        // vmovdqu
        u16x16::from(add_res0)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0..filter_idx0 + 16));
        u16x16::from(add_res1)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0 + 16..filter_idx0 + 32));
        u16x16::from(add_res2)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx1..filter_idx1 + 16));
        u16x16::from(add_res3)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx1 + 16..filter_idx1 + 32));
    }
}
#[inline]
pub unsafe fn cl_filtered_inner_loop(
    vec: &[u8],
    query_weight: u16x16,
    upper_bounds: &mut Vec<u16>,
    filter_idx0: usize,
) {
    unsafe {
        // deal with rust type bullshit
        // these won't get compiled to ASM, will end up as vmovzxbw
        let block_weights0 = u8x16::from_slice(&vec.get_unchecked(filter_idx0..filter_idx0 + 16));
        let block_weights1 =
            u8x16::from_slice(&vec.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
        let block_weights2 =
            u8x16::from_slice(&vec.get_unchecked(filter_idx0 + 32..filter_idx0 + 48));
        let block_weights3 =
            u8x16::from_slice(&vec.get_unchecked(filter_idx0 + 48..filter_idx0 + 64));
        // likewise, won't get compiled
        let current_accum0 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0..filter_idx0 + 16));
        let current_accum1 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
        let current_accum2 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0 + 32..filter_idx0 + 48));
        let current_accum3 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0 + 48..filter_idx0 + 64));
        // vpmovzxbw
        let b0 = _mm256_cvtepu8_epi16(__m128i::from(block_weights0));
        let b1 = _mm256_cvtepu8_epi16(__m128i::from(block_weights1));
        let b2 = _mm256_cvtepu8_epi16(__m128i::from(block_weights2));
        let b3 = _mm256_cvtepu8_epi16(__m128i::from(block_weights3));
        // vpmullw
        let res0 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b0));
        let res1 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b1));
        let res2 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b2));
        let res3 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b3));
        // vpaddusw
        let add_res0 = _mm256_adds_epu16(__m256i::from(current_accum0), res0);
        let add_res1 = _mm256_adds_epu16(__m256i::from(current_accum1), res1);
        let add_res2 = _mm256_adds_epu16(__m256i::from(current_accum2), res2);
        let add_res3 = _mm256_adds_epu16(__m256i::from(current_accum3), res3);
        // vmovdqu
        u16x16::from(add_res0)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0..filter_idx0 + 16));
        u16x16::from(add_res1)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0 + 16..filter_idx0 + 32));
        u16x16::from(add_res2)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0 + 32..filter_idx0 + 48));
        u16x16::from(add_res3)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0 + 48..filter_idx0 + 64));
    }
}

#[inline]
pub unsafe fn cl_filtered_inner_loop32(
    vec: &[u8],
    query_weight: u16x16,
    upper_bounds: &mut Vec<u16>,
    filter_idx0: usize,
) {
    unsafe {
        // deal with rust type stuff
        // these won't get compiled to ASM, will end up as vmovzxbw
        let block_weights0 = u8x16::from_slice(&vec.get_unchecked(filter_idx0..filter_idx0 + 16));
        let block_weights1 =
            u8x16::from_slice(&vec.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
        // likewise, won't get compiled
        let current_accum0 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0..filter_idx0 + 16));
        let current_accum1 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0 + 16..filter_idx0 + 32));
        // vpmovzxbw
        let b0 = _mm256_cvtepu8_epi16(__m128i::from(block_weights0));
        let b1 = _mm256_cvtepu8_epi16(__m128i::from(block_weights1));
        // vpmullw
        let res0 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b0));
        let res1 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b1));
        // vpaddusw
        let add_res0 = _mm256_adds_epu16(__m256i::from(current_accum0), res0);
        let add_res1 = _mm256_adds_epu16(__m256i::from(current_accum1), res1);
        // vmovdqu
        u16x16::from(add_res0)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0..filter_idx0 + 16));
        u16x16::from(add_res1)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0 + 16..filter_idx0 + 32));
    }
}
#[inline]
pub unsafe fn cl_filtered_inner_loop16(
    vec: &[u8],
    query_weight: u16x16,
    upper_bounds: &mut Vec<u16>,
    filter_idx0: usize,
) {
    unsafe {
        // deal with rust type stuff
        // these won't get compiled to ASM, will end up as vmovzxbw
        let block_weights0 = u8x16::from_slice(&vec.get_unchecked(filter_idx0..filter_idx0 + 16));
        // likewise, won't get compiled
        let current_accum0 =
            u16x16::from_slice(&upper_bounds.get_unchecked(filter_idx0..filter_idx0 + 16));
        // vpmovzxbw
        let b0 = _mm256_cvtepu8_epi16(__m128i::from(block_weights0));
        // vpmullw
        let res0 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b0));
        // vpaddusw
        let add_res0 = _mm256_adds_epu16(__m256i::from(current_accum0), res0);
        // vmovdqu
        u16x16::from(add_res0)
            .copy_to_slice(&mut upper_bounds.get_unchecked_mut(filter_idx0..filter_idx0 + 16));
    }
}

#[inline]
pub fn compute_upper_bounds(
    query_ranges: &[&[crate::index::posting_list::CompressedBlock]],
    query_weights: &[u8],
    vector_len: usize,
) -> Vec<u16> {
    let mut upper_bounds: Vec<u16> = vec![0; vector_len];

    // Iterate over each vector in scores and add its elements to the result
    for (&vec, &weight) in query_ranges.iter().zip(query_weights.iter()) {
        for (bid, block) in vec.iter().enumerate() {
            for &(offset, score) in &block.max_scores {
                let multiplied = score as u16 * weight as u16;
                upper_bounds[bid * 256 + offset] =
                    upper_bounds[bid * 256 + offset].saturating_add(multiplied);
            }
        }
    }
    upper_bounds
}

#[inline]
pub fn compute_asc_raw_intrinsic(
    query_range_maxs: &[&[u8]],
    query_range_avgs: &[&[u8]],
    query_weights: &[u8],
    vector_len: usize,
) -> (Vec<u16>, Vec<f32>) {
    let mut upper_bounds: Vec<u16> = vec![0; vector_len];
    let mut averages: Vec<f32> = vec![0_f32; vector_len];

    let stride = 32;
    let n_blocks = query_range_maxs[0].len() / stride;
    for i in 0..n_blocks - 1 {
        for j in 0..query_weights.len() {
            unsafe {
                let query_weight = u16x16::splat(*query_weights.get_unchecked(j) as u16);
                unfiltered_inner_loop(
                    query_range_maxs.get_unchecked(j),
                    query_range_avgs.get_unchecked(j),
                    query_weight,
                    &mut upper_bounds,
                    &mut averages,
                    i * stride,
                    i * 2 * stride,
                );
            }
        }
    }
    for i in (n_blocks - 1) * stride..query_range_maxs[0].len() {
        for j in 0..query_weights.len() {
            let weight = query_weights[j];
            let mult = query_range_maxs[j][i] as u16 * weight as u16;
            upper_bounds[i] = upper_bounds[i].saturating_add(mult);

            let bytes = [query_range_avgs[j][2 * i], query_range_avgs[j][2 * i + 1]];
            averages[i] += f16::to_f32(f16::from_ne_bytes(bytes)) * weight as f32;
        }
    }

    (upper_bounds, averages)
}

#[inline]
pub unsafe fn unfiltered_inner_loop(
    max_vec: &[u8],
    avg_vec: &[u8],
    query_weight: u16x16,
    upper_bounds: &mut Vec<u16>,
    averages: &mut Vec<f32>,
    i: usize,
    j: usize, // j is for raw f16s
) {
    unsafe {
        // deal with rust type stuff
        // these won't get compiled to ASM, will end up as vmovzxbw
        let block_weights0 = u8x16::from_slice(&max_vec.get_unchecked(i..i + 16));
        let block_weights1 = u8x16::from_slice(&max_vec.get_unchecked(i + 16..i + 32));
        // likewise, won't get compiled
        let current_accum0 = u16x16::from_slice(&upper_bounds.get_unchecked(i..i + 16));
        let current_accum1 = u16x16::from_slice(&upper_bounds.get_unchecked(i + 16..i + 32));
        // now floating point stuff
        let block_avgs0 = u8x16::from_slice(&avg_vec.get_unchecked(j..j + 16));
        let block_avgs1 = u8x16::from_slice(&avg_vec.get_unchecked(j + 16..j + 32));
        let block_avgs2 = u8x16::from_slice(&avg_vec.get_unchecked(j + 32..j + 48));
        let block_avgs3 = u8x16::from_slice(&avg_vec.get_unchecked(j + 48..j + 64));
        let average_accum0 = f32x8::from_slice(&averages.get_unchecked(i..i + 8));
        let average_accum1 = f32x8::from_slice(&averages.get_unchecked(i + 8..i + 16));
        let average_accum2 = f32x8::from_slice(&averages.get_unchecked(i + 16..i + 24));
        let average_accum3 = f32x8::from_slice(&averages.get_unchecked(i + 24..i + 32));
        // vpmovzxbw
        let b0 = _mm256_cvtepu8_epi16(__m128i::from(block_weights0));
        let b1 = _mm256_cvtepu8_epi16(__m128i::from(block_weights1));
        // vpmullw
        let res0 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b0));
        let res1 = _mm256_mullo_epi16(__m256i::from(query_weight), __m256i::from(b1));
        // vpaddusw
        let add_res0 = _mm256_adds_epu16(__m256i::from(current_accum0), res0);
        let add_res1 = _mm256_adds_epu16(__m256i::from(current_accum1), res1);
        // vmovdqu
        u16x16::from(add_res0).copy_to_slice(&mut upper_bounds.get_unchecked_mut(i..i + 16));
        u16x16::from(add_res1).copy_to_slice(&mut upper_bounds.get_unchecked_mut(i + 16..i + 32));
        // now floating point stuff, and let's see how the compiler optimizes and reorders all this jazz
        // vcvtph2ps
        let f0 = _mm256_cvtph_ps(__m128i::from(block_avgs0));
        let f1 = _mm256_cvtph_ps(__m128i::from(block_avgs1));
        let f2 = _mm256_cvtph_ps(__m128i::from(block_avgs2));
        let f3 = _mm256_cvtph_ps(__m128i::from(block_avgs3));
        // need to change query weights into a float too
        let qw_32 = _mm256_cvtepi16_epi32(__m128i::from(u16x8::from_slice(
            &query_weight.as_array()[0..8],
        )));
        let qw_f = _mm256_cvtepi32_ps(qw_32);
        // vfmadd132ps
        let fadd_res0 = _mm256_fmadd_ps(f0, qw_f, __m256::from(average_accum0));
        let fadd_res1 = _mm256_fmadd_ps(f1, qw_f, __m256::from(average_accum1));
        let fadd_res2 = _mm256_fmadd_ps(f2, qw_f, __m256::from(average_accum2));
        let fadd_res3 = _mm256_fmadd_ps(f3, qw_f, __m256::from(average_accum3));
        // vmovups
        f32x8::from(fadd_res0).copy_to_slice(&mut averages.get_unchecked_mut(i..i + 8));
        f32x8::from(fadd_res1).copy_to_slice(&mut averages.get_unchecked_mut(i + 8..i + 16));
        f32x8::from(fadd_res2).copy_to_slice(&mut averages.get_unchecked_mut(i + 16..i + 24));
        f32x8::from(fadd_res3).copy_to_slice(&mut averages.get_unchecked_mut(i + 24..i + 32));
    }
}
