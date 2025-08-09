#![recursion_limit = "1024"]
#![feature(pointer_is_aligned_to)]
#![feature(portable_simd)]

pub mod ciff;
pub mod index;
mod proto;
pub mod query;
pub mod search;
pub mod util;

pub use ciff::CiffToBmp;
