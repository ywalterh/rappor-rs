use bit_vec::BitVec;
use client::encode;
use ndarray::Array2;

// This file is created to host logic of creating different design matrix
// in our unit test we use 'a - e', in rapport github repo they use v1 - v190
// should be able to support both easily
// matrix is actually also a sum of bits instead of what we used to think
// an array of bits aren't good enough for compare against, very easily we can get weird results
//
// Come up with candidate strings and see the distribution matches
//
//
// Cohorts implement different sets of h hash functions for their Bloom filters
// therefore here we are using 2 as number of hashes for the same cohort
pub fn create_design_matrix(
    num_bloombits: usize,
    num_cohorts: usize,
    candidate_strings: &[&str],
) -> Array2<f64> {
    //  32 is encode_factory.k ?
    let mut design_matrix = Array2::<f64>::zeros((num_cohorts, candidate_strings.len()));
    for i in 0..candidate_strings.len() {
        let word: String = candidate_strings[i].into();
        let mut col = design_matrix.column_mut(i);
        for cohort in 0..num_cohorts {
            //for each cohort, generate a count
            let encode_factory = encode::EncoderFactory::new(1);
            // to make it easier for here, use num of hashes to be 1
            // in real word results, we could increase the number
            let bits = encode_factory.get_bloom_bits(cohort as u32, &word);
            // the number of bits returned are the number of hashes. how do we feed them as X??
            // this is currently failing until we understand what's going on
            assert_eq!(
                bits.len(),
                encode_factory.num_hashes,
                "the bits array should be the same size of num of hashes"
            );

            // instead of having a matrix of a this
            // row.append(cohort * num_bloombits + (bit_to_set + 1)) but why?
            // this is the actual map used in decoding
            // not the one we thought it would be
            for jj in 0..bits.len() {
                col[cohort] = (cohort * num_bloombits + (bits[jj] as usize + 1)) as f64;
            }

            // design matrix should be km X M ---> number of bit is 16 in my case, and number of
            // cohorts are 60 and M is the number of candidate strings are .. 60 or so
            // in the R code in rappor repo
            //
            //
            //  # stretch cohorts to bits
            //  filter_bits <- as.vector(matrix(1:nrow(map), ncol = m)[,filter_cohorts, drop = FALSE])
            //  map_filtered <- map[filter_bits, , drop = FALSE]
            //  es <- EstimateBloomCounts(params, counts)
            //
            //
            //
        }
    }

    design_matrix
}

pub const test_candidate_strings: &'static [&'static str] = &["a", "b", "c", "d", "e"];

pub fn u32_to_bitvec(input: u32) -> BitVec {
    let mut bv: BitVec = BitVec::new();
    // assume 32 bit
    for i in (0..32).rev() {
        let k = input >> i;
        bv.push((k & 1) == 1);
    }
    bv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_matrix() {
        let candidate_strings = super::test_candidate_strings;
        let matrix = super::create_design_matrix(16, 32, candidate_strings);

        // uncomment if want sanity check on matrix
        // println! {"resuling matrix is {}", matrix};
        assert_eq!(matrix.len(), 16 * 5);
        // make sure there's at least some non-negative stuff..
        let mut all_zero = true;
        for row in matrix.genrows() {
            for i in 0..row.len() {
                if row[i] != 0. {
                    all_zero = false;
                    break;
                }
            }
        }

        assert!(!all_zero)
    }

    #[test]
    fn test_u32_to_bitvec() {
        let input = 1;
        let bv = u32_to_bitvec(input);
        assert!(bv.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, true
        ]));

        let input = 0;
        let bv = u32_to_bitvec(input);
        assert!(bv.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false
        ]));

        let input = 17;
        let bv = u32_to_bitvec(input);

        assert_eq!(bv.len(), 32);
        assert!(bv.eq_vec(&[
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false, false, false,
            false, false, false, true, false, false, false, true
        ]));
    }
}
