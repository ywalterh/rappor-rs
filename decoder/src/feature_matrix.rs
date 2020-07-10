use bit_vec::BitVec;
use client::encode;
use ndarray::Array2;

// This file is created to host logic of creating different design matrix
// in our unit test we use 'a - e', in rapport github repo they use v1 - v190
// should be able to support both easily
// matrix is actually also a sum of bits instead of what we used to think
// an array of bits aren't good enough for compare against, very easily we can get weird results
pub fn create_design_matrix(num_bloombits: usize, candidate_strings: &[&str]) -> Array2<f64> {
    //  32 is encode_factory.k ?
    let mut design_matrix = Array2::<f64>::zeros((num_bloombits, candidate_strings.len()));
    for i in 0..candidate_strings.len() {
        let encode_factory = encode::EncoderFactory::new(1);
        let irr = encode_factory.encode(1, candidate_strings[i].into());
        let bits = u32_to_bitvec(irr);
        assert_eq!(bits.len(), 32, "should be a size 32?");
        let mut col = design_matrix.column_mut(i);
        for j in 0..col.len() {
            if bits[j] {
                col[j] = 1.;
            } else {
                col[j] = 0.;
            }
        }

        // instead of having a matrix of a this
        // row.append(cohort * num_bloombits + (bit_to_set + 1))
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
        let matrix = super::create_design_matrix(16, candidate_strings);

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
