use super::lasso;
use bit_vec::BitVec;
use client::encode;
use ndarray::*;

fn to_a1(bv: &BitVec) -> Array1<f64> {
    Array1::from(
        bv.iter()
            .map(|bit| if bit { 1. } else { 0. })
            .collect::<Vec<f64>>(),
    )
}

// take Y and X then produce a model that fit for step 3
// take ndarray
// producce a fit result Y of X
// TODO fix to use lasso here, or at least something similar
// select candidate strings corresponding to non-zero coefficients.
fn linear_regression(x: Array2<f64>, y: Array1<f64>) -> Result<(), ErrorKind> {
    let encode_factory = encode::Factory::new(0.01);
    let encoded = encode_factory.process("a".into());
    let bv = string_to_bitvec(encoded);
    let y = estimate_y(&bv);

    // train the model of desigm matrix
    // the default bahavior of this is five candidate strings
    let matrix = create_design_matrix();
    let mut lasso_factory = lasso::LassoFactory::new(32);
    lasso_factory.train(matrix, &y[0]);
    Ok(())
}

fn create_design_matrix() -> Array2<f64> {
    let candidate_strings = ["a", "b", "c", "d", "e"];
    // what's 32?
    let mut design_matrix = Array2::<f64>::zeros((5, 32));

    for i in 0..candidate_strings.len() {
        let encode_factory = encode::Factory::new(0.01);
        let bf = encode_factory.initialize_bloom_to_bitarray(candidate_strings[i].into());
        let mut row = design_matrix.row_mut(i);
        for j in 0..row.len() {
            if bf.bits[i] {
                row[j] = 1.;
            } else {
                row[j] = 0.;
            }
        }
    }

    design_matrix
}

//Estimate the number of times each bit i within cohort
//j, tij , is truly set in B for each cohort. Given the
//number of times each bit i in cohort j, cij was set in
//a set of Nj reports, the estimate is given by
//Let Y be a vector of tij s, i  [1, k], j  [1, m].
fn estimate_y(bv: &BitVec) -> Vec<Array1<f64>> {
    let k = bv.len(); // size of filter let h = 1.; // number of hash functions
    let f = 0.5; // permanent response randomizer
    let p = 0.5; // temporary response randomizer
    let q = 0.5; // temporary response randomizer
    let m = 1.; // number of cohorts (groups of hash functions used by clients)

    // Cohorts implement different sets of h hash functions for their Bloom filters, thereby
    // reducing the chance of accidental collisions of two strings
    // across all of them.
    // what's the hash function?
    let cohorts = vec![vec![bv]]; // for now, just one cohort of only one client

    let init = || Array1::<f64>::zeros(k);

    let reported_counts_by_cohort = cohorts
        .iter()
        .map(|cohort| {
            cohort
                .iter()
                .fold(init(), |acc, curr_bv| acc + to_a1(curr_bv))
        })
        .collect::<Vec<Array1<f64>>>(); // TODO need to capture number of reports per cohort here too
    let n = 1.; // cheating here hard coding to 1 report per cohort

    // this is y, pass it along
    let estimated_true_counts_by_cohort = reported_counts_by_cohort
        .iter()
        .map(|counts| {
            counts.map(|count| (count - (p + 0.5 * f * q - 0.5 * f * p) * n) / ((1. - f) * (q - p)))
        })
        .collect::<Vec<Array1<f64>>>();

    estimated_true_counts_by_cohort
}

// likely something received from the web is
// a string, need to convert it to bitvec
fn string_to_bitvec(s: String) -> BitVec {
    // I think I'm sending ones and zeros. let's see
    let mut bv: BitVec = BitVec::new();
    for c in s.chars() {
        bv.push(c == '0');
    }
    bv
}

// Fit a regular least-squares regression using the selected
// variables to estimate counts, their standard errors and
// p-values.
fn least_square_regression() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_matrix() {
        create_design_matrix();
    }

    #[test]
    fn test_string_to_bitvec() {
        // give me a y!
        // translate
        let encode_factory = encode::Factory::new(0.01);
        let encoded = encode_factory.process("a".into());
        let bv = string_to_bitvec(encoded);
        assert_eq!(bv.len(), 921);
    }

    #[test]
    fn test_fit_model() {}

    /*
    We need to do regress(X, Y).L2_regularization(lambda)
    Where X and Y are data, lambda is a hyperparameter (the default in Numpy/scikit is 1 so start with that maybe)
    */
    #[test]
    fn test() -> Result<(), ErrorKind> {
        // let's say we have five candidate strings
        // and we received cohort from a particular report

        // test case
        let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
        let y = estimate_y(&bv);

        // create design matrix X of size km X M where M is the number of candidate strings
        // the matrix is 1 if bloom filter bits for each string for each cohort
        // in our case, we probably have k = 1, because we are lazy and only one cohort
        // let's say M = 5 for this test
        let mut x = Array2::<f64>::zeros((bv.len(), 5));
        for mut row in x.genrows_mut() {
            for i in 0..row.len() {
                if bv[i] {
                    row[i] = 1.;
                } else {
                    row[i] = 0.;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_regression() {
        // define some test data
        let data_y = array![0.3, 1.3, 0.7];
        let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];

        // linear_regression(data_x, data_y)
    }
}
