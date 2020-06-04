use bit_vec::BitVec;
use ndarray::array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::*;

#[cfg(test)]
mod tests {
    use super::*;
    fn to_a1(bv: &BitVec) -> Array1<f32> {
        Array1::from(
            bv.iter()
                .map(|bit| if bit { 1. } else { 0. })
                .collect::<Vec<f32>>(),
        )
    }

    // take Y and X then produce a model that fit for step 3
    // take ndarray
    // producce a fit result Y of X
    // TODO fix to use lasso here, or at least something similar
    // select candidate strings corresponding to non-zero coefficients.
    fn linear_regression(x: Array2<f32>, y: Array1<f32>) -> Result<(), ErrorKind> {
        Ok(())
    }

    //Estimate the number of times each bit i within cohort
    //j, tij , is truly set in B for each cohort. Given the
    //number of times each bit i in cohort j, cij was set in
    //a set of Nj reports, the estimate is given by
    //Let Y be a vector of tij s, i  [1, k], j  [1, m].
    fn estimate_y(bv: BitVec) -> Vec<Array1<f32>> {
        let k = bv.len(); // size of filter
        let h = 1.; // number of hash functions
        let f = 0.5; // permanent response randomizer
        let p = 0.5; // temporary response randomizer
        let q = 0.5; // temporary response randomizer
        let m = 1.; // number of cohorts (groups of hash functions used by clients)

        // Cohorts implement different sets of h hash functions for their Bloom filters, thereby
        // reducing the chance of accidental collisions of two strings
        // across all of them.
        // what's the hash function?
        let cohorts = vec![vec![bv]]; // for now, just one cohort of only one client

        let init = || Array1::<f32>::zeros(k);

        let reported_counts_by_cohort = cohorts
            .iter()
            .map(|cohort| {
                cohort
                    .iter()
                    .fold(init(), |acc, curr_bv| acc + to_a1(curr_bv))
            })
            .collect::<Vec<Array1<f32>>>(); // TODO need to capture number of reports per cohort here too
        let n = 1.; // cheating here hard coding to 1 report per cohort

        // this is y, pass it along
        let estimated_true_counts_by_cohort = reported_counts_by_cohort
            .iter()
            .map(|counts| {
                counts.map(|count| {
                    (count - (p + 0.5 * f * q - 0.5 * f * p) * n) / ((1. - f) * (q - p))
                })
            })
            .collect::<Vec<Array1<f32>>>();

        estimated_true_counts_by_cohort
    }

    // Fit a regular least-squares regression using the selected
    // variables to estimate counts, their standard errors and
    // p-values.
    fn least_square_regression() {}

    /*
    We need to do regress(X, Y).L2_regularization(lambda)
    Where X and Y are data, lambda is a hyperparameter (the default in Numpy/scikit is 1 so start with that maybe)
    */
    #[test]
    fn test() -> Result<(), ErrorKind> {
        // test case
        let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
        estimate_y(bv);
        Ok(())
    }

    #[test]
    fn test_regression() -> Result<(), ErrorKind> {
        // define some test data
        let data_y = array![0.3, 1.3, 0.7];
        let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];

        linear_regression(data_x, data_y)
    }
}
