use ndarray::array;
use ndarray_glm::{linear::Linear, model::ModelBuilder, standardize::standardize};
use ndarray_glm::error::RegressionError;
use bit_vec::BitVec;
use ndarray::ArrayBase;
use ndarray::Array2;
use ndarray::Array1;
use ndarray::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn to_a1(bv : &BitVec) -> Array1<f32> {
        Array1::from(
            bv.iter()
                .map(|bit| if bit { 1. } else { 0. }).collect::<Vec<f32>>()
        )
    }

    #[test]
    fn test() -> Result<(), RegressionError> {
        let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
        let k = bv.len(); // size of filter
        let h = 1; // number of hash functions
        let f = 0.5; // permanent response randomizer
        let p = 0.5; // temporary response randomizer
        let q = 0.5; // temporary response randomizer
        let m = 1; // number of cohorts (groups of hash functions used by clients)

        let cohorts = vec![vec![bv]]; // for now, just one cohort of only one client

        let init = || Array1::<f32>::zeros(k);
       
        let true_counts_by_cohort = cohorts.iter().map(|cohort| {
            cohort.iter().fold(init(), |acc, curr_bv| {
                acc + to_a1(curr_bv)
            })
        });


        // define some test data
        let data_y = array![0.3, 1.3, 0.7];
        let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
        // The design matrix can optionally be standardized, where the mean of each independent
        // variable is subtracted and each is then divided by the standard deviation of that variable.
        let data_x = standardize(data_x);
        // The model is generic over floating point type for the independent data variables.
        // If the second argument is blank (`_`), it will be inferred if possible.
        // L2 regularization can be applied with l2_reg().
        let model = ModelBuilder::<Linear, f32>::new(&data_y, &data_x).l2_reg(1e-5).build()?;
        let fit = model.fit()?;
        // Currently the result is a simple array of the MLE estimators, including the intercept term.
        println!("Fit result: {}", fit.result);
        Ok(())
    }
}
