use ndarray::array;
use ndarray_glm::{linear::Linear, model::ModelBuilder, standardize::standardize};
use ndarray_glm::error::RegressionError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() -> Result<(), RegressionError> {
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
