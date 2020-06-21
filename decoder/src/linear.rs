extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

#[derive(Debug, Default)]
pub struct Factory {
    coefficients: Array2<f64>,
    num_of_observations: usize,
    num_of_coefficient: usize,

    degrees_of_freedom_error: usize,
    degrees_of_freedom_regression: usize,

    sse: f64,
    coef_standard_errors: Array1<f64>,
    coef_t: Array2<f64>,
    coef_p: f64,

    residuals: Array1<f64>,
}

impl Factory {
    pub fn new() -> Self {
        Factory::default()
    }

    // let's not deal with labels
    // we only take x as 2D array, and y as 1D array
    pub fn ols(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), error::LinalgError> {
        let inv_xx = (x.t().dot(x)).inv()?;
        let xy = x.t().dot(y);
        self.coefficients = inv_xx.dot(x);
        self.num_of_observations = y.shape()[0];
        self.num_of_coefficient = x.shape()[1];
        self.degrees_of_freedom_error = self.num_of_observations - self.num_of_coefficient;
        self.degrees_of_freedom_regression = self.num_of_coefficient - 1;
        self.residuals = y - &x.dot(&self.coefficients);
        // not sure what sse is, maybe @dylan knows?
        self.sse = self.residuals.dot(&self.residuals);
        // into_diag only works for 1D array
        self.coef_standard_errors = (self.sse * &inv_xx).into_diag().mapv(f64::sqrt);
        self.coef_t = &self.coefficients / &self.coef_standard_errors;
        /* I don't have cdf yet
        self.coef_p =
            (1. - cdf(
                self.coef_t.mapv(f64::abs),
                self.degrees_of_freedom_regression,
            )) * 2.;

            */
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::*;
    use ndarray_linalg::*;

    #[test]
    fn test_solve_linalg() -> Result<(), error::LinalgError> {
        // this is just an example of solving a linear regression
        // and a is a square matrix, which is required
        let a: Array2<f64> = random((3, 3));
        let b: Array1<f64> = random(3);
        let x = a.solve(&b)?;
        println!("resutl is {:?}", x);
        Ok(())
    }

    #[test]
    fn test_ols() -> Result<(), error::LinalgError> {
        let mut f = Factory::new();
        let x = array![[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]];
        let y = array![1.0, 2.0, 2.3, 0.4];
        //f.ols(&x, &y)?;
        println!("{:?}", f);
        Ok(())
    }
}
