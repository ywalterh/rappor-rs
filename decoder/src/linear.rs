extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use statrs::distribution::{StudentsT, Univariate};

#[derive(Debug, Default)]
pub struct Factory {
    pub coefficients: Array1<f64>,
    num_of_observations: usize,
    num_of_coefficient: usize,

    degrees_of_freedom_error: usize,
    degrees_of_freedom_regression: usize,

    sse: f64,
    coef_standard_errors: Array1<f64>,
    coef_t: Array1<f64>,
    pub coef_p: Array1<f64>,

    residuals: Array1<f64>,
}

impl Factory {
    pub fn new() -> Self {
        Factory::default()
    }

    // let's not deal with labels
    // we only take xk'jas 2D array, and y as 1D array
    pub fn ols(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), error::LinalgError> {
        let inv_xx = (x.t().dot(x)).inv()?;
        let xy = x.t().dot(y);
        self.coefficients = inv_xx.dot(&xy);
        self.num_of_observations = y.shape()[0];
        self.num_of_coefficient = x.shape()[1];
        self.degrees_of_freedom_error = self.num_of_observations - self.num_of_coefficient;
        self.degrees_of_freedom_regression = self.num_of_coefficient - 1;
        self.residuals = y - &x.dot(&self.coefficients);
        // not sure what sse is, maybe @dylan knows?
        self.sse = self.residuals.dot(&self.residuals) / self.degrees_of_freedom_error as f64;
        // into_diag only works for 1D array
        self.coef_standard_errors = (self.sse * &inv_xx).into_diag().mapv(f64::sqrt);
        self.coef_t = &self.coefficients / &self.coef_standard_errors;
        self.coef_p =
            (1. - self.cdf(
                &self.coef_t.mapv(f64::abs),
                self.degrees_of_freedom_error as f64,
            )) * 2.;
        Ok(())
    }

    // make cdf works on an array so that we can calculate p-value for all the cofficients
    fn cdf(&self, distribution: &Array1<f64>, df: f64) -> Array1<f64> {
        distribution.mapv(|x| {
            let n = StudentsT::new(x, 1., df).unwrap();
            n.cdf(1.)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::*;
    use rand::distributions::Uniform;

    #[test]
    fn test_cdf() {
        let real_weights = array![
            30.068216996847337,
            -34.433605927665084,
            -8.344245977622037,
            0.,
            2.56660336442419,
            3.9890280109854217,
            3.119234805950629,
            2.0726629033639665,
            1.187156437842501,
            0.42083617225270087,
            0.,
            0.,
            0.,
            0.,
            -0.7679035712433514,
            -1.316257337552551
        ];

        let f = Factory::new();
        let result = f.cdf(&real_weights.mapv(f64::abs), 2.);
    }

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
        // initialize x
        let mut x = Array1::<f64>::zeros(60);
        for mut row in x.genrows_mut() {
            let mut j = 60.0;
            for i in 0..row.len() {
                row[i] = j * std::f64::consts::PI / 180_f64;
                j = j + 4.0;
            }
        }

        // initialize X 2D array with 60 x 5
        let mut x_matrix = Array2::<f64>::zeros((x.len(), 5));
        for j in 0..x.len() {
            let mut row = x_matrix.row_mut(j);
            row[0] = x[j];
            for i in 1..row.len() {
                row[i] = row[i - 1] * x[j];
            }
        }

        // randomize Y
        let outputs = x.mapv(f64::sin) + Array::random(60, Uniform::new(0., 0.15));
        let mut f = Factory::new();
        f.ols(&x_matrix, &outputs)?;
        Ok(())
    }
}
