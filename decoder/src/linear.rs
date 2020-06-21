extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

#[derive(Debug, Default)]
pub struct Factory {
    coefficients: Array1<f64>,
    num_of_observations: usize,
    num_of_coefficient: usize,

    degrees_of_freedom_error: usize,
    degrees_of_freedom_regression: usize,

    sse: f64,
    coef_standard_errors: Array1<f64>,
    coef_t: Array1<f64>,
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
        self.coefficients = inv_xx.dot(&xy);
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
        // initialize x
        let mut x = Array1::<f64>::zeros(60);
        for mut row in x.genrows_mut() {
            let mut j = 60.0;
            for i in 0..row.len() {
                row[i] = j * std::f64::consts::PI / 180_f64;
                j = j + 4.0;
            }
        }

        // initialize X 2D array with 60 x 16
        let mut x_matrix = Array2::<f64>::zeros((x.len(), 16));
        for j in 0..x.len() {
            let mut row = x_matrix.row_mut(j);
            row[0] = x[j];
            for i in 1..row.len() {
                row[i] = row[i - 1] * x[j];
            }
        }

        // randomize Y
        // let Y = x.mapv(f64::sin) + Array::random(60, Uniform::new(0., 0.15));
        //
        // use fix Y instead, why introduce sporadic in testing
        let outputs = array![
            1.06576338,
            1.00608589,
            0.69537381,
            0.94979894,
            1.06349612,
            0.87679492,
            1.03434863,
            1.01567311,
            1.00003454,
            0.96833186,
            1.04976168,
            1.15075133,
            0.80629667,
            1.08142497,
            0.93308857,
            0.93279605,
            0.65854724,
            0.80828129,
            0.96582538,
            0.53268764,
            0.34612837,
            0.32627941,
            0.56982979,
            0.82721666,
            0.57529033,
            0.59291348,
            0.29050974,
            0.41761115,
            0.0984859,
            0.1617371,
            -0.04009758,
            -0.15215283,
            -0.11926686,
            -0.27933299,
            -0.07936639,
            -0.31276815,
            -0.34670514,
            -0.52011641,
            -0.34144842,
            -0.69758068,
            -0.54375288,
            -0.74728915,
            -0.88405983,
            -0.86141134,
            -0.94972624,
            -0.89793005,
            -0.94966508,
            -0.88035836,
            -0.86628362,
            -0.99240876,
            -0.98869355,
            -0.95115776,
            -1.08037269,
            -0.89316682,
            -0.86818818,
            -0.95427063,
            -0.61109018,
            -0.81343768,
            -0.94402473,
            -0.95312111
        ];

        let mut f = Factory::new();
        f.ols(&x_matrix, &outputs)?;
        Ok(())
    }
}
