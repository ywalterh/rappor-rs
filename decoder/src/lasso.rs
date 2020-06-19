// this lasso implementation might be very naive and based on ndarray
// but it should work fine enough for no
// converting this into a test case and use the same strategy but written in rust
// https://github.com/J3FALL/LASSO-Regression/blob/master/lasso.py
use ndarray::*;
use ndarray_linalg::norm::normalize;
use ndarray_linalg::norm::NormalizeAxis;

#[derive(Debug)]
pub struct LassoFactory {
    pub weights: Array1<f64>,
}

fn get_initial_weights(nums: usize) -> Array1<f64> {
    let mut weights = Array1::<f64>::zeros(nums);
    for mut row in weights.genrows_mut() {
        row.fill(0.5);
    }

    return weights;
}

impl LassoFactory {
    pub fn new(nums: usize) -> Self {
        LassoFactory {
            weights: get_initial_weights(nums),
        }
    }

    // get feature matrix as x, and output as y
    // return weights
    pub fn train(&mut self, x: Array2<f64>, y: &Array1<f64>) {
        let l1_penalty = 0.01;
        let tolerance = 0.01;
        let x_normalized = self.normalize_features(x);
        self.cyclical_coordinate_descent(x_normalized, y, l1_penalty, tolerance);
    }

    // this is L2??
    fn normalize_features(&self, x: Array2<f64>) -> Array2<f64> {
        let (n, _) = normalize(x, NormalizeAxis::Column);
        return n;
    }

    pub fn predict_output(&self, feature_matrix: &Array2<f64>) -> Array1<f64> {
        return feature_matrix.dot(&self.weights);
    }

    fn coordinate_descent_step(
        &self,
        num_features: usize,
        feature_matrix: &Array2<f64>,
        output: &Array1<f64>,
        l1_penalty: f64,
    ) -> f64 {
        let predication = self.predict_output(feature_matrix);

        let mut new_weight_i = 0.;
        for i in 0..(num_features + 1) {
            let col = feature_matrix.column(i);
            //XXX walterh - spent too much time deal with this line
            // please read ndarray-rs Binary Opertors between arrays and scalar
            let ro_i = (&col * &(output - &predication + self.weights[i] * &col)).sum();
            //println!("RO {} : {}", i, ro_i);
            if i == 0 {
                new_weight_i = ro_i
            } else if ro_i < -l1_penalty / 2. {
                new_weight_i = ro_i + (l1_penalty / 2.);
            } else if ro_i > l1_penalty / 2. {
                new_weight_i = ro_i - (l1_penalty / 2.);
            } else {
                new_weight_i = 0.;
            }
        }

        new_weight_i
    }

    fn cyclical_coordinate_descent(
        &mut self,
        feature_matrix: Array2<f64>,
        output: &Array1<f64>,
        l1_penalty: f64,
        tolerance: f64,
    ) {
        let mut condition = true;
        while condition {
            let mut max_change = 0.;
            for i in 0..self.weights.len() {
                let old_weight_i = self.weights[i];
                self.weights[i] =
                    self.coordinate_descent_step(i, &feature_matrix, &output, l1_penalty);
                let coordinate_change = (old_weight_i - self.weights[i]).abs();

                if coordinate_change > max_change {
                    max_change = coordinate_change
                }
            }

            if max_change < tolerance {
                condition = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_lasso() {
        let mut undertest = LassoFactory::new(16);

        // initialize x
        let mut x = Array1::<f64>::zeros(60);
        for mut row in x.genrows_mut() {
            let mut j = 60.0;
            for i in 0..row.len() {
                row[i] = j * PI / 180_f64;
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

        x_matrix = undertest.normalize_features(x_matrix);
        let feature_matrix = x_matrix.clone();

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

        undertest.train(x_matrix, &outputs);
        // make sure the model (weights) returned is what we got from python or C++
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

        let mut delta_sum = 0.;
        for i in 0..undertest.weights.len() {
            delta_sum = delta_sum + (undertest.weights[i] - real_weights[i]).abs();
        }

        assert!(
            delta_sum < 5.,
            format!(
                "Too different from real weights {}, getting \n{}\nexpected \n{}\n",
                delta_sum, undertest.weights, real_weights
            )
        );

        // also compare the result to sklearn lasso implementation
        // it can only make predications, so compare predications
        let sklearn_predication = array![
            1.00037868,
            0.99713958,
            0.99318062,
            0.98839889,
            0.98268459,
            0.97592131,
            0.96798647,
            0.95875172,
            0.94808359,
            0.93584415,
            0.92189185,
            0.90608248,
            0.88827024,
            0.86830903,
            0.84605384,
            0.82136235,
            0.79409669,
            0.76412545,
            0.73132578,
            0.69558587,
            0.65680746,
            0.61490873,
            0.56982731,
            0.52152357,
            0.46998408,
            0.41522533,
            0.35729755,
            0.29628876,
            0.23232885,
            0.16559378,
            0.09630969,
            0.02475698,
            -0.04872593,
            -0.12373916,
            -0.1998185,
            -0.27643334,
            -0.3529856,
            -0.42881006,
            -0.50317638,
            -0.57529341,
            -0.64431616,
            -0.70935621,
            -0.76949618,
            -0.82380937,
            -0.87138527,
            -0.91136247,
            -0.94297012,
            -0.96557976,
            -0.97876914,
            -0.98240044,
            -0.97671515,
            -0.96244858,
            -0.9409671,
            -0.91443189,
            -0.88599342,
            -0.86002131,
            -0.84237512,
            -0.84072207,
            -0.86490877,
            -0.92739459
        ];

        let my_predication = undertest.predict_output(&feature_matrix);
        let mut delta_sum = 0.;
        for i in 0..sklearn_predication.len() {
            delta_sum = delta_sum + (my_predication[i] - sklearn_predication[i]).abs();
        }

        assert!(
            delta_sum < 2.,
            format!(
                "Too different from real predication {}, getting \n{}\nexpected \n{}\n with {:?}",
                delta_sum, my_predication, sklearn_predication, undertest
            )
        );
    }
}
