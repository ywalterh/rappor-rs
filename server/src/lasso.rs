// this lasso implementation might be very naive and based on ndarray
// but it should work fine enough for no
// converting this into a test case and use the same strategy but written in rust
// https://github.com/J3FALL/LASSO-Regression/blob/master/lasso.py
use ndarray::*;
use ndarray_linalg::norm::normalize;
use ndarray_linalg::norm::NormalizeAxis;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::f64::consts::PI;

fn linear_regression(x: Array2<f64>, y: Array1<f64>) -> Result<(), ErrorKind> {
    Ok(())
}

// this is L2??
fn normalize_features(x: Array2<f64>) -> Array2<f64> {
    let (n, _) = normalize(x, NormalizeAxis::Column);
    return n;
}

fn predict_output(feature_matrix: &Array2<f64>, weights: &Array1<f64>) -> Array1<f64> {
    return feature_matrix.dot(weights);
}

fn coordinate_descent_step(
    num_features: usize,
    feature_matrix: &Array2<f64>,
    output: &Array1<f64>,
    weights: &Array1<f64>,
    l1_penalty: f64,
) -> f64 {
    let predication = predict_output(feature_matrix, weights);

    let mut new_weight_i = 0.;
    for i in 0..(num_features + 1) {
        let col = feature_matrix.column(i);
        //XXX walterh - spent too much time deal with this line
        // please read ndarray-rs Binary Opertors between arrays and scalar
        let ro_i = (&col * &(output - &predication + weights[i] * &col)).sum();
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
    feature_matrix: Array2<f64>,
    output: Array1<f64>,
    initial_weights: Array1<f64>,
    l1_penalty: f64,
    tolerance: f64,
) -> Array1<f64> {
    let mut condition = true;
    let mut weights = initial_weights.clone();

    let mut tries = 0;
    while condition {
        if tries > 1000 {
            panic!("Tried more than 100 times")
        }
        tries = tries + 1;

        let mut max_change = 0.;
        for i in 0..weights.len() {
            let old_weight_i = initial_weights[i];
            weights[i] = coordinate_descent_step(i, &feature_matrix, &output, &weights, l1_penalty);
            let coordinate_change = (old_weight_i - weights[i]).abs();

            if coordinate_change > max_change {
                max_change = coordinate_change
            }
        }

        println!("max_change : {}", max_change);
        if max_change < tolerance {
            condition = false;
        }
    }

    weights
}

fn get_weights() -> Array1<f64> {
    let mut weights = Array1::<f64>::zeros(16);
    for mut row in weights.genrows_mut() {
        row.fill(0.5);
    }
    return weights;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso() {
        let data_y = array![0.3, 1.3, 0.7];
        let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];

        // use our regression
        linear_regression(data_x, data_y);
    }

    #[test]
    fn test_more_complicated() {
        // initialize x
        let mut x = Array1::<f64>::zeros(60);
        let mut i = 60_f64;
        for mut row in x.genrows_mut() {
            row.fill(i * PI / 180_f64);
            i = i + 4_f64;
        }

        // initialize X 2D array with 60 x 16
        let mut X = Array2::<f64>::zeros((x.len(), 16));
        for j in 0..x.len() {
            let mut row = X.row_mut(j);
            let x_val = x[j];
            row[0] = x_val;
            for i in 1..row.len() {
                row[i] = row[i - 1] * x_val;
            }
        }

        X = normalize_features(X);
        println!("{}", X);

        // randomize Y
        // let Y = x.mapv(f64::sin) + Array::random(60, Uniform::new(0., 0.15));
        //
        // use fix Y instead, why introduce sporadic in testing
        let Y = array![
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

        let l1_penalty = 0.01;
        let tolerance = 0.01;

        let weights = cyclical_coordinate_descent(X, Y, get_weights(), l1_penalty, tolerance);
        println!("{}", weights);
        // make sure the model (weights) returned is what we got from python or C++
        let real_weights = array![
            34.2442, -44.6799, 0., 0., 0., 5.22371, 5.47178, 0.586693, 0., 0., 0., 0., 0., 0., 0.,
            -1.61534
        ];
    }
}
