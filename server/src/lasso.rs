// this lasso implementation might be very naive and based on ndarray
// but it should work fine enough for no
// converting this into a test case and use the same strategy but written in rust
// https://github.com/J3FALL/LASSO-Regression/blob/master/lasso.py
use ndarray::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::f64::consts::PI;

fn linear_regression(x: Array2<f32>, y: Array1<f32>) -> Result<(), ErrorKind> {
    Ok(())
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

    /*
    copy test setup from the example python script
    x = np.array([i * np.pi / 180 for i in range(60, 300, 4)])

    X = np.zeros((len(x), 16))
    for idx in range(len(x)):
        X[idx, 0] = x[idx]
        for j in range(1, 16):
            X[idx, j] = X[idx, j - 1] * x[idx]

    np.random.seed(10)
    Y = np.sin(x) + np.random.normal(0, 0.15, len(x))
    */
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
        let mut j = 0;
        for mut row in X.genrows_mut() {
            // whuuuut???
            let x_val = *x.get(j).unwrap();
            row[0] = x_val;
            for i in 1..row.len() {
                row[i] = row[i - 1] * x_val;
            }
        }

        // randomize Y
        let mut Y = x.mapv(f64::sin) + Array::random((1, 60), Uniform::new(0., 0.15));
    }
}
