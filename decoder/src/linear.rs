#[cfg(test)]
mod tests {
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
    fn test_ols() {
        let xs = array![2.0, 3.0, 4.0, 5.0];
        let ys = array![1.0, 2.0, 2.3, 0.4];
    }
}
