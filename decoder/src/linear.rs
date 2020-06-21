#[cfg(test)]
mod tests {
    use ndarray::*;
    use ndarray_linalg::*;

    #[test]
    fn test_solve_linalg() -> Result<(), error::LinalgError> {
        let a: Array2<f64> = random((3, 3));
        let b: Array1<f64> = random(3);
        let x = a.solve(&b)?;
        println!("resutl is {:?}", x);
        Ok(())
    }
}
