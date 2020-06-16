use blas::*;
use lapack::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blas() {
        let mone = vec![2.0, 3.0, 4.0, 5.0];
        let mtwo = vec![0.5, 1.0, 1.5, 2.0];
        let mut result = vec![0.0, 0.0, 0.0, 0.0];
        unsafe {
            dgemm(
                b'N',
                b'N',
                2,
                2,
                2,
                1.0,
                &mone,
                2,
                &mtwo,
                2,
                1.0,
                &mut result,
                2,
            );
        }

        println!("{:?}", &result);
    }

    #[test]
    fn test_ols() {
        let (rows, cols) = (4, 1);
        let scale_by = 1.0;
        let xs = vec![2.0, 3.0, 4.0, 5.0];
        let ys = vec![1.0, 2.0, 2.3, 0.4];
        let mut xtx = vec![0.0];
        let mut xty = vec![0.0];

        let mut result = vec![0.0];

        unsafe {
            dgemm(
                b'T',
                b'N',
                cols, // M : number of rows of X'
                cols, // N : number of colums of X
                rows, // K
                scale_by,
                &xs,
                rows,
                &xs,
                rows,
                scale_by,
                &mut result,
                rows,
            );
            dgemm(
                b'T', b'N', cols, cols, rows, scale_by, &xs, rows, &ys, rows, scale_by, &mut xty,
                rows,
            );
        }

        // IPIV will be the pivot indices for the LU decomposition, with minimum dimensions of (m, n)
        let mut ipiv = vec![0];

        let mut status = 0;
        // Then we'll need the inverse of xtx
        unsafe {
            // In production, check status after each step
            // First we need to generate the LU decomposition of the XTX matrix
            dgetrf(1, 1, &mut xtx, 1, &mut ipiv, &mut status);
        }

        unsafe {
            // Then we can generate the inverse of the matrix from the LU decomposition
            let mut workspace = vec![0.0];
            dgetri(1, &mut xtx, 1, &mut ipiv, &mut workspace, 1, &mut status);
        }
        // XTX has now been overwritten by its inverse
        // Back to BLAS
        unsafe {
            dgemm(
                b'T',
                b'N',
                1,
                1,
                4,
                scale_by,
                &xtx,
                4,
                &xty,
                4,
                scale_by,
                &mut result,
                4,
            );
        }

        println!("result is : {:?}", &result);
    }
}
