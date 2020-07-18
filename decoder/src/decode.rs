use super::feature_matrix;
use super::lasso;
use super::linear;
use bit_vec::BitVec;
use client::encode;
use ndarray::*;
use std::io::ErrorKind;

pub struct Factory {
    pub encoder: encode::EncoderFactory,
}

const num_cohorts: usize = 64;

impl Factory {
    pub fn new() -> Self {
        Factory {
            encoder: encode::EncoderFactory::new(1),
        }
    }

    pub fn fit_model(&self, result: &Array2<f64>, y_vector: &Array1<f64>) {
        let mut ols_factory = linear::Factory::new();
        let result = ols_factory.ols(result, y_vector);
        match result {
            Ok(()) => {
                println!("coef is {}", ols_factory.coefficients);
                println!("coef_p is {}", ols_factory.coef_p);

                // compare to Bonferroni correction of 0.05/M
                // in this case, there are 5 candidate strings, so 0.01 is the correction level
                // less 0.01
                for (i, p) in ols_factory.coef_p.iter().enumerate() {
                    if *p < 0.01 {
                        println!("Found one signifant i {}", i)
                    }
                }
            }
            Err(err) => {
                println!("skip due to error {}", err);
            }
        }
    }

    //Estimate the number of times each bit i within cohort
    //j, tij , is truly set in B for each cohort. Given the
    //number of times each bit i in cohort j, cij was set in
    //a set of Nj reports, the estimate is given by
    //Let Y be a vector of tij s, i  [1, k], j  [1, m].
    pub fn estimate_y(&self, cohorts: Vec<Vec<BitVec>>) -> Vec<Array1<f64>> {
        let k = self.encoder.k; // size of filter let h = 1.; // number of hash functions
        let f = self.encoder.f;
        let p = self.encoder.p;
        let q = self.encoder.q;

        // let m = cohorts.len();
        // Cohorts implement different sets of h hash functions for their Bloom filters, thereby
        // reducing the chance of accidental collisions of two strings
        // across all of them.
        // what's the hash function?
        let init = || Array1::<f64>::zeros(k);

        let reported_counts_by_cohort = cohorts
            .iter()
            .map(|cohort| {
                cohort
                    .iter()
                    .fold(init(), |acc, curr_bv| acc + to_a1(curr_bv))
            })
            .collect::<Vec<Array1<f64>>>();

        // TODO need to capture number of reports per cohort here too
        // now we have more cohorts, need to update this
        //let n = (&bv).len() as f64; // cheating here hard coding to 1 report per cohort
        // this is y, pass it along
        let estimated_true_counts_by_cohort = reported_counts_by_cohort
            .iter()
            .map(|counts| {
                let n = counts.len() as f64;
                counts.map(|count| {
                    (count - (p + 0.5 * f * q - 0.5 * f * p) * n) / ((1. - f) * (q - p))
                })
            })
            .collect::<Vec<Array1<f64>>>();

        estimated_true_counts_by_cohort
    }

    // take Y and X then produce a model that fit for step 3
    // take ndarray
    // producce a fit result Y of X
    // TODO fix to use lasso here, or at least something similar
    // select candidate strings corresponding to non-zero coefficients.
    pub fn lasso_select_string(&self, y: &Array1<f64>) -> Result<Array2<f64>, ErrorKind> {
        // train the model of desigm matrix
        // the default bahavior of this is five candidate strings
        let matrix = feature_matrix::create_design_matrix(
            self.encoder.num_bloombits as usize,
            num_cohorts,
            feature_matrix::test_candidate_strings,
        );
        let mut lasso_factory = lasso::LassoFactory::new(5);
        lasso_factory.train(&matrix, y);
        println!("predicated lasso weights is {}", lasso_factory.weights);

        // pick the strings with non-zero coefficiency
        // look through weights
        // we have a,b,c,d,e
        // return another matrix I believe?
        // and then use the selected matrix to run through OLS to select counts of actual hits
        let mut left_candiate_string_index = Vec::new();
        for (index, w) in lasso_factory.weights.iter().enumerate() {
            if *w != 0. {
                left_candiate_string_index.push(index);
            }
        }

        let mut updated_feature_matrix =
            Array2::<f64>::zeros((*&matrix.shape()[0], left_candiate_string_index.len()));

        // update stream based on non-zero coefficiency
        let mut index = 0;
        for mut row in updated_feature_matrix.genrows_mut() {
            let original_row = matrix.row(index);
            for j in 0..row.len() {
                row[j] = original_row[j];
            }
            index = index + 1;
        }

        Ok(updated_feature_matrix)
    }
}

fn to_a1(bv: &BitVec) -> Array1<f64> {
    Array1::from(
        bv.iter()
            .map(|bit| if bit { 1. } else { 0. })
            .collect::<Vec<f64>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::*;
    use rayon::prelude::*;
    use std::error::Error;

    #[test]
    fn test_fit_model() -> Result<(), error::LinalgError> {
        let f = Factory::new();

        for _ in 0..1 {
            // have 20 reporting instead of one
            let mut bvs = Vec::new();
            // let n = 1000
            let n = 1000;
            // let's sholve 99% "a"
            for i in 0..n {
                let mut test_string = "a";
                if i > 900 {
                    test_string = "b";
                }
                let encoded = f.encoder.encode(1, test_string.into());
                bvs.push(feature_matrix::u32_to_bitvec(encoded));
            }

            let y = f.estimate_y(vec![bvs]);
            let y_vector = &y[0];
            let result = f.lasso_select_string(y_vector).unwrap();
            // then run this against ols again
            // TODO before we can comfortably use lasso which should rule out
            // most cases using non-zero coefficient, no point running OLS and causing
            // problem
            // print out coef and coef_p
            let mut ols_factory = linear::Factory::new();
            let result = ols_factory.ols(&result, y_vector);
            match result {
                Ok(()) => {
                    println!("coef is {}", ols_factory.coefficients);
                    println!("coef_p is {}", ols_factory.coef_p);

                    // compare to Bonferroni correction of 0.05/M
                    // in this case, there are 5 candidate strings, so 0.01 is the correction level
                    // less 0.01
                    for (i, p) in ols_factory.coef_p.iter().enumerate() {
                        if *p < 0.01 {
                            println!("Found one signifant i {}", i)
                        }
                    }
                }
                Err(err) => {
                    println!("skip due to error {}", err);
                }
            }
        }
        Ok(())
    }

    /*
    We need to do regress(X, Y).L2_regularization(lambda)
    Where X and Y are data, lambda is a hyperparameter (the default in Numpy/scikit is 1 so start with that maybe)
    */
    #[test]
    fn test_estimate_y() -> Result<(), ErrorKind> {
        let f = Factory::new();
        // let's say we have five candidate strings
        // and we received cohort from a particular report

        // test case
        let bv = BitVec::from_bytes(&[0b10100000, 0b00010010, 0b00010010, 0b00010010]);
        let y = f.estimate_y(vec![vec![bv]]);

        // another bv of the same
        let bv = BitVec::from_bytes(&[0b10100000, 0b00010010, 0b00010010, 0b00010010]);

        // create design matrix X of size km X M where M is the number of candidate strings
        // the matrix is 1 if bloom filter bits for each string for each cohort
        // in our case, we probably have k = 1, because we are lazy and only one cohort
        // let's say M = 5 for this test
        let mut x = Array2::<f64>::zeros((f.encoder.k, 5));
        for mut row in x.genrows_mut() {
            for i in 0..row.len() {
                if bv[i] {
                    row[i] = 1.;
                } else {
                    row[i] = 0.;
                }
            }
        }

        let nan = f64::NAN;
        assert!(y[0].sum() != nan);
        Ok(())
    }

    use std::collections::HashMap;
    use std::sync::mpsc::channel;
    use std::time::Instant;

    #[test]
    fn test_regression() -> Result<(), Box<dyn Error>> {
        let f = Factory::new();
        let mut rdr = csv::Reader::from_path("tests/test-cases.csv")?;

        let now = Instant::now();
        let mut data_map = HashMap::<usize, Vec<String>>::new();
        // load in memory first and then. it's only 13M data
        // partitioning data here
        for result in rdr.records() {
            // The iterator yields Result<StringRecord, Error>, so we check the
            // error here.
            let r = result?;
            let cohort_id = r.get(1).unwrap().parse::<usize>().unwrap();
            let cohort_vec = data_map.entry(cohort_id).or_insert(Vec::new());
            cohort_vec.push(r.get(2).unwrap().into());
        }

        let (sender, receiver) = channel();
        // clone sender everytime we need something returned
        // it's a lot easier to dive in parallel in cohorts
        // process in parallel
        data_map
            .par_iter()
            .for_each_with(sender, |sender_c, (_, v)| {
                let (sender, receiver) = channel();
                v.par_iter().for_each_with(sender, |s_t, reported_string| {
                    s_t.send(feature_matrix::u32_to_bitvec(
                        f.encoder.encode(1, reported_string.clone()),
                    ))
                    .unwrap();
                });

                sender_c.send(receiver.iter().collect()).unwrap();
            });

        let cohorts: Vec<Vec<BitVec>> = receiver.iter().collect();
        println!("size of cohorts is {}", cohorts.len());
        println!("It took {} ms to process text", now.elapsed().as_millis());

        let now = Instant::now();
        f.estimate_y(cohorts);
        println!("It took {} ms to estimate y", now.elapsed().as_millis());

        // select lasso in parallel, rest channel for other purpose
        // this part is not useful yet
        // until we have real lasso and OLS implementation
        /*
        let now = Instant::now();
        result.par_iter().for_each(|v| {
            let mut ols_factory = linear::Factory::new();
            let selected_strings = f.lasso_select_string(v);
            ols_factory.ols(&selected_strings.unwrap(), v);
        });
        println!(
            "It took {} ms to do lasso and ols",
            now.elapsed().as_millis()
        );
        */

        Ok(())
    }
}
