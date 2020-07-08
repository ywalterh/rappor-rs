use super::decode::Factory;
use ndarray::*;
use std::collections::HashMap;
use std::error::Error;

fn sum_bits(
    data_csv: &str,
    num_cohorts: usize,
    num_bloombits: usize,
) -> Result<Array2<u8>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(data_csv)?;

    let mut data_map = HashMap::<usize, Vec<String>>::new();

    // load in memory first and then. it's only 13M data
    // partitioning data here
    // this is actually very fast when build with --release
    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        //
        // each row is user_id, cohort, unused_bloom, unused_prr, irr) = row
        // so in this case is only cohort_id, irr
        let r = result?;
        let cohort_id = r.get(1).unwrap().parse::<usize>().unwrap();
        let cohort_vec = data_map.entry(cohort_id).or_insert(Vec::new());

        cohort_vec.push(r.get(2).unwrap().into());
    }

    let array = Array2::zeros((64, 16));
    Ok(array)
}

// Return number of positive bits in a string of IRR string
fn count_bits(irr: String) -> usize {
    let mut sum = 0;
    for bit in irr.chars() {
        if bit == '1' {
            sum = sum + 1;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_count_bits() {
        let sum_result = count_bits("00001000".into());
        assert_eq!(1, sum_result);
    }

    // try to work out how to use glmnet
    #[test]
    fn test_sum_bits() -> Result<(), Box<dyn Error>> {
        let now = Instant::now();
        sum_bits("tests/case_reports.csv", 64, 16)?;
        println!("Took {}ms", now.elapsed().as_millis());
        Ok(())
    }
}
