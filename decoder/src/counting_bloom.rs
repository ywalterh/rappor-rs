use super::decode::Factory;
use ndarray::*;
use std::collections::HashMap;
use std::error::Error;

// This method is probably not useful for actual production use
fn sum_bits(
    data_csv: &str,
    num_cohorts: usize,
    num_bloombits: usize,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(data_csv)?;

    let mut data_map = HashMap::<usize, usize>::new();

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
        let irr = r.get(4).unwrap().parse::<String>().unwrap();

        let mut counts = count_bits(irr);
        if data_map.contains_key(&cohort_id) {
            counts = counts + data_map.get(&cohort_id).unwrap();
        }
        data_map.insert(cohort_id, counts);
    }

    Ok(())
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
        sum_bits("tests/case_reports_shrinked.csv", 64, 16)?;
        println!("Took {}ms", now.elapsed().as_millis());
        Ok(())
    }
}
