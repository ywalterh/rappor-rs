use bloom::{ASMS,BloomFilter};

// This contains the client implementation of RAPPOR
pub struct Factory {
    rate: f32,
}

impl Factory {
    fn new(rate: f32) -> Self {
        Factory { rate }
    }

    fn process(&self, value: String) -> String {
        // step1: hash client's value v onto Bloom filter B of size k using h hash function
        // let's say k is 32
        let mut bf = BloomFilter::with_rate(self.rate, 32);
        bf.insert(&value); 
        // this is the B[i] set
        let bv = bf.bits;
        for i in bv {
            println!("{}", i);
        }

        // permanent randomized response
        return "".into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn it_works() {
        let f = Factory::new(0.01);
        let result = f.process("test".into());
        assert_ne!(result, "");
    }

}
