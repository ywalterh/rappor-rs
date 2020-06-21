use bloom::{BloomFilter, ASMS};
use rand::Rng;

// This contains the client implementation of RAPPOR
pub struct Factory {
    k: usize,
    h: u32,
    pub f: f64,
    pub p: f64,
    pub q: f64,
}

impl Factory {
    pub fn new(h: u32) -> Self {
        Factory {
            k: 32,
            h,
            f: 0.5,
            p: 0.5,
            q: 0.75,
        }
    }

    pub fn initialize_bloom_to_bitarray(&self, value: String) -> BloomFilter {
        // step1: hash client's value v onto Bloom filter B of size k using h hash function
        // let's say k is 32
        let mut bf = BloomFilter::with_size(self.k, self.h);
        bf.insert(&value);
        return bf;
    }

    pub fn process(&self, value: String) -> String {
        // this is the B[i] set
        let bi = self.initialize_bloom_to_bitarray(value).bits;

        // permanent randomized response with f, 1/2 f, 1/2f to 0, 1 - f with Bi
        let mut rng = rand::thread_rng();
        let m = 200;

        let mut perm_randomized = Vec::<bool>::new();
        let k = (m as f64 * 0.5 * self.f) as u8;
        let l = (m as f64 * 1.0 * self.f) as u8;
        for b in bi {
            let random_number = rng.gen_range(0, m);
            if random_number <= k {
                perm_randomized.push(true);
            } else if k < random_number && random_number <= l {
                perm_randomized.push(false);
            } else {
                perm_randomized.push(b);
            }
        }

        // instant randomized response
        let mut instant_randomized = String::new();
        let q_k = (m as f64 * self.q) as u8;
        let p_k = (m as f64 * self.p) as u8;
        for b in perm_randomized {
            let random_number = rng.gen_range(0, m);
            if b {
                if random_number <= q_k {
                    instant_randomized.push('1');
                } else {
                    instant_randomized.push('0');
                }
            } else {
                if random_number <= p_k {
                    instant_randomized.push('1');
                } else {
                    instant_randomized.push('0');
                }
            }
        }

        return instant_randomized;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let f = Factory::new(1);
        let result = f.process("test".into());
        assert_ne!(result, "");
        println!("{}", result);
        assert_eq!(result.len(), 32);
    }
}
