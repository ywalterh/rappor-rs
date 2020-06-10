use bloom::{BloomFilter, ASMS};
use rand::Rng;

// This contains the client implementation of RAPPOR
pub struct Factory {
    k: u32,
    rate: f32,
    f: f32,
    p: u8,
    q: u8,
}

impl Factory {
    pub fn new(rate: f32) -> Self {
        Factory {
            k: 32,
            rate,
            f: 0.2,
            p: 1,
            q: 0,
        }
    }

    pub fn initialize_bloom_to_bitarray(&self, value: String) -> BloomFilter {
        // step1: hash client's value v onto Bloom filter B of size k using h hash function
        // let's say k is 32
        let mut bf = BloomFilter::with_rate(self.rate, self.k);
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
        let k = (m as f32 * 0.5 * self.f) as u8;
        let l = (m as f32 * 1.0 * self.f) as u8;
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
        let mut instant_randomized: Vec<u8> = Vec::<u8>::new();
        for b in perm_randomized {
            if b {
                instant_randomized.push(self.p);
            } else {
                instant_randomized.push(self.q);
            }
        }

        // report instant_randomized
        return format!("{:?}", instant_randomized);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let f = Factory::new(0.01);
        let result = f.process("test".into());
        assert_ne!(result, "");
    }
}
