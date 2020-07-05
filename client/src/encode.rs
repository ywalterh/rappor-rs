use bloom::{BloomFilter, ASMS};
use byteorder::{BigEndian, WriteBytesExt};
use hmac::{Hmac, Mac, NewMac};
use rand::Rng;
use sha2::Sha256;

// This contains the client implementation of RAPPOR
pub struct EncoderFactory {
    pub k: usize,
    h: u32,
    pub f: f64,
    pub p: f64,
    pub q: f64,
}

impl EncoderFactory {
    pub fn new(h: u32) -> Self {
        EncoderFactory {
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

    // This is the encode method
    // return the actual String to transfer through something like WASM
    // should we just use gRPC or some sort and use byte directly
    pub fn encode(&self, cohort_id: u32, value: String) -> String {
        // Transfer cohort id into first four byte during hasing
        // not sure if this is necessary
        // to do that in Rust, I'm including two crates hex and byte order
        // @Cleanup
        let mut wrt: Vec<u8> = vec![];
        wrt.write_u32::<BigEndian>(cohort_id).unwrap();
        let value_to_encode = format! {"{}{}", hex::encode(wrt), value};

        // the return type of this digest is a 16 bit of u8
        let digest = md5::compute(value_to_encode.as_bytes());
        assert!(digest.len() == 16);
        let digest_array: [u8; 16] = digest.into();

        // use bit wise or to caculate final bloom for randomization
        let mut bloom = 0;
        for b in digest_array.iter() {
            let b_shift = 1 << b;
            bloom |= b_shift;
        }

        // mask prr value
        // permanent randomized response with f, 1/2 f, 1/2f to 0, 1 - f with Bi
        // the reference implementation uses hmac, so I'm using hmac too
        type HmacSha256 = Hmac<Sha256>;
        let mut hmac = HmacSha256::new_varkey(b"secret").expect("HMAC can take key of any size");

        // the digest if the hash is 'v\x8d\x87Lul\xf6\t\xc8B\xaa\xbf\t\x03@\xb1' in python code
        println!("digest of {} is  {:x}", value_to_encode, digest);

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
    fn test_encoder_encode() {
        let f = EncoderFactory::new(1);
        let result = f.encode(1, "abc".into());
        assert_ne!(result, "");
        println!("{}", result);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_bloomfilter_bits() {
        // the only variable we are taking is num of hashes
        let f = EncoderFactory::new(1);
        let bi = f.initialize_bloom_to_bitarray("abc".into()).bits;
        println!("resulting bis is {:?}", bi);
    }
}
