use bloom::{BloomFilter, ASMS};
use byteorder::{BigEndian, ByteOrder};
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
    pub num_hashes: usize,
    pub num_bloombits: u8,
}

impl EncoderFactory {
    pub fn new(h: u32) -> Self {
        EncoderFactory {
            k: 32,
            h,
            f: 0.5,
            p: 0.5,
            q: 0.75,
            num_hashes: 2,
            num_bloombits: 16,
        }
    }

    pub fn initialize_bloom_to_bitarray(&self, value: String) -> BloomFilter {
        // step1: hash client's value v onto Bloom filter B of size k using h hash function
        // let's say k is 32
        let mut bf = BloomFilter::with_size(self.k, self.h);
        bf.insert(&value);
        return bf;
    }

    // even thought it's declaring 32 bit, but we currenlty only using 16 bit
    fn get_bloom(
        &self,
        cohort_id: u32,
        word: &String,
        num_hashes: usize,
        num_bloombits: u8, // num_bloombits is u8 because we are using a mod (%) below
    ) -> u32 {
        // Transfer cohort id into first four byte during hasing
        // not sure if this is necessary
        // to do that in Rust, I'm including two crates hex and byte order
        // @Cleanup
        let mut buf = [0; 4];
        BigEndian::write_u32(&mut buf, cohort_id);
        let value_to_encode = format! {"{}{}", hex::encode(&buf), word};

        // the return type of this digest is a 16 bit of u8
        // Each has has  byte, which means we could have up to 256 bit bloom filters
        // There are 16 bytes in aan MD5, so we can have up to 16 hash functions
        // per bloom filter, in this case, we need to find out how many
        // we are using, hence num_hashes are taken
        let digest = md5::compute(value_to_encode.as_bytes());
        assert!(
            num_hashes <= digest.len(),
            "Can't have more num_hashes than digest length"
        );

        let digest_array: [u8; 16] = digest.into();

        // get bits per hash functions
        let mut bits = vec![];
        for i in 0..num_hashes {
            bits.push(digest_array[i] % num_bloombits);
        }

        // use bit wise or to caculate final bloom for randomization
        let mut bloom = 0;
        for b in bits.iter() {
            let b_shift = 1 << b;
            bloom |= b_shift;
        }

        bloom
    }

    fn get_prr_masks(&self, secret: &str, bloom: u32, num_bits: usize) -> (u32, u32) {
        let mut buf = [0; 4];
        // Transfer to big Endian do it again again st the same buffer, should be OK?
        // afterall this is rust
        BigEndian::write_u32(&mut buf, bloom);
        println!("Big Endian of bloom is {}", hex::encode(buf));

        // mask prr value
        // permanent randomized response with f, 1/2 f, 1/2f to 0, 1 - f with Bi
        // the reference python implementation uses hmac, so I'm using hmac too
        type HmacSha256 = Hmac<Sha256>;
        let mut hmac =
            HmacSha256::new_varkey(secret.as_bytes()).expect("HMAC can take key of any size"); // why not unwrap.. lolz
        hmac.update(&buf);
        let digest = hmac.finalize().into_bytes();
        assert!(digest.len() == 32, "size of digest should be 32"); //@Cleanup this is probably redundunt
        assert!(num_bits <= digest.len(), "max digest if sha256 is 32bit");

        let threshold128 = (self.f * 128.) as u8;
        let mut uniform = 0;
        let mut f_mask = 0;

        for i in 0..num_bits {
            let byte = digest[i] as u8;
            let u_bit = byte & 0x01; // 1 bit of entropy
            uniform |= (u_bit as u32) << 1;
            let rand128 = byte >> 1; // 7 bits of entropy
            let noise_bit = (rand128 < threshold128) as u32;
            f_mask |= noise_bit << i;
        }

        (uniform, f_mask)
    }

    // This is the encode method
    // return the actual String to transfer through something like WASM
    // should we just use gRPC or some sort and use byte directly
    pub fn encode(&self, cohort_id: u32, word: String) -> String {
        let bloom = self.get_bloom(cohort_id, &word, self.num_hashes, self.num_bloombits);
        let (uniform, f_mask) = self.get_prr_masks("secret", bloom, 32);
        /*
            # Suppose bit i of the Bloom filter is B_i.  Then bit i of the PRR is
            # defined as:
            #
            # 1   with prob f/2
            # 0   with prob f/2
            # B_i with prob 1-f

            # Uniform bits are 1 with probability 1/2, and f_mask bits are 1 with
            # probability f.  So in the expression below:
            #
            # - Bits in (uniform & f_mask) are 1 with probability f/2.
            # - (bloom_bits & ~f_mask) clears a bloom filter bit with probability
            # f, so we get B_i with probability 1-f.
            # - The remaining bits are 0, with remaining probability f/2.
        */
        let prr = (bloom & !f_mask) | (uniform & f_mask);
        println!("prr ir {}", prr);

        /*
            # Compute Instantaneous Randomized Response (IRR).
            # If PRR bit is 0, IRR bit is 1 with probability p.
            # If PRR bit is 1, IRR bit is 1 with probability q
        */
        let mut rng = rand::thread_rng();
        let threashold256_p = (self.p * 256.) as u8;
        let p_bits = (rng.gen::<u8>() < threashold256_p) as u32;
        let threashold256_q = (self.q * 256.) as u8;
        let q_bits = (rng.gen::<u8>() < threashold256_q) as u32;

        // this is RAPPOR
        let irr = (p_bits & !prr) | (q_bits & prr);
        irr.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_bloom() {
        let f = EncoderFactory::new(1);
        let bloom = f.get_bloom(1, &"abc".into(), f.num_hashes, f.num_bloombits);
        assert_eq!(bloom, 8196);
    }

    #[test]
    fn test_get_prr_masks() {
        let f = EncoderFactory::new(1);
        assert_eq!(
            f.get_prr_masks("secret", 5000, f.num_bloombits as usize),
            (2, 48078)
        );
    }

    #[test]
    fn test_encoder_encode() {
        let f = EncoderFactory::new(1);
        let result = f.encode(1, "abc".into());
        assert_ne!(result, "");
        println!("Result is {}", result);
    }

    #[test]
    fn test_bloomfilter_bits() {
        // the only variable we are taking is num of hashes
        let f = EncoderFactory::new(1);
        let bi = f.initialize_bloom_to_bitarray("abc".into()).bits;
        println!("resulting bis is {:?}", bi);
    }
}
