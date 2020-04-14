use bloom::{ASMS,BloomFilter};

// This contains the client implementation of RAPPOR
pub struct Factory {
    rate: f32,
}

// 8 bit
fn to_binary(set: &mut Vec<u8>, mut decimal: u8) {
    let mut bits = Vec::<u8>::new();
    if decimal == 0 {
        bits.push(0);
    } else {
        while decimal > 0 {
            if decimal % 2 == 0 {
                bits.push(0);
            } else {
                bits.push(1);
            }
            decimal /= 2;
        }
    }
    // add empty bit back
    let k = 8 - bits.len();
    for _ in  0..k {
        bits.push(0);
    }
    
    // reverse the bits
    bits.reverse();
    set.append(&mut bits);
}

// lack of good way of doing it for now, it would be better to use byte array directly
pub fn string_to_binary(value: String) -> Vec<u8> {
    let bytes = value.as_bytes();
    let mut result = Vec::<u8>::new();
    for byte in bytes {
        to_binary(&mut result, byte.clone());
    }
    return result;
}

impl Factory {
    fn new(rate: f32) -> Self {
        Factory { rate }
    }

    fn process(&self, value: String) -> String {
        let bits = string_to_binary(value);
        // step1: hash client's value v onto Bloom filter B of size k using h hash function
        let mut bf = BloomFilter::with_rate(self.rate, bits.len() as u32);
        for bit in bits {
            bf.insert(&bit);
        }
        // this is the B[i] set
        let bv = bf.bits;
        for i in bv {
            println!("{}", i);
        }
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

    #[test]
    fn test_string_to_binary() {
        let mut assertion = Vec::<u8>::new();
        for b in "01110100011001010111001101110100".as_bytes() {
            if b.eq(&48) {
                assertion.push(0);
            } else {
                assertion.push(1);

            }
        }

        assert_eq!(
            string_to_binary("test".into()),
            assertion
        );
    }
}
