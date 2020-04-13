extern crate bloom;
// This contains the client implementation of RAPPOR
use bloom::{ASMS, BloomFilter};


pub struct Factory {
    rate: f32 
}


// lack of good way of doing it for now, it would be better to use byte array directly
pub fn string_to_binary(value: String) -> String { let bytes = value.as_bytes();
    let mut result = String::new();
    for byte in bytes {
        let byte_string =  format!("{:b}", byte);
        let added_zero = 8 - byte_string.len();
        for _ in 0..added_zero {
            result.push('0');
        }
        result.push_str(byte_string.as_str());
    }
    return result; 
}

impl Factory {
    fn new(rate: f32) -> Self {
        Factory{rate}
    }

    fn process(&self, value: String) -> String{
        let bits = string_to_binary(value);
        let k = bits.len();
        // step1: hash client's value v onto Bloom filter B of size k using h hash function
        let mut filter = BloomFilter::with_rate(self.rate, k as u32);
        for bit in bits.as_bytes() {
            println!("Adding {}", bit);
            filter.insert(bit);
        }

        return "".into()
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
        assert_eq!(string_to_binary("test".into()), "01110100011001010111001101110100");
    }
}