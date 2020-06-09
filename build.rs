use std::env;

fn main() {
    let out_dir = env::var("MANIFEST_DIR").unwrap();
    println!("cargo:warning=test");
    panic!();
}
