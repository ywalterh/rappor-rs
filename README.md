# fuzzy-men
Implement simple RAPPOR in both client and server side

![](https://github.com/ywalterh/fuzzy-men/workflows/Rust/badge.svg)

## High level requirement
* Basic RAPPOR implementation in client with rust and WASM
* Basic learning backend implement in rust and some web analytics technologoty
* A good high volume database to test the implementation
* Parameter tuning

## Build instruction

Install and configure wasm-pack:

`curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`

Update npm

`npm install npm@latest -g`

Build the project and then run the dev server

```
wasm-pack build
cd www
npm install
npm run start
```

### Server

`sudo apt update && sudo apt install gfortran libopenblas-dev`

`yay -S gcc-fortran libopenblas`


## Comments
(WH) don't think this is easy at all!!
(DJW) easy stuff is boring Walter

## References

https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42852.pdf

https://www.erikpartridge.com/2019-03/rust-ml-simd-blas-lapack
