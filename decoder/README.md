# fuzzy-men (decoder)
Fuzzy-men decoder

## Build instruction
Prepare system

`sudo apt update && sudo apt install gfortran libopenblas-dev`

`yay -S gcc-fortran libopenblas`

Docker build

`docker build -t fuzzymen .`

## References
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42852.pdf

## Todo

- [x] add reference test from ols.py
- [x] implement a CDF and then caclualte p value in OLS
- [x] Compare the result with rappor implementation in github
- [ ] further enhance lasso implementation and OLS implementation
- [ ] Use glmnet like the R implementation 
