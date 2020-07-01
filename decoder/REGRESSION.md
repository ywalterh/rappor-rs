# fuzzy-men regression suite (decoder)
Fuzzy-men decoder regression

## Grab RAPPOR and generate a test result
https://github.com/google/rappor

Might need tweak from setup.sh, talk to Walter if you need help
```
./setup.sh
./build.sh
./demo.sh quick-python
```

`tests/gen_true_values.R exp 100 100000 10 64 /tmp/test-cases.csv`
This generates 1 mil test cases in that csv file

## Todo
- [ ] Run the test case againts our implementation
