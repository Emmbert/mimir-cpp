# Mímir private search C++ implementation

This repo contains benchmarks for Mímir

### Setup:

Clone fhe deck into the same parent directory as this project:

```bash
projects/
├── fhe-deck-core/     ← the cloned library, untouched, its own git repo
└── mimir/             ← your own git repo
    ├── CMakeLists.txt
    ├── include/
    ├── src/
    ├── tests/
    └── benchmarks/
```

If fhe-deck is somewhere else, then change path in CMakeLists.txt accordingly.

Patch a bug in the fhe-deck library. in polynomials.ppt add the modulus reduction in line 36:
`out_cast.m_eval_long[i] = (m_eval_long[i] + other_cast.m_eval_long[i]) % m_modulus;`
Otherwise the addition of ciphertexts in ntt Eval form would fail.

### Build

Run

```bash
cmake -S . -B build
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -- -j$(nproc)
```

Then run executable as wished. For Kamil's simple test:

```bash
./build/kamils_test

# output should be something like:
#======== Performance Test ===========
#Starting the score computation
#Key switching the cluster choice query
#Choosing the Cluster
#Time taken: 61 ms
#Finished
```

### For running the tests:

Reminder: Patch a bug in the fhe-deck library. in polynomials.ppt add the modulus reduction in line 36:
`out_cast.m_eval_long[i] = (m_eval_long[i] + other_cast.m_eval_long[i]) % m_modulus;`
Otherwise the addition of ciphertexts in ntt Eval form would fail.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest --output-on-failure
```

There is an additional test, that shows the time speedup of the NTT evaluation form during the scoring calculations. Therefore run:
```bash
./benchmark_multiplication_speed
```

To run the test with actual mimir parameters, add the parameter file destination, as i.e.:
```bash
MIMIR_TEST_PARAMS_FILE=<parameters_file> ctest --output-on-failure
MIMIR_TEST_PARAMS_FILE=<parameters_file> ./test_lwe_to_rlwe_roundtrip
```

To run the database test with an actual database file:
```bash
MIMIR_TEST_DATABASE_FILE=../cpp_database_files/db_MSMarco_5100_l10_rho2_c2.mdb ./test_real_database_loading
```

### For running the benchmarks

Single-threaded:

```bash
cmake --build . -j
./benchmark_latency <parameters_file> 
```

Multi-threaded on <num_threads> many threads in parallel:

```bash
cmake --build . -j
OMP_NUM_THREADS=<num_threads> ./benchmark_latency_parallel <parameters_file> 
```

Simulation over several worker machines, specified in params.cpp:

```bash
cmake --build . -j
./benchmark_latency_distributed <parameters_file> <number_vms> <desired_cluster_index>
```


### Start Server and Client Applications

The database used by the server is loaded from xxx.

Start the server with:
```bash
./server ../parameter_files/<scheme parameters file> ../cpp_database_files/<database file> 8080```
./server ../parameter_files/test_database_params_with_splits.json ../cpp_database_files/test_db_MSMarco_5100_l10_rho2_c2.mdb 8080```
```

