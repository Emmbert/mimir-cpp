# Mímir private search C++ implementation

This repo contains benchmarks for Mímir. The code needs AVX-support to be run in appropriate time. To reconstruct the
benchmarks of Mímir, the database loading, and server and clienet implementation are not needed. Also the tests are not
important to verify the results from the paper.

## Setup:

Clone
our [fork https://github.com/Emmbert/fhe-deck-core/tree/mimir-dependency](https://github.com/Emmbert/fhe-deck-core/tree/mimir-dependency)
of the FHE-Deck library into the same parent directory as this project and switch to branch `mimir-dependency`:

```bash
projects/
├── fhe-deck-core/     ← the cloned fhe-deck-core fork, branch mimir-dependencies
└── mimir-cpp/
    ├── CMakeLists.txt
    ├── include/
    ├── src/
    ├── tests/
    └── benchmarks/
```

If fhe-deck is somewhere else, then change path in CMakeLists.txt accordingly.

## Build

Run

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . -j
```

## Running the benchmarks

Although this takes a **really long** time, for all parameter sets the tests can be run before. For shorter testing
time, it might make sense to only run the `ctest -L benchmarking` tests, (although they will still take a long time).

If the single-threaded or multi-threaded benchmarks fail (because of RAM shortage), default to run the `*_pool.cpp`
version, which is memory friendly. This option simplification was not needed for Mímir's final parameter sets.

To run all benchmarks for a parameter file at once do this. The results will be written to a file saved in a created
folder mimirXXX/ in your working directory (/scripts).

```bash
cd scripts
./run_benchmarks.sh --paramfile "../parameter_files/mimirI.json" --threads 32 --numservers 10
```

Else you can call the benchmarks seperately:
Single-threaded:

```bash
cd build
./benchmark_latency <parameters_file> 
```

(On our servers I run `krenew -- nice -n1 ./benchmark_latency_seeded` to have higher cpu priority ;-) )

Multi-threaded on <num_threads> many threads in parallel:

```bash
cd build
OMP_NUM_THREADS=<num_threads> ./benchmark_latency_parallel <parameters_file> 
```

Simulation over several worker machines, specified in params.cpp:

```bash
cd build
./benchmark_latency_distributed <parameters_file> <number_vms> <desired_cluster_index>
```

## Running the tests:

Running all tests with the default values takes approx 10 minutes. Running the tests with the real parameters may take a
whole hour, so best, run only a subset of tests (see later note on this).

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
ctest --output-on-failure
```

To run the test with actual mimir parameters, add the parameter file destination, as i.e.:

```bash
MIMIR_TEST_PARAMS_FILE=<parameters_file> ctest --output-on-failure
MIMIR_TEST_PARAMS_FILE=<parameters_file> < test_executable> # i.e. ./test_lwe_to_rlwe_roundtrip
```

There is an additional test, that shows the time speedup of the NTT evaluation form during the scoring calculations.
Therefore run:

```bash
./benchmark_multiplication_speed
```

Also there are two benchmarks, that verify, that query generation, and RLWE ad RGSW switching are precise, by
benchmarking the time it takes to generally encrypt random LWE messages or switch LWE to RLWE and RGSW messages:

```bash
OMP_NUM_THREADS=1 ./benchmark_switch_isolated ../parameter_files/<params_file>  # this should take as long as 'RLWE switching' and 'RGSW switching' in 'benchmark_latency_seeded'
OMP_NUM_THREADS=1 ./benchmark_lwe_encrypt_count ../parameter_files/<params_file>  # this should take as long as 'RLWE switching' and 'RGSW switching' in 'benchmark_latency_seeded'
```

To run the database test with an actual database file:

```bash
MIMIR_TEST_DATABASE_FILE=../cpp_database_files/db_MSMarco_5100_l10_rho2_c2.mdb ./test_real_database_loading
```

To run just a subgroup of all tests:

```bash
ctest -L crypto          # only the crypto correctness tests
ctest -L benchmarking    # only the protocol correctness tests
ctest -L server_client   # only the server/database/client tests
ctest -LE crypto         # everything except crypto (same set as above here, but useful in general)
ctest -L crypto -L database    # NOTE: this is AND-of-neither-quite -- see caveat below
ctest --print-labels     # confirm both labels registered correctly
```

## Start Server and Client Applications

Disclaimer: This is work in progress. The database used by the server is loaded from `cpp_database_files/`.

Start the server with:

```bash
./server ../parameter_files/<scheme parameters file> ../cpp_database_files/<database file> 8080```
./server ../parameter_files/test_database_params_with_splits.json ../cpp_database_files/test_db_MSMarco_5100_l10_rho2_c2.mdb 8080```
```

Run clint queries from seperate process:

```bash
./client ../parameter_files/<scheme parameters file> 127.0.0.1 8080
./client ../parameter_files/test_database_params_with_splits.json 127.0.0.1 8080
```
