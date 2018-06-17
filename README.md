# BitCrack

A set of tools for brute-forcing Bitcoin private keys. Currently the project requires a CUDA GPU. The main purpose of this project is to contribute to the effort of solving the [Bitcoin puzzle transaction](https://blockchain.info/tx/08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15): A transaction with 32 addresses that become increasingly difficult to crack.

Currently this project is CUDA only, but I would love to bring it to other architectures if there is enough interest in the project.

## Dependencies

Visual Studio 2015

CUDA Toolkit


## Using the tools

### Usage
```
KeyFinder.exe [OPTIONS] [TARGETS]

Where [TARGETS] are one or more Bitcoin address

Options:

-i, --in FILE
    Read addresses from FILE, one address per line. If FILE is "-" then stdin is read.

-o, --out FILE
    Append private keys to FILE, one per line.

-d, --device N
    Use device with ID equal to N. Run CudaInfo.exe to see a list of available devices.

-b, --blocks BLOCKS
    The number of CUDA blocks.

-t, --threads THREADS
    Threads per block.

-p, --per-thread NUMBER
    Each thread will process NUMBER keys at a time.

-s, --start KEY
    Start the search at KEY. KEY is any valid private key in hexadecimal format.

-r, --range RANGE
    Number of keys to search.

-c, --compressed
    Search for compressed keys (default). Can be used with -u to also search uncompressed keys.

-u, --uncompressed
    Search for uncompressed keys. Can be used with -c to search compressed keys.



```

### Examples


The simplest usage, the keyspace will begin at 0, and the CUDA parameters will be chosen automatically
```
KeyFinder.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

Multiple keys can be searched at once with minimal impact to performance. Provide the keys on the command line, or in a file with one address per line
```
KeyFinder.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH 15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP 19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT
```

To start the search at a specific private key, use the `-s` option:

```
KeyFinder.exe -s 6BBF8CCF80F8E184D1D300EF2CE45F7260E56766519C977831678F0000000000 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```


Use the `-b,` `-t` and `-p` options to specify the number of blocks, threads per block, and keys per thread.
```
KeyFinder.exe -b 32 -t 256 -p 16 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

Use the `-r` or `--range` option to specify how many keys to search before stopping. For instance, to search up to 1 billion keys from the starting key:

```
KeyFinder.exe -s 6BBF8CCF80F8E184D1D300EF2CE45F7260E56766519C977831678F0000000000 -r 1000000000
```

Note:

Integer values can be specified in decimal (e.g. `123`) or in hexadecimal using the `0x` prefix or `h` suffix (e.g. `0x1234` or `1234h`)


## Choosing the right CUDA parameters

There are 3 parameters that affect performance: blocks, threads per block, and keys per thread.


`blocks:` Should be a multiple of the number of compute units on the device. The default is 16 times the number of compute units.

`threads:` The number of threads in a block. This must be a multiple of 32. The default is 256.

`Keys per thread:` The performance (keys per second) increases asymptotically with this value. The default is 16. Increasing this value will cause the kernel to run longer, but more keys will be processed.


## Supporting this project

If you find this project useful and would like to support it, consider making a donation. Your support is greatly appreciated!

**BTC**: `1LqJ9cHPKxPXDRia4tteTJdLXnisnfHsof`

**LTC**: `LfwqkJY7YDYQWqgR26cg2T1F38YyojD67J`

**ETH**: `0xd28082CD48E1B279425346E8f6C651C45A9023c5`
