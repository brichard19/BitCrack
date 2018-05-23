# BitCrack

A set of tools for brute-forcing Bitcoin private keys. Currently the project requires a CUDA GPU.


## Dependencies

Visual Studio 2015

CUDA Toolkit



## Using the tools

### Usage
```
KeyFinder.exe [OPTIONS] TARGET

Where TARGET is a Bitcoin address

Options:

-d, --device            The device to use
-b, --blocks            Number of blocks
-t, --threads           Threads per block
-p, --per-thread        Keys per thread
-s, --start             Staring key, in hex
-r, --range             Number of keys to search
-c, --compressed        Compressed keys (default)
-u, --uncompressed      Uncompressed keys
```

### Examples


The simplest usage, the keyspace will being at 0, and the CUDA parameters will be chosen automatically
```
KeyFinder.exe 1FshYsUh3mqgsG29XpZ23eLjWV8Ur3VwH
```

To start the search at a specific private key, use the `-s` option

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

Integer values can be specified in decimal (e.g. `123`), or in hexadecimal using the `0x` prefix or `h` suffix (e.g. `0x1234` or `1234h`)


## Choosing the right CUDA parameters

There are 3 parameters that affect performance: blocks, threads per block, and keys per thread.


blocks: Should be a multiple of the number of compute units on the device. The default is 16 times the number of compute units.

threads: This must be a multiple of 32. The default is 256.

Keys per thread: The performance (keys per second) increases asymptotically with this value. The default is 16. Increasing this value will cause the kernel to run longer, but more keys will be processed.


## Supporting this project

If you find this project useful and would like to support it, consider making a donation. Your support is greatly appreciated!

**BTC**: `1LqJ9cHPKxPXDRia4tteTJdLXnisnfHsof`

**LTC**: `LfwqkJY7YDYQWqgR26cg2T1F38YyojD67J`

**ETH**: `0xd28082CD48E1B279425346E8f6C651C45A9023c5`
