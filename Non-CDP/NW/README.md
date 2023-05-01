# Needleman_Wunsch_GPU

# Overview

This code implements the Needleman-Wunsch algorithm for exact string matching.

# Requirements

To compile, nvcc is required. 

# Instructions

To compile:

```
make
```

To run:

```
./nw [flags]
```

Optional flags:

```
  -N <N>    specifies the size of the strings to match

  -0        run GPU version 0
  -1        run GPU version 1
  -2        run GPU version 2
  -3        run GPU version 3
            NOTE: It is okay to specify multiple different GPU versions in the
                  same run. By default, only the CPU version is run.
```
