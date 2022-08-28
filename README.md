# Epi4Tensor

<p>
  <a href="https://doi.org/10.1145/3545008.3545066" alt="Publication">
    <img src="http://img.shields.io/badge/DOI-10.1145/3545008.3545066-blue.svg"/></a>

</p>

*Epi4Tensor* is an implementation on modern NVIDIA GPU microarchitectures of an algorithm for efficient fourth-order exhaustive epistasis detection, a bioinformatics application that searches for associations between single nucleotide polymorphisms (SNPs) and a given observable trait.
NVIDIA GPUs with third generation tensor cores (Ampere microarchitecture) are efficiently exploited through the extensive use of AND+POPC 1-bit precision operations, while compatibility with second generation tensor cores (Turing microarchitecture) is achieved relying on XOR+POPC 1-bit precision operations.
This software has been subject to optimization on systems with A100 (Ampere/SM80) and Titan RTX (Turing/SM75) GPUs.
However, it should also work well with many other Turing and Ampere GPUs.

## Software and hardware requirements

* CUDA Toolkit 11.4 and CUTLASS 2.5 (more recent)
* NVIDIA Turing (SM75) or Ampere (SM80/SM86) GPU

## Compilation and usage instructions

Download and uncompress the CUTLASS library to your preferred location, reflecting it in the Makefile. Compile the tool using the `make` command.

To test the produced binary you can pass one of the provided sample datasets (<a href="https://drive.google.com/file/d/10Rvoc-qNyvRrEE4yJO9Yh7bEOQatC7T3/view?usp=sharing">download from here</a>) as input, e.g:

```bash
$ ./bin/epi4tensor datasets/db_256snps_131072samples.txt
```

The tool has multi-GPU support. If your system has multiple compatible GPUs, you can customize the tool by setting `NUM_GPUS` in the Makefile to a number that represents your particular configuration.

## In papers and reports, please refer to this tool as follows

Ricardo Nobre, Aleksandar Ilic, Sergio Santander-Jiménez, and Leonel Sousa. 2022. Tensor-Accelerated Fourth-Order Epistasis Detection on GPUs. In 51st International Conference on Parallel Processing (ICPP ’22), August 29-September 1, 2022, Bordeaux, France. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3545008.3545066
