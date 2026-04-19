# Midterm Preparation for the Embedded System Designs
The goal is to prepare live coding midterm for the Embedded System Designe. This course aims to optimize applications in the Jetson Orin Nano board.

First, read through each slide and then make problems. Skip slides that are hard to build example. For example, we learned flash attnetion, but it's something hard for you to really make examples.
Instead, focus on making an example that forces to use different optimizations. There are a few applications we discussed throughout the course:Vector addition, Matrix multiplication, Matrix transpose, Reduce, Prefix-Sum and so on. Skip DNN, flash attention and sequence alignment.
Focus on covering the optimizations the lecture taught. OpenMP, tiling, etc.

Note, I'll be running this via Slurm in the target board. This is a host. Do not try to build and make here.

## Rules
For each problem, you must specify the dimension. Or, you must make clear conditions.
For each problem, you must make your own solution in the `solution/mine`.
For each problem, you must make testcases.
The testcase should be visualized
```sh
TEST01: <description>
SUCCESS 
TEST02:
AVG X.Xms vs. AVG O.Oms
```
The TEST01 always tests correctness. The TEST02 benches the both implementation of mine and your's. Iterate 10 times.
Running a test must be easy. 
`sbatch scritps/test.sh --<problem number|optional>`

Every problem should have a name of `<ID>_<Description>.<cpp|cu>`.


## Folder Strucuture
```sh
|- lecture // Lecture slides
|- logs
|- problem // make problems here
|- solution
    |- mine // I'll copy the 
    |- gold // Add your answer
|- test // Make tests here.
|- scritps
```