# Sum Reduce

Hierarchical Approach
```c
constexpr max_len = 128;
int rec_reduce(float *numbers, int start, int length){
  int sum = 0;
  if (length < 128){
    for(int i=0; i < length; i++){
      sum += numbers[i];
    }
  }
  else {
    sum += rec_reduce(numbers, start, max_len);
    start += max_len
  }
  return sum;
}

int reduce(float *numbers, int length){
  return rec_reduce(numbers, 0, length);
}
```
- [ ] Implement this in a tail-recursive way for longer length

# Matrix Multiplication
## CPU
- [ ] kij order with mutex
- [ ] Cache way conflict > solve this by padding
- [ ] Use `vaddq_u32`
- [ ] Use various openmp

# Convolution
A single convolution layer with various specs.
Want to parallelize in many dimensions.
- [ ] Depthwise convolution (Give a hint for this)
- [ ] Other fun specialized kernels?
- [ ] im2col + matmul

# Pooling

# FC Layer

# Misc
Think of a meaningful example on this.
- [ ] OMP critical
- [ ] OMP private family
- [ ] OMP hardware thread
- [ ] OMP Master
- [ ] OMP reduction
