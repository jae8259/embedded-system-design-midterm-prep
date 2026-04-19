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
- [ ] Cache way conflict > solve this by padding
- [ ] Use various openmp

# Convolution
- [ ] Other fun specialized kernels?

# Pooling

# FC Layer
