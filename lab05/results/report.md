# Lab 5: Frameworks and Automatic Differentiation

## 1. Gradient verification

```text
Our gradients: [42.0, 28.0, 14.0]
PyTorch gradients: [42.0, 28.0, 14.0]
Numerical gradients: [42.000000000896875, 28.000000000361066, 13.99999999946999]
Max difference with PyTorch: 0.0
Max difference with numerical: 8.968754627858289e-10
```
## 2. Training comparison

```text
Runs: 5
Epochs: 100
Architecture: 2 -> 8 -> 1
Custom autograd median time: 11.299519 seconds
PyTorch median time: 0.094133 seconds
Custom average accuracy: 0.9833
PyTorch average accuracy: 0.8833
```
## 3. Eager and traced execution

```text
Eager time: 0.000094196 seconds
Traced time: 0.000139388 seconds
```
## 4. Tensor comparison

```text
Shape: torch.Size([1000, 1000])
Data type: torch.float32
Contiguous tensor: True
Transposed shape: torch.Size([1000, 1000])
Transposed tensor contiguous: False
Contiguous operation time: 0.000777080 seconds
Non-contiguous operation time: 0.000990113 seconds
```
## 5. Conclusion

The custom automatic differentiation engine was implemented and its gradients were compared with PyTorch and numerical estimates. The training experiment compared the custom model with a PyTorch implementation on the make_moons dataset. Eager and traced execution times were measured, and contiguous and non-contiguous tensor operations were compared.
