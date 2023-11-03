# Microsoft/phi-1 Latency Optimization with DirectML
This folder contains a sample use case of Olive to optimize the [Microsoft/phi-1](https://huggingface.co/microsoft/phi-1) model using ONNX conversion and general ONNX performance tuning.

Performs optimization pipeline:

    PyTorch Model -> [Convert to ONNX] -> [Tune performance] -> Optimized ONNX Model

Outputs the best metrics, model, and corresponding Olive config.

# Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

```
# Clone Olive repo and install Olive from source
git clone https://github.com/microsoft/olive
python -m pip install .
```

Once you've installed Olive, install the requirements for this sample matching the version of the library you are using:
```
cd olive/examples/directml/phi_1
pip install -r requirements.txt
```

# Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `microsoft_phi.py` helper script:

```
python microsoft_phi.py --optimize
```

The language model phi-1 is large, and the optimization process is resource intensive. The optimization process can easily take more than 128GB of memory. You can still optimize the model on a machine with less memory, but you'd have to increase your paging file size accordingly and the conversion process will take significantly longer to complete (many hours).

Once the script successfully completes, the optimized ONNX pipeline will be stored under `models/optimized/microsoft/phi-1`.

# Test Inference
Re-running the script with `--inference` 
```
# sample inputs
inputs = tokenizer('''def print_prime(n):
               """
               Print all primes between 1 and n
               """'''
```

```
python .\optimize.py --inference
def print_prime(n):
               """
               Print all primes between 1 and n
               """
               for num in range(2, n+1):
                   for i in range(2, num):
                       if num % i == 0:
                           break
                   else:
                       print(num)

           def is_prime(n):
               """
               Returns True if n is prime, False otherwise
               """
               if n < 2:
                   return False
               for i in range(2, int(n**0.5)+1):
                   if n % i == 0:
                       return False
               return True

           print_prime(20)
           # Output: 2 3 5 7 11 13 17 19
   """
   <YOUR CODE HERE>
```
