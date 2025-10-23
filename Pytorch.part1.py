#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.learnpytorch.io/00_pytorch_fundamentals/


# In[2]:


import torch
torch.__version__


# In[3]:


# Scalar
scalar = torch.tensor(7)
scalar


# In[4]:


scalar.ndim


# In[5]:


scalar.item()


# In[6]:


# Vector
vector = torch.tensor([7, 7])
vector


# In[7]:


# Check the number of dimensions of vector
vector.ndim


# In[8]:


# Check shape of vector
vector.shape


# In[9]:


# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
MATRIX


# In[10]:


# Check number of dimensions
MATRIX.ndim


# In[11]:


MATRIX.shape


# In[12]:


# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
TENSOR


# In[13]:


# Check number of dimensions for TENSOR
TENSOR.ndim


# In[14]:


# Check shape of TENSOR
TENSOR.shape # That means there's 1 dimension of 3 by 3.


# In[15]:


# a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector


# In[16]:


import numpy as np
tensor = np.array([
    [7, 4, 0, 1],
    [1, 9, 2, 3],
    [5, 6, 8, 8]
])
print(tensor)


# In[17]:


import torch
tensor = torch.tensor([
    [7, 4, 0, 1],
    [1, 9, 2, 3],
    [5, 6, 8, 8]
])
print(tensor)


# In[18]:


# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
random_tensor, random_tensor.dtype


# In[19]:


# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
random_image_size_tensor.shape, random_image_size_tensor.ndim


# In[ ]:





# In[20]:


# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
zeros, zeros.dtype


# In[21]:


# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones, ones.dtype


# In[22]:


# You can use torch.arange(start, end, step) to do so.

# Where:
#    start = start of range (e.g. 0)
#    end = end of range (e.g. 10)
#    step = how many steps in between each value (e.g. 1)


# In[23]:


# Use torch.arange(), torch.range() is deprecated 
zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten


# In[74]:


print("Tensor datatypes")


# In[24]:


# Tensor datatypes

# The most common type (and generally the default) is torch.float32 or torch.float.
# This is referred to as "32-bit floating point".
# But there's also 16-bit floating point (torch.float16 or torch.half) and 64-bit floating point (torch.float64 or torch.double).
# And to confuse things even more there's also 8-bit, 16-bit, 32-bit and 64-bit integers.


# In[25]:


# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device


# In[26]:


print("Information from tensors")


# In[27]:


# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU


# In[ ]:





# In[28]:


print("Manipulating tensors : tensor operations")


# In[29]:


# Create a tensor of values and add a number to it
tensor = torch.tensor([1, 2, 3])
tensor + 10


# In[30]:


# Multiply it by 10
tensor * 10


# In[31]:


# Tensors don't change unless reassigned
tensor


# In[32]:


# Subtract and reassign
tensor = tensor - 10
tensor


# In[33]:


# Add and reassign
tensor = tensor + 10
tensor


# In[34]:


# Can also use torch functions
torch.multiply(tensor, 10)


# In[35]:


# Original tensor is still unchanged 
tensor


# In[36]:


# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)


# In[37]:


print("Matrix multiplication (is all you need)")


# In[38]:


print('''
The main two rules for matrix multiplication to remember are:

    The inner dimensions must match:

    (3, 2) @ (3, 2) won't work
    (2, 3) @ (3, 2) will work
    (3, 2) @ (2, 3) will work

    The resulting matrix has the shape of the outer dimensions:

    (2, 3) @ (3, 2) -> (2, 2)
    (3, 2) @ (2, 3) -> (3, 3)
''')

# @" in Python is the symbol for matrix multiplication.


# In[39]:


import torch
tensor = torch.tensor([1, 2, 3])
tensor.shape


# In[40]:


print("The difference between element-wise multiplication and matrix multiplication is the addition of values")

# Operation 	Calculation 	Code
# Element-wise multiplication 	[1*1, 2*2, 3*3] = [1, 4, 9] 	tensor * tensor
# Matrix multiplication 	[1*1 + 2*2 + 3*3] = [14] 	tensor.matmul(tensor)

print("tensor:", tensor)


# In[41]:


# Element-wise matrix multiplication
tensor * tensor


# In[42]:


# Matrix multiplication
torch.matmul(tensor, tensor)


# In[ ]:





# In[43]:


print('''
The dot product (also called the scalar product) is one of the most fundamental operations in linear algebra, 
and it comes up everywhere in AI, bioinformatics, and deep learning.
''')

print('''

Let’s go step by step 👇

🔹 Definition

If you have two vectors of the same length:

a=[a1,a2,a3,…,an]
b=[b1,b2,b3,…,bn]

then their dot product is defined as:

a⋅b = a1b1 + a2b2 + a3b3 + … + anbn

It produces a single scalar number, not another vector.

🔹 Example

Let’s take the two vectors you had:


a=[1,2,3]
b=[1,2,3]

Then:

a⋅b=1∗1+2∗2+3∗3=1+4+9=14
a⋅b=1∗1+2∗2+3∗3=1+4+9=14

''')


# In[44]:


# Element-wise matrix multiplication
tensor * tensor

# Matrix multiplication
torch.matmul(tensor, tensor)

# Can also use the "@" symbol for matrix multiplication, though not recommended
tensor @ tensor


# In[45]:


# You can do matrix multiplication by hand but it's not recommended.
# The in-built torch.matmul() method is faster.

get_ipython().run_line_magic('time', '')

# Matrix multiplication by hand 

# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value

get_ipython().run_line_magic('time', 'value = sum(tensor[i] * tensor[i] for i in range(len(tensor)))')


# In[46]:


get_ipython().run_cell_magic('time', '', 'torch.matmul(tensor, tensor)\n')


# In[78]:


import torch, time

# Define two matrices
tensor = torch.tensor([[1., 2.],
                       [3., 4.],
                       [5., 6.]])   # shape (3, 2)

tensor2 = torch.tensor([[1., 2., 3.],
                        [4., 5., 6.]])   # shape (2, 3)

# Measure execution time manually
start = time.time()
result = torch.mm(tensor, tensor2)
end = time.time()

print(result)
print("Result shape:", result.shape)
print(f"Execution time: {(end - start)*1000:.3f} ms")

# torch.matmul(tensor, tensor) is more general: with two 1-D inputs it computes the dot product, so it works and returns tensor(14).
# torch.mm(tensor, tensor) requires both args to be shape (m, n) and (n, p). With a 1-D vector it raises RuntimeError: self must be a matrix.


# In[79]:


# Operator	Function	Input dimensions	Best for	Notes
# @	Python operator for matrix multiplication	Follows the same rules as torch.matmul	General use	Most readable; recommended for most code
# torch.mm(a, b)	Strict matrix × matrix multiply	Both must be 2D: (m × n) @ (n × p)	When you know both tensors are 2D	Slightly faster, lower overhead
# torch.matmul(a, b)	Generalized matrix multiply	Works for 1D, 2D, batched (3D+)	Flexible for vectors, matrices, or batches	Safest for dynamic tensor shapes


# In[47]:


print("Transposition in PyTorch")

print('''
You can perform transposes in PyTorch using either:

    torch.transpose(input, dim0, dim1) - where input is the desired tensor to transpose and dim0 and dim1 are the dimensions to be swapped.
    tensor.T - where tensor is the desired tensor to transpose.
''')


# In[80]:


# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)


# In[81]:


# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)


# In[82]:


# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")


# In[83]:


print("You can also use torch.mm() which is a short for torch.matmul().")
# http://matrixmultiplication.xyz/
print("Note: A matrix multiplication like this is also referred to as the dot product of two matrices.")


# In[84]:


# torch.mm is a shortcut for matmul
torch.mm(tensor_A, tensor_B.T)


# In[85]:


print('''

The torch.nn.Linear() module (we'll see this in action later on), also known as a feed-forward layer 
or fully connected layer, implements a matrix multiplication between an input x and a weights matrix A.

''')


# In[86]:


print('''

A very neural network layer is basically a way to transform an input vector into an output vector — 
using matrix multiplication (a generalization of dot products) plus a bias shift.

So, when we say:

y=x⋅AT+b

we’re describing what happens inside one fully connected (dense) layer.

🔹 Step 1: The ingredients

x → the input vector (features)

Example: if you have 3 input features → 

x=[x1,x2,x3]

A → the weight matrix

Each row (or column, depending on convention) corresponds to the weights for one neuron.

Shape: [output_dim, input_dim]

b → the bias vector

One bias per output neuron.

y → the output vector

Computed as the weighted sum of inputs plus the bias.

''')


# In[87]:


print('''

Matrix multiplication is just a compact way to perform all these dot products at once.

Each neuron = one dot product.

A layer = many neurons.

So we can represent the entire layer as one matrix multiply:

y = torch.matmul(x, A.T) + b

or 

layer = torch.nn.Linear(in_features=3, out_features=2)

y = layer(x)

''')


# In[88]:


print('''
    x is the input to the layer (deep learning is a stack of layers like torch.nn.Linear() and others on top of each other).
    
    A is the weights matrix created by the layer, this starts out as random numbers that get adjusted as a neural network learns to better represent patterns in the data (notice the "T", that's because the weights matrix gets transposed).
    
        Note: You might also often see W or another letter like X used to showcase the weights matrix.
    
    b is the bias term used to slightly offset the weights and inputs.
    
    y is the output (a manipulation of the input in the hopes to discover patterns in it).

''')


# In[89]:


print('''

What Is an Affine Transformation?

An affine transformation is a function that applies a linear transformation (matrix multiplication) plus a translation (bias).

In vector form:

y = Ax + b

where:

x is the input vector

A is a matrix that applies scaling, rotation, shearing, etc.

b is a bias (translation) vector that shifts the result.

''')

print(
    
'''

Difference Between Linear and Affine Transformations

Type	Formula	Includes Translation?	Example

Linear	
y=Ax
y=Ax	❌ No	Rotation, scaling about origin

Affine	
y=Ax+b
y=Ax+b	✅ Yes	Rotation, scaling, and translation

''')


# In[90]:


# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)

# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2,  # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 

x = tensor_A
print(x)

output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")


# In[91]:


print('''

Inside PyTorch’s nn.Linear

When you write:

layer = torch.nn.Linear(in_features=3, out_features=2)

PyTorch automatically:

creates a random weight matrix A of size [2, 3]

creates a bias vector b of size [2]

and computes:

y = x @ A.T + b

''')


# In[92]:


import torch
import torch.nn as nn

# Create a fully connected layer: 3 inputs -> 2 outputs
layer = nn.Linear(in_features=3, out_features=2)
print(layer)

# Print the randomly initialized weights and bias
print("Weights (A):")
print(layer.weight)  # shape [2, 3]
print("\nBias (b):")
print(layer.bias)    # shape [2]

# Example input vector (1 sample, 3 features)
x = torch.tensor([[1.0, 2.0, 3.0]])  # shape [1, 3]
print(x)

# === Method 1: use the layer directly ===
y1 = layer(x)
print("\nOutput from nn.Linear:", y1)


# In[93]:


# === Method 2: manual matrix multiplication ===

A = layer.weight
b = layer.bias
y2 = x @ A.T + b

print("Output from manual computation:", y2)

# Check they match
print("\nDo they match?", torch.allclose(y1, y2))


# In[94]:


print('''

torch.nn helps you:

Define layers (like Linear, Conv2d, LSTM, etc.)
Add activation functions (like ReLU, Sigmoid, Softmax)
Combine layers into a full model using nn.Module
Compute loss functions (like CrossEntropyLoss, MSELoss)
Build custom models with automatic gradient tracking

''')


# In[95]:


print(''' 

Finding the min, max, mean, sum, etc (aggregation)

''')


# In[96]:


# Create a tensor
x = torch.arange(0, 100, 10)
x

# This creates a 1D tensor that goes from 0 up to (but not including) 100, in steps of 10:
# torch.arange(start, end, step) works like Python’s range() 
# but returns a tensor instead of a list.

# If you wanted to create the same tensor manually, it should be:

y = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
print(y)


# In[97]:


print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")


# In[98]:


# You can also do the same as above with torch methods.

torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)


# In[99]:


# Positional min/max

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

# .argmax() means “argument of the maximum” — it returns the index (position) where the maximum value occurs in the tensor.
# .argmin() does the same thing but for the minimum value.


# In[100]:


print('''

Change tensor datatype : a common issue with deep learning operations is having your tensors in different datatypes.

If one tensor is in torch.float64 and another is in torch.float32, you might run into some errors.

float32 uses 4 bytes per number

float64 uses 8 bytes per number

So float64 takes twice as much memory and is slower on most GPUs — but it’s more accurate for scientific or mathematical computations.

Type	Name	Precision	Typical Use

torch.float32	32-bit floating point (a.k.a. single precision)	~7 decimal digits	Default for deep learning (fast & efficient)

torch.float64	64-bit floating point (a.k.a. double precision)	~15–16 decimal digits	Used when high numerical precision is required

''')

# First we'll create a tensor and check its datatype (the default is torch.float32).

print('''

The difference between torch.float32 and torch.float64 is about precision 

(how many digits of accuracy numbers are stored with) and memory usage.

''')

import torch

a32 = torch.tensor([3.1415926535], dtype=torch.float32)
a64 = torch.tensor([3.1415926535], dtype=torch.float64)

print(a32)  # tensor([3.1416])
print(a64)  # tensor([3.14159265])


# In[101]:


# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
tensor.dtype


# In[102]:


# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16


# In[103]:


# Create an int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8


# In[104]:


print('''

Reshaping, stacking, squeezing and unsqueezing :

Because of the rules of matrix multiplication, if you've got shape mismatches, you'll run into errors. 

These methods help you make sure the right elements of your tensors are mixing with the right elements of other tensors

''')


# In[106]:


# Create a tensor
import torch
x = torch.arange(1., 8.)
print(x, x.shape)

# Now let's add an extra dimension with torch.reshape().
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

# torch.arange(1., 8.) creates a 1D tensor with values from 1.0 to 7.0 (the end value 8. is excluded)
# So x = [1., 2., 3., 4., 5., 6., 7.]
# x.shape = torch.Size([7]) means this tensor has 1 dimension with 7 elements.

# .reshape(1, 7) changes the shape of x to have 2 dimensions:
#  - 1 row (often a batch dimension)
#  - 7 columns (the features)
# It doesn’t change the data, just how it’s viewed.

# Now it’s a 2D tensor:
# [[1., 2., 3., 4., 5., 6., 7.]]


# In[107]:


print("Torch shape")


# In[108]:


# We can also change the view with torch.view()

# Change view (keeps same data as original but changes view)
# See more: https://stackoverflow.com/a/54507446/7900723

z = x.view(1, 7)
z, z.shape

# .view() is similar to .reshape(): it changes the shape of a tensor without changing the data.
# The key difference:
# .view() works only on contiguous tensors (tensors stored in continuous memory blocks).
# .reshape() is more flexible — it can handle non-contiguous tensors (PyTorch will copy data if necessary).


# In[109]:


print("Torch arrange")


# In[110]:


print('''

| Operation              | Purpose                        | Output Shape | Notes              |
| ---------------------- | ------------------------------ | ------------ | ------------------ |
| `torch.arange(1., 8.)` | Create tensor `[1, 2, ..., 7]` | `(7,)`       | 1D tensor          |
| `.reshape(1, 7)`       | Add a new dimension (row)      | `(1, 7)`     | Safe and flexible  |
| `.view(1, 7)`          | Change shape (no data copy)    | `(1, 7)`     | Faster but limited |

''')


# In[111]:


# Remember though, changing the view of a tensor with torch.view() really only creates a new view of the same tensor.

x = torch.arange(1., 8.)
print(x)

z = x.view(1, 7)
print(z)

print("Changing the view changes the original tensor too.")

# Changing z changes x
z[:, 0] = 5
z, x

# z[:, 0] means:
# “select all rows (:)”
# “and only the first column (0)”

# So you’re selecting the first element of each row (in this case, just one value — 1.0).

# Because z was created using .view(), it is a view (a shallow copy) of the same data stored in memory as x.
# That means x and z share the same underlying memory


# In[112]:


print("Torch stack")


# In[113]:


# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
x_stacked


# In[114]:


print(''' 

How about removing all single dimensions from a tensor?
To do so you can use torch.squeeze() (I remember this as squeezing the tensor to only have dimensions over 1).

''')


# In[115]:


print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# This is a 2D tensor — it has:
# 1 row
# 7 columns

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# Now it’s a 1D tensor, because you removed one dimension (the outer list-like wrapper).
# Its shape becomes: torch.Size([7])
# A flat array of length 7” (only one dimension).

# Tensor	Dimensionality	Shape	Description
# tensor([[5., 2., 3., 4., 5., 6., 7.]])	2D	[1, 7]	Row vector (2D matrix)
# tensor([5., 2., 3., 4., 5., 6., 7.])


# In[116]:


print('''

And to do the reverse of torch.squeeze() you can use torch.unsqueeze() to add a dimension value of 1 
at a specific index.

''')


# In[117]:


print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

# What .unsqueeze(dim=0) does:

# It inserts a new dimension at position 0 (the first axis).
# The data doesn’t change — only how PyTorch views it.

# So shape [7] becomes [1, 7].

# Now the tensor looks like:
#    [[5., 2., 3., 4., 5., 6., 7.]]

# This is now a 2D tensor — one row and seven columns.


# In[118]:


print("Torch Permute")


# In[119]:


print('''

You can also rearrange the order of axes values with torch.permute(input, dims), 
where the input gets turned into a view with new dims.

''')


# In[120]:


# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")


# In[121]:


print(x_original)

# This creates a random tensor with shape (224, 224, 3).
# torch.rand() → creates random numbers between 0 and 1
# size=(224, 224, 3) → means:
# 224 rows (height)
# 224 columns (width)
# 3 channels (like RGB)

# So you can think of it like an image with shape Height × Width × Channels
# 🖼️ → (H, W, C) = (224, 224, 3)


# In[122]:


print(x_permuted)

# The original order was (0, 1, 2) = (H, W, C)
# You are changing it to (2, 0, 1) = (C, H, W)

# This is extremely common when preparing image data for deep learning models (like CNNs), because:
# PyTorch expects image tensors as (C, H, W) — channels first format.
# But most image libraries (e.g. Pillow, OpenCV, matplotlib) use (H, W, C) — channels last.


# In[123]:


# Meaning:
# You didn’t change the data values, only how PyTorch interprets the axes.
# It’s like rotating the axes labels in memory — not copying or reshaping the data.


# In[124]:


print('''
⚙️ General rule
Function	Description	Example
.reshape()	Changes shape (number of elements per dimension)	(6,) → (2,3)
.unsqueeze()	Adds a new dimension	(7,) → (1,7)
.permute()	Reorders existing dimensions	(H, W, C) → (C, H, W)
''')


# In[125]:


# Let’s look at how to reverse the axis order, 
# i.e. go from (C, H, W) back to (H, W, C) — which is what most image libraries expect for display 
# (e.g. Matplotlib, OpenCV, Pillow)


# In[126]:


# PyTorch models and tensors: (C, H, W) (“channels first”)
# Visualization libraries (like matplotlib.pyplot.imshow): (H, W, C) (“channels last”)


# In[127]:


import matplotlib.pyplot as plt

# Example: show a PyTorch tensor image correctly
img = torch.rand(3, 224, 224)         # Simulate an RGB image
img_hwc = img.permute(1, 2, 0)        # Convert to (H, W, C)

plt.imshow(img_hwc)                   # Works correctly now
plt.axis('off')
plt.show()


# In[128]:


print("PyTorch indexing (selecting data from tensors)")


# In[129]:


# If you've ever done indexing on Python lists or NumPy arrays, 
# indexing in PyTorch with tensors is very similar.


# In[130]:


# Create a tensor 
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape


# In[131]:


print("Indexing values goes OUTER dimension -> INNER dimension (check out the square brackets)")


# In[132]:


x = torch.tensor([
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]
])
print(x)

# Shape = (1, 3, 3)
# → 1 block (0th dimension), each block has 3 rows × 3 columns.


# In[133]:


print('''

🧩 The 0th dimension is the first index you can slice — the “outermost container.”

It’s like the row in a matrix or the image index in a batch.

''')


# In[134]:


# What “0th dimension” means

# In PyTorch (and NumPy), dimensions are numbered starting from 0, just like array indices.

# x.shape = (a, b, c)

# 0th dimension → axis a → the “outermost” dimension
# 1st dimension → axis b → rows inside each outer element
# 2nd dimension → axis c → columns (or elements inside each row)


# In[135]:


# Example 1 — 2D tensor (matrix)

x = torch.tensor([
  [1, 2, 3],
  [4, 5, 6]
])

# Shape = (2, 3)
# 0th dimension → index for rows
# 1st dimension → index for columns

# x[0] → tensor([1, 2, 3])   # first row (0th dimension)
# x[:, 0] → tensor([1, 4])   # first column (1st dimension)

print(x)
print(x.shape)


# In[136]:


# Example 2 — 3D tensor

x = torch.tensor([
  [[1, 2, 3],
   [4, 5, 6]],

  [[7, 8, 9],
   [10, 11, 12]]
])

# Shape = (2, 2, 3)
print(x)
print(x.shape)


# In[137]:


print('''
| Dimension | Meaning      | Example                  |
| --------- | ------------ | ------------------------ |
| **0th**   | which block  | `x[0]` → first 2×3 block |
| **1st**   | which row    | `x[0, 1]` → `[4, 5, 6]`  |
| **2nd**   | which column | `x[0, 1, 2]` → `6`       |
''')


# In[138]:


# Rule of thumb
# The 0th dimension is always the outermost axis — 
# the one you’d iterate over first if you looped through the tensor.

# So if your tensor shape is (batch, channels, height, width) (like in CNNs):
# 0th → batch
# 1st → channels
# 2nd → height
# 3rd → width

img_batch = torch.rand(32, 3, 224, 224)

# Dimension	Meaning
# 0th	batch of 32 images
# 1st	3 color channels (RGB)
# 2nd	224 pixels tall
# 3rd	224 pixels wide

print(img_batch)


# In[139]:


# Create a tensor 
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape


# In[140]:


# Get all values of 0th dimension and the 0 index of 1st dimension
x[:, 0]


# In[141]:


# Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]


# In[142]:


# Get index 0 of 0th and 1st dimension and all values of 2nd dimension 
x[0, 0, :] # same as x[0][0]


# In[143]:


# PyTorch tensors & NumPy
print("PyTorch tensors & NumPy")


# In[144]:


print('''

The two main methods you'll want to use for NumPy to PyTorch (and back again) are:

torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.
torch.Tensor.numpy() - PyTorch tensor -> NumPy array.

''')


# In[145]:


# NumPy array to tensor
import torch
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
array, tensor


# In[146]:


# By default, NumPy arrays are created with the datatype float64 and 
# if you convert it to a PyTorch tensor, it'll keep the same datatype (as above).

# However, many PyTorch calculations default to using float32.
# So if you want to convert your NumPy array (float64) -> PyTorch tensor (float64) 
# -> PyTorch tensor (float32), you can use tensor = torch.from_numpy(array).type(torch.float32).

tensor2 = torch.from_numpy(array).type(torch.float32)
tensor2


# In[147]:


# Because we reassigned tensor above, if you change the tensor, the array stays the same
# Change the array, keep the tensor
array = array + 1
array, tensor


# In[148]:


# And if you want to go from PyTorch tensor to NumPy array, you can call tensor.numpy()

# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
tensor, numpy_tensor


# In[149]:


# And if you want to go from PyTorch tensor to NumPy array, you can call tensor.numpy()

# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
tensor, numpy_tensor


# In[150]:


# And the same rule applies as above, if you change the original tensor, the new numpy_tensor stays the same.

# Change the tensor, keep the array the same
tensor = tensor + 1
tensor, numpy_tensor


# In[151]:


print("Reproducibility (trying to take the random out of random)")


# In[152]:


import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere)")
random_tensor_A == random_tensor_B


# In[153]:


# But what if you wanted to create two random tensors with the same values.
# As in, the tensors would still contain random values but they would be of the same flavour.

# That's where torch.manual_seed(seed) comes in, where seed is an integer (like 42 but it could be anything) 
# that flavours the randomness


# In[154]:


import torch
import random

# Set the random seed
RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
torch.manual_seed(seed=RANDOM_SEED) 
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called 
# Without this, tensor_D would be different to tensor_C 
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
random_tensor_C == random_tensor_D


# In[ ]:





# In[155]:


print("Running tensors on GPUs (and making faster computations)")


# In[156]:


# Getting PyTorch to run on the GPU

# Once you've got a GPU ready to access, the next step is getting PyTorch to use for storing data (tensors) 
# and computing on data (performing operations on tensors).

# To do so, you can use the torch.cuda package


# In[157]:


# Check for GPU
import torch
torch.cuda.is_available()


# In[158]:


get_ipython().system('nvidia-smi')


# In[159]:


import torch

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# Get device name
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    
    # GPU memory usage
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Memory allocated: {mem_allocated:.2f} GB")
    print(f"Memory reserved:  {mem_reserved:.2f} GB")
    print(f"Total GPU memory: {mem_total:.2f} GB")


# In[160]:


# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[161]:


# You can count the number of GPUs PyTorch has access to using 
torch.cuda.device_count()


# In[162]:


print("Putting tensors (and models) on the GPU")

# You can put tensors (and models, we'll see this later) on a specific device by calling to(device) on them.
# Where device is the target device you'd like the tensor (or model) to go to.


# In[163]:


# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu

# If two GPUs were available, they'd be 'cuda:0' and 'cuda:1' respectively, up to 'cuda:n').


# In[164]:


print("Moving tensors back to the CPU")

# What if we wanted to move the tensor back to CPU?
# For example, you'll want to do this if you want to interact with your tensors with NumPy 
# (NumPy does not leverage the GPU).

# Let's try using the torch.Tensor.numpy() method on our tensor_on_gpu

# If tensor is on GPU, can't transform it to NumPy (this will error)
# tensor_on_gpu.numpy()

# Instead, to get a tensor back to CPU and usable with NumPy we can use Tensor.cpu().
# This copies the tensor to CPU memory so it's usable with CPUs.

# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu


# In[165]:


# the original tensor is still on GPU.
tensor_on_gpu


# In[ ]:





# In[166]:


# https://www.learnpytorch.io/02_pytorch_classification/


# In[ ]:




