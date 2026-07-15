# 1 简易用例

## 1.1 基本编译与运行

场景：Python中的一个tensor的数组（假设是[1, 2, 3, 4]），用自定义c++代码打印该数组。

用例源码：[torch_ext/easyJIT](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/BasicCUDA/tree/master/pytorch/torch_ext/easy_jit) 代码总数**小于20行**。

- 函数实现（[demo.cu](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/BasicCUDA/blob/master/pytorch/torch_ext/easy_jit/demo.cu)）代码如下：

```cpp
#include <torch/extension.h>

void printArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i=0; i <  input.numel(); i++) {
        printf("%d\\n", ptr[i]);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("print_array", &printArray, "");
}
```

- 调用代码（[run.py](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/BasicCUDA/blob/master/pytorch/torch_ext/easy_jit/run.py)）如下

```cpp
import torch
from torch.utils.cpp_extension import load

ext_module = load(name="demo",  sources=["demo.cu"], verbose=True)
print("Module directory: ", ext_module.__file__)
ext_module.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
```

- 输出结果

**问题1：**torch中定义函数/变量/类，在C++中如何使用?

加入“#include<torch/extension.h>”[头文件](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=%E5%A4%B4%E6%96%87%E4%BB%B6&zhida_source=entity)引用，通过它实现对torch c++接口的调用，比如torch中定义的tensor类，创建方式：torch::Tensor。torch/extension.h内容如下，它将几乎所有的torch接口都包进来了。

![image.png](attachment:ac27f11c-fbad-4234-a364-c918e19ad19d:image.png)

接口的使用与查看，可以从[torch源码](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch)查找。在**.h文件**中暴露的接口**，都可以在自定义的c++函数中调用**。

**问题2：**如何将自定义函数转换为python可调用的函数?

采用[pybind11](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=pybind11&zhida_source=entity)工具，调用语法：

![image.png](attachment:078c67f7-0003-4e29-a86a-a92cfc0aeed8:image.png)

**问题3**：如何编译C++代码，生成调用模块?

采用JIT实时[编译器](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=%E7%BC%96%E8%AF%91%E5%99%A8&zhida_source=entity)来完成编译，torch python API提供了一个load方法，可以完成编译与模块加载：

![image.png](attachment:8a501109-38db-4030-bf1e-76e86a00eb80:image.png)

- # 参数name：模块名称，<注意>要与c++中pybind11 定义的保持一致
    
- # 参数sources：需要编译的c++文件
    
- # 参数verbose：打印加载过程详情；

程序执行到[load方法](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=2&q=load%E6%96%B9%E6%B3%95&zhida_source=entity)时，会通过Ninja编译器完成c++代码编译生成 python可调用的so包，并将so加载进来；

[load函数](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=load%E5%87%BD%E6%95%B0&zhida_source=entity)还可以设置其它参数，比如增加头文件的路径“_extra_include_paths_”。其它参数具体解释参看[【链接】](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/cpp_extension.html)。

**问题4**：python中如何调用编译好的模块与函数?

通过torch的load函数编译过后，函数的返回值即为可直接调用的模块，与import一个模块操作相同；比如本例，调用模块中的打印函数：

```cpp
ext_module.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
```

**避免重复编译方式：**首先通过load加载编译模块，然后获取到编译后的so包位置，然后将so包放入代码位置，直接import 这个模块。

先执行：

```python
import torch
from torch.utils.cpp_extension import load

ext_module = load(name="demo",  sources=["demo.cu"], verbose=True)
print("Module directory: ", ext_module.__file__)
```

打印得到so包位置，比如，"/root/.cache/.../demo.so" ， 将so包拷贝到‘[run.py](http://run.py)’文件夹下，直接运行：

```cpp
# run.py
import torch
import demo  # 自定义模块
demo.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
```

## 1.2 内联方式直接调用

在代码量较少时，我们可以用内联加载的方式简化代码书写。推荐两种写法（本质上这两种方式的编译是相同的，只是书写语法有差异），一种用'PYBIND11_MODULE'，另一种用TorchScript。还是用上述场景为例，实现的代码如下：

**方式1([run_inline_v1.py](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/BasicCUDA/blob/master/pytorch/torch_ext/easy_load/run_inline_v1.py))**：

```python
import torch
from torch.utils.cpp_extension import load_inline

cpp_src = """
#include <torch/extension.h>
void printArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i=0; i <  input.numel(); i++) {
        printf("%d\\\\n", ptr[i]);
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("print_array", &printArray, "");
}
"""

ext_module = load_inline(name="print_array", cpp_sources=cpp_src, verbose=True)
ext_module.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
```

**方式2 ([run_inline_v2.py](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/BasicCUDA/blob/master/pytorch/torch_ext/easy_load/run_inline_v2.py))**：

```python
import torch
from torch.utils.cpp_extension import load_inline

cpp_src = """
#include <torch/script.h>

void printArray(torch::Tensor input) {
    int *ptr = (int *)input.data_ptr();
    for(int i=0; i <  input.numel(); i++) {
        printf("%d\\\\n", ptr[i]);
    }
}

static auto registry = torch::RegisterOperators("new_ops::print_array", &printArray);
"""

load_inline(name="print_array", cpp_sources=cpp_src, is_python_module=False, verbose=True)
torch.ops.new_ops.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
```

## 1.3 Setup编译安装

自定义c++，除了运行时进行实时编译，我们还可以用python 里面的setup将模块安装到环境里面。只要自定义的源码不修改，就不需要重复编译的过程。在上述用例easyJIT基础上面，C++代码保持不变，增加一个编译安装过程。

编译安装 ([setup.py](http://setup.py))

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='my_extension',
      ext_modules=[CUDAExtension('my_extension', ['my_extension.cpp']),],
      cmdclass={'build_ext': BuildExtension})
```

```python
import torch
import my_extension

my_extension.print_array(torch.tensor([4, 3, 2, 1], dtype=torch.int))
```

安装与运行

```python
python setup install
python run.py
```

# 2 CUDA算子调用时间

自定义算子常用场景：做一些预研/新算法尝试的时候，发现PyTorch里面没有现成的调用模块，或者已有的调用模块缺少维护，其速度堪忧。一般比较有效的解决方法是采用CUDA API完成一些计算量大的操作。 本例例举了用自定义的CUDA reduce实现对1D数据的求和操作，类似torch.sum功能。

## 2.1 自定义算子介绍

对于一个数组[1.0, 2.34, 3.14, .... n] 的求和，一般而言我们可以采用for循环：

```python
for (int i=1; i < data.size; ++i) {
    rst += data[i];
}
```

这个计算的[时间复杂度](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=%E6%97%B6%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6&zhida_source=entity)为n，所以当数据太大时，计算时间较长。为了降低总耗时，可用CUDA Reduce操作完成求和运算，通过**多线程来对数据进行分块计算；**

## 2.2 接口实现

```cpp
void arraySumCUDA(float *arrData, const int dataSize) {
    int grid = max(dataSize / THREAD_PER_BLOCK, 1);
    arraySumWithSHMKernel<<<grid, THREAD_PER_BLOCK>>>(arrData, dataSize);
}
```

```cpp
#include "sumArray.h"
#include <torch/extension.h>

torch::Tensor torchSumArray(torch::Tensor input) {
    int dataSize = input.numel();
    float* devInData = (float *)input.data_ptr();
    arraySumCUDA(devInData, dataSize);
    return input[0];
}

PYBIND11_MODULE(sum_array, m) {
    m.def("sum_array", &torchSumArray, "");
}
```

该步骤需要完成[数据转换](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%8D%A2&zhida_source=entity)，因为CUDA的kernel ‘不认识’tensor类，所以要将tensor里面的[数据指针](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E6%8C%87%E9%92%88&zhida_source=entity)传递给[kernel函数](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=kernel%E5%87%BD%E6%95%B0&zhida_source=entity)。 其次，再把接口用pybind进行转换到python能调用的接口。

## 2.3 调用与运行

调用的方式和前面的‘easyJIT’案例一样，我们可以直接在模块中编译：

```cpp
from torch.utils.cpp_extension import load

ext_module = load(name="sum_array",
                  extra_include_paths=["./"] ,
                  sources=["sumArray.cu", "glueCode.cpp"],
                  verbose=True)
```

**注意：**自定义算子一般是解决**特定场景**下的运算，可能‘随便写写’都能比原生pytorch的API快，并不意味着自定义的算子能替代原生API。 要做到算子的完整替换，就得考虑**通用场景**、算子的入参**边界值、类型，**以及**接口校验**等。但当这些附加功能都加上时，速度又不一定比原生的API快。

# 3 自定义功能模块

一般而言，在PyTorch的计算函数编写时，我们需要定义函数的前向（forward）和后向（backward）计算。这里拿torch官方的LSTM改编的例子来看，要完成前、后向算子自定义需要哪些工作。

## 3.1 算子介绍

在[自然语言处理](https://zhida.zhihu.com/search?content_id=216887928&content_type=Article&match_order=1&q=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86&zhida_source=entity)中有一种算子叫做LSTM（Long Short Term Memory），假设现在我们要把它的遗忘门**Tanh**替换成ELU（Exponential Linear Unit），由于ELU不遗忘，这样LSTM就成了LLTM（Long Long Term Memory），计算的模块图如下

![image.png](attachment:a888925f-4652-4c13-8519-2a9e5915e9dc:image.png)

如果要用PyTorch API写，只要将LSTM模块的tanh计算替换成elu，内容如下：

```python
class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)
        gate_weights = F.linear(X, self.weights, self.bias)
        gates = gate_weights.chunk(3, dim=1)
        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        candidate_cell = F.elu(gates[2]) # 此处用记忆门elu替换了遗忘门tranh运算。
        new_cell = old_cell + candidate_cell * input_gate
        new_h = torch.tanh(new_cell) * output_gate
        return new_h, new_cell

```

## 3.2 功能实现

功能实现的步骤与‘sumArray’例子的基本一样，不同的是需要我们实现**前向**和**后向**函数。

**步骤1：**在cuda中实现LLTM的运算

将LLTM的功能模块翻译成CUDA kernel，代码：

此处不展开介绍

## 3.3 运行与测试