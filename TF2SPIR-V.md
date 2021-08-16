# Step 1
Clone Tensorflow form https://github.com/tensorflow/tensorflow  
We can dump kernel LLVM IR from `https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc`  

In `CpuCompiler::RunBackend`, we add the following code to dump IR when op run in XLA JIT

```
ir_string = llvm_ir::DumpModuleToString(*llvm_module);
std::ofstream out("kernel.ll");
out << ir_string;
out.close();
```
after  
```
string ir_string;
if (embed_ir_in_executable) {
  ir_string = llvm_ir::DumpModuleToString(*llvm_module);
}
```

# Step 2
Build Tensorflow with offical tutorial  
Install Tensorflow with pip

# Step 3
Run any tensorflow operation in python with XLA, e.g., tf.nn.conv2d  
Kernel IR will be dumped to current directory  

# Step 4
Build spirv-llvm-translator with https://github.com/KhronosGroup/SPIRV-LLVM-Translator   

The translator can be built as a regular LLVM subproject. To do that you need to clone it into the llvm/projects or llvm/tools directory.
```
git clone https://github.com/llvm/llvm-project.git
cd llvm-project/llvm/projects
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
```
Run (or re-run) cmake as usual for LLVM. After that you should have llvm-spirv and check-llvm-spirv targets available.
```
mkdir llvm-project/build && cd llvm-project/build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang"
make llvm-spirv -j`nproc`
```

# Step 5
Install LLVM-13 with this tutorial https://apt.llvm.org/  

Translate kernel.ll from step 3 to bitcode  
`llvm-as kernel.ll -o kernel.bc`  

Translate kernel.bc to .spv using llvm-spirv from step 4  
`llvm-spirv -spirv-max-version 1.3 kernel.bc`

# Step 6
Use `test_spirv.cpp` to test kernel  
```c++
#include <iostream>
#include <vector>
#include <string>
#include <CL/cl2.hpp>

#include <fstream>

using namespace std;


static std::vector<cl_uchar> readSPIRVFromFile(
    const std::string& filename )
{
    std::ifstream is(filename, std::ios::binary);
    std::vector<cl_uchar> ret;
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return ret;
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    ret.reserve(filesize);
    ret.insert(
        ret.begin(),
        std::istreambuf_iterator<char>(is),
        std::istreambuf_iterator<char>() );

    return ret;
}

int main( int argc , char **argv ) {
	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty()) {
		printf("No platforms!\n");		
		return 1;
	}

	std::cout << "Platform number: " << platforms.size() <<"\n";
	
	cl::Platform platform = platforms[0];
	std::vector<cl::Device> Devices;

	std::cout << "Platform : " << platform.getInfo<CL_PLATFORM_NAME>() <<"\n";
	std::cout << "Platform version: " << platform.getInfo<CL_PLATFORM_VERSION>() <<"\n";

	platform.getDevices(CL_DEVICE_TYPE_GPU, &Devices);
	if (Devices.empty()) {
		printf("No Devices!\n");
		return 1;
	}

	std::cout << "Device number: " << Devices.size() <<"\n";
	cl::Device device = Devices[0];
	std::cout << "Device : " << device.getInfo<CL_DEVICE_NAME>() <<"\n";

	cl::Context context({device});
	std::string fileName("kernel.spv");
	printf("Reading SPIR-V from file: %s\n", fileName.c_str());
    std::vector<cl_uchar> spirv = readSPIRVFromFile(fileName);
	cl_program clprogram = nullptr;
	clprogram = clCreateProgramWithIL(context(), spirv.data(), spirv.size(), nullptr);
	cl::Program program({clprogram});

	if (program.build({device}) != CL_SUCCESS) {
        printf("Fail to build\n");
        return 1;
    }
    


	return 0;
}
```
Export VSI library path and build `test_spirv.cpp`  
In our case:  
```
export LIBRARY_PATH=$LIBRARY_PATH:/home/BossVSim/sdk/vsimulator/lib/
g++ test_spirv.cpp -o test_spirv -std=c++11 -lOpenCL -I/home/BossVSim/sdk/vsimulator/include
```

Finally run `./test_spirv`
