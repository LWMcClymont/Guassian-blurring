#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>

// OpenCL header
#include <CL/cl.hpp>

// Helper files from tutorials
#include "common.h"
#include "bmpfuncs.h"

#define NUM_ITERATIONS 2000

int main()
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::CommandQueue queue;			// commandqueue for a context and device
	
	// A kernel for each pass
	cl::Kernel verticalPassKernel;
	cl::Kernel horizontalPassKernel;

	// Input and output objects
	unsigned char* inputData;
	unsigned char* outputData;

	// Image info
	int imageWidth, imageHeight, imageSize;

	cl::ImageFormat imageFormat;
	cl::Image2D inputVerticalBuffer, inputHorizontalBuffer, outputVerticalBuffer, outputHorizontalBuffer;

	cl::Event profileEvent;
	cl_ulong timeStart, timeEnd, timeTotal = 0;

	try
	{
		// Select the Device
		if (!select_one_device(&platform, &device))
		{
			quit_program("Device not selected.");
		}

		// Create the context using selected device
		context = cl::Context(device);

		// Build the program
		if (!build_program(&program, &context, "task.cl"))
		{
			quit_program("OpenCL program build error.");
		}

		// Create the kernel 
		verticalPassKernel = cl::Kernel(program, "verticalPass");
		horizontalPassKernel = cl::Kernel(program, "horizontalPass");

		// Create the commandqueue
		queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

		// Read the input file
		inputData = read_BMP_RGB_to_RGBA("peppers.bmp", &imageWidth, &imageHeight);

		// Allocate memory for the outputs
		imageSize = imageWidth * imageHeight * 4;
		outputData = new unsigned char[imageSize];

		// Set the image format
		imageFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8); // CL_UNORM_INT8 = 0.0-1.0

		// Read images from device to host
		cl::size_t<3> origin, region;
		origin[0] = origin[1] = origin[2] = 0;
		region[0] = imageWidth;
		region[1] = imageHeight;
		region[2] = 1;

		cl::NDRange offset(0, 0);
		cl::NDRange globalSize(imageWidth, imageHeight);

		// Create input and output buffers for the vertical pass
		inputVerticalBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)inputData);
		outputVerticalBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)outputData);

		// Set kernal arguments
		verticalPassKernel.setArg(0, inputVerticalBuffer);
		verticalPassKernel.setArg(1, outputVerticalBuffer);

		for (int i = 0; i < NUM_ITERATIONS; i++)
		{
			// Enqueue the kernel
			queue.enqueueNDRangeKernel(verticalPassKernel, offset, globalSize);
			// Read output data from kernel
			queue.enqueueReadImage(outputVerticalBuffer, CL_TRUE, origin, region, 0, 0, outputData);

			// Set the horizontal buffer to use the data from the vertical pass
			inputHorizontalBuffer = cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)outputData);

			// Output buffer for horizontal pass
			outputHorizontalBuffer = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)outputData);

			horizontalPassKernel.setArg(0, inputHorizontalBuffer);
			horizontalPassKernel.setArg(1, outputHorizontalBuffer);

			// Enqueue horizontal buffer
			queue.enqueueNDRangeKernel(horizontalPassKernel, offset, globalSize);

			// Read output from kernel 
			queue.enqueueReadImage(outputHorizontalBuffer, CL_TRUE, origin, region, 0, 0, outputData, NULL, &profileEvent);

			timeStart = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeEnd = profileEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

			timeTotal += timeEnd - timeStart;

		}	
		printf("Average time = %lu\n", timeTotal / NUM_ITERATIONS);

		// Write output to file
		write_BMP_RGBA_to_RGB("Task.bmp", outputData, imageWidth, imageHeight);

		std::cout << "Completed!" << std::endl;

		// Deallocate memory
		free(inputData);
		free(outputData);

	}
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

	std::cout << "\npress a key to quit...";
	std::cin.ignore();

	return 0;
}