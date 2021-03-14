// Gaussian blurring filter
__constant float filter[7] = { 0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598 };

// Sampler
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
								CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define VERTICAL_PASS 0
#define HORIZONTAL_PASS 1

__kernel void verticalPass (read_only image2d_t inputImage, write_only image2d_t outputImage)
{
	int positionX = get_global_id(0);
	int positionY = get_global_id(1);

	int2 coord = (int2)(0);
	float4 pixel = (float4)(0);
	float4 sum = (float4)(0);

	int filterIndex = 0;

	coord.x = positionX;
	for (int i = -3; i < 3; i++)
	{
		coord.y = positionY + i;
		pixel = read_imagef(inputImage, sampler, coord);
		sum.xyz += pixel.xyz * filter[filterIndex++];
	}

	coord = (int2)(positionX, positionY);
	write_imagef(outputImage, coord, sum);
}

__kernel void horizontalPass (read_only image2d_t inputImage, write_only image2d_t outputImage)
{
	int positionX = get_global_id(0);
	int positionY = get_global_id(1);

	int2 coord = (int2)(0);
	float4 pixel = (float4)(0);
	float4 sum = (float4)(0);

	int filterIndex = 0;

	coord.y = positionY;
	for (int i = -3; i < 3; i++)
	{
		coord.x = positionX + i;
		pixel = read_imagef(inputImage, sampler, coord);
		sum.xyz += pixel.xyz * filter[filterIndex++];
	}

	coord = (int2)(positionX, positionY);
	write_imagef(outputImage, coord, sum);
}
