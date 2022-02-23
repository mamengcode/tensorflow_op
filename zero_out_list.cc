#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>

using namespace tensorflow;

REGISTER_OP("ZeroOutList")
	.Attr("T: list({int32})")
    .Input("to_zero: T")
    .Input("to_zero1: T")
    .Output("zeroed: int32")
    // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    //   c->set_output(0, c->input(0));
    //   return Status::OK();
    // })
    ;


class ZeroOutListOp : public OpKernel {
public:
	explicit ZeroOutListOp(OpKernelConstruction* context) : OpKernel(context) {
		// std::cout << "Total number of input is " << context->num_inputs() << "\n";
	}

	void Compute(OpKernelContext* context) override {
		int nInputs = context->num_inputs();
		std::cout << "Total number of input is " << nInputs << "\n";

		for (int i = 0; i < nInputs; ++i) {
			const Tensor& input_tensor = context->input(i);
			std::cout << "input_tensor(" << i << ") with shape " << input_tensor.shape() << "\n";
			// std::cout << "input_tensor(" << i << ") with value " << input_tensor << "\n";
		}
		// Grab the input tensor
		// const Tensor& input_tensor = context->input(0);
		// std::cout << "input_tensor shape is " << input_tensor.shape() << "\n";
		// auto input = input_tensor.flat<int32>();
		// std::cout << "flat input is " << input << "\n";

		// const Tensor& input_tensor1 = context->input(1);
		// std::cout << "input_tensor1 shape is " << input_tensor1.shape() << "\n";
		// auto input = input_tensor.flat<int32>();

		// Create an output tensor
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(),
		                                                 &output_tensor));
		auto output_flat = output_tensor->flat<int32>();
		output_flat(0) = 155;

		// // Set all but the first element of the output tensor to 0.
		// const int N = input.size();
		// for (int i = 1; i < N; i++) {
	 //  		output_flat(i) = 0;
		// }

		// // Preserve the first input value if possible.
		// if (N > 0) output_flat(0) = input(0);
	}
};


REGISTER_KERNEL_BUILDER(Name("ZeroOutList").Device(DEVICE_CPU), ZeroOutListOp);