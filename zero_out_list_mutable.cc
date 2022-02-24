#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "tensorflow/core/framework/common_shape_fns.h"

#include "tensorflow/core/framework/op_kernel.h"

#include <iostream>

using namespace tensorflow;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// REGISTER_OP("ZeroOutListMutable")
// 	// .Attr("N: int = 2")
// 	// .Attr("T: int32")
// 	.Input("to_zero: Ref(int32)")
//     // .Input("to_zero: Ref(N * int32)")
//     // .Input("to_zero1: Ref(N * int32)")
//     .Output("zeroed: int32");
//     // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//     //   c->set_output(0, c->input(0));
//     //   return Status::OK();
//     // })
//     // ;

 REGISTER_OP("ZeroOutListMutable")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](InferenceContext* c) {
      bool validate_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("validate_shape", &validate_shape));
      if (validate_shape) {
        return shape_inference::MergeBothInputsShapeFn(c);
      }

      c->set_output(0, c->input(1));
      return Status::OK();
    });


class ZeroOutListMutableOp : public OpKernel {
private:
	int32_t N_;

public:
	explicit ZeroOutListMutableOp(OpKernelConstruction* context) : OpKernel(context) {
		// std::cout << "Total number of input is " << context->num_inputs() << "\n";
		// context->GetAttr("N", &N_);
		// std::cout << "get value of N " << N_ << '\n';
	}

	void Compute(OpKernelContext* context) override {
		int nInputs = context->num_inputs();
		std::cout << "Total number of input is " << nInputs << "\n";

		// for (int i = 0; i < nInputs; ++i) {
		// 	const Tensor& input_tensor = context->input(i);
		// 	std::cout << "input_tensor(" << i << ") with shape " << input_tensor.shape() << "\n";
		// 	// std::cout << "input_tensor(" << i << ") with value " << input_tensor << "\n";
		// }

		// Test inputlist
		// OpMutableInputList list1;
		// context->mutable_input_list("ref", &list1);
		// std::cout << "list of argument 1 size " << list1.size() << '\n';
		// std::cout << "The shape of tensors in list1 is : ";
		// for(int i = 0; i < list1.size(); ++i) {
		// 	std::cout << list1.at(i, false).shape() << " ";
		// 	auto t = list1.at(i, false);
		// 	auto tf = t.flat<int32>();
		// 	tf(0) = 0;
		// 	std::cout << "set t[0] = 0\n";
		// }
		// std::cout << " that's for tensors in list1\n";

		// OpInputList list2;
		// context->input_list("to_zero1", &list2);
		// std::cout << "list of argument 2 size " << list2.size() << '\n';
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
		OP_REQUIRES_OK(context, context->allocate_output(0, context->input(0).shape(),
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


REGISTER_KERNEL_BUILDER(Name("ZeroOutListMutableOp").Device(DEVICE_CPU), ZeroOutListMutableOp);