#define EIGEN_USE_THREADS

// #include "tensorflow/core/framework/common_shape_fns.h"
// #include "tensorflow/core/framework/op_kernel.h"
// #include <tensorflow/core/framework/op.h>
// #include <tensorflow/core/framework/shape_inference.h>
// #include <tensorflow/core/framework/tensor.h>
// #include <tensorflow/core/kernels/variable_ops.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/kernels/training_op_helpers.h>

// #include "training_op_helpers.h"

#include <iostream>

using namespace tensorflow;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


REGISTER_OP("ZeroOutListMutable")
	.Input("ref: N * resource")
	.Input("value: T")
	.Output("output_ref: T")
	.Attr("T: numbertype")
	.Attr("N: int >= 1")
	// .Attr("use_locking: bool = true")
	.SetShapeFn([](InferenceContext *c){
      c->set_output(0, c->input(0));
      return Status::OK(); 
	});

// namespace byted_optimizer {
// namespace tensorflow {

template <typename Device, typename T>
class ZeroOutListMutableOp : public OpKernel
{
private:
	// bool use_exclusive_lock_;
	int32_t N_;

public:
	explicit ZeroOutListMutableOp(OpKernelConstruction *context) : OpKernel(context)
	{
		// OP_REQUIRES_OK(context, context->GetAttr("use_locking", &use_exclusive_lock_));
		// std::cout << "Total number of input is " << context->num_inputs() << "\n";
		context->GetAttr("N", &N_);
		std::cout << "get value of N =" << N_ << '\n';
	}

	void Compute(OpKernelContext *context) override
	{
		int nInputs = context->num_inputs();
		std::cout << "Total number of input is " << nInputs << "\n";

		Tensor weight1, weight2;
		OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
								context, 0, false, false, &weight1));
		std::cout << "input 1 = " << weight1.flat<T>() << '\n';
		auto wf = weight1.flat<T>();
		wf(0) = 0;

		OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
								context, 1, false, false, &weight2));
		std::cout << "input 2 = " << weight2.flat<T>() << '\n';
		auto wf2 = weight2.flat<T>();
		wf2(0) = 0;

		// Tensor ref;
		// for(int i = 0; i < N_; ++i) 
		// {
			// ref = context->mutable_input(0, false);
			// std::cout << "input ref = " << ref.flat<T>() << '\n';
		// 	auto ref_flat = ref.flat<T>();
		// 	ref_flat(0) = 0;
		// 	std::cout << "after assignment, input ref =" << ref_flat << '\n';
		// }

		// const Tensor &val = context->input(1);
		// std::cout << "val = " << val.scalar<T>() << '\n';

		// auto flat_ref = ref.flat<T>();
		// flat_ref(0) = 5;
		// std::cout << "after assignment, input ref = " << ref.flat<T>() << '\n';

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
		Tensor *output_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, weight1.shape(),
														 &output_tensor));
		auto output_flat = output_tensor->flat<T>();
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

#define REGISTER_CPU(T)                                                       \
	REGISTER_KERNEL_BUILDER(                                                  \
		Name("ZeroOutListMutable").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
		ZeroOutListMutableOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// } // namespace tensorflow
// } // namespace byted_optimizer