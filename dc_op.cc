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

REGISTER_OP("DelayCompensation")
	.Input("grad: T")
	.Input("prev_weight: resource")
	.Input("weight: T")
	.Input("lambda: float")
	.Output("updated_grad: T")
	.Attr("T: numbertype")
	// .Attr("use_locking: bool = true")
	.SetShapeFn([](InferenceContext *c)
				{
      c->set_output(0, c->input(0));
      return Status::OK(); });

// namespace byted_optimizer {
// namespace tensorflow {

template <typename Device, typename T>
class DelayCompensationOp : public OpKernel
{
private:
	// bool use_exclusive_lock_;

public:
	explicit DelayCompensationOp(OpKernelConstruction *context) : OpKernel(context)
	{
		// context->GetAttr("lambda", &dc_lambda_);
		// std::cout << "get value of lambda =" << dc_lambda_ << '\n';
	}

	void Compute(OpKernelContext *context) override
	{
		int nInputs = context->num_inputs();
		std::cout << "Total number of input is " << nInputs << "\n";

		Tensor prev_weight;
		OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
									context, 1, false, false, &prev_weight));
		// std::cout << "input 1 = " << weight1.flat<T>() << '\n';
		Tensor weight = context->input(2);
		Tensor grad = context->input(0);
		Tensor dc_lambda = context->input(3);
		std::cout << "shape of grad is " << grad.shape() << '\n';
		std::cout << "num of elements of dc_lambda is " << dc_lambda.NumElements() << '\n';

		// Create an output tensor
		Tensor *updated_grad = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, grad.shape(),
														 &updated_grad));
		auto output_flat = updated_grad->flat<T>();
		auto pp_flat = prev_weight.flat<T>();
		auto p_flat = weight.flat<T>();
		auto g_flat = grad.flat<T>();
		typename TTypes<T>::Scalar lambda = dc_lambda.scalar<T>();

		for (int i = 0; i < grad.NumElements(); i++)
		{
			output_flat(i) = lambda() * g_flat(i) * (p_flat(i) - pp_flat(i));
			pp_flat(i) = p_flat(i);
		}
	}
};

#define REGISTER_CPU(T)                                                      \
	REGISTER_KERNEL_BUILDER(                                                 \
		Name("DelayCompensation").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
		DelayCompensationOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// ======================================================================================
REGISTER_OP("DelayCompensationGroup")
	.Input("grad: N * T")
	.Input("prev_weight: N * resource")
	.Input("weight: N * T")
	.Input("lambda: float")
	.Output("updated_grad: N * T")
	.Attr("T: numbertype")
	.Attr("N: int >= 1")
	// .Attr("use_locking: bool = true")
	.SetShapeFn([](InferenceContext *c)
				{
		int nOutput = c->num_outputs();
		for(int i = 0; i < nOutput; ++i){
			c->set_output(i, c->input(i));
		}
      	return Status::OK(); });

template <typename Device, typename T>
class DelayCompensationGroupOp : public OpKernel
{
private:
	// bool use_exclusive_lock_;

public:
	explicit DelayCompensationGroupOp(OpKernelConstruction *context) : OpKernel(context)
	{
		// context->GetAttr("lambda", &dc_lambda_);
		// std::cout << "get value of lambda =" << dc_lambda_ << '\n';
	}

	void Compute(OpKernelContext *context) override
	{
		int nInputs = context->num_inputs();
		std::cout << "Total number of input is " << nInputs << "\n";

		const int nOutputs = context->num_outputs();
		std::cout << "Total number of ouput is " << nOutputs << "\n";

		Tensor grad0 = context->input(0);
		auto grad0_flat = grad0.flat<T>();
		for (int i = 0; i < grad0_flat.size(); i++){
			std::cout << grad0_flat(i) << '\n';
		}

		// Tensor prev_weight;
		// OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
		// 						context, 1, false, false, &prev_weight));
		// // std::cout << "input 1 = " << weight1.flat<T>() << '\n';
		// Tensor weight = context->input(2);
		// Tensor grad = context->input(0);
		// Tensor dc_lambda = context->input(3);
		// std::cout << "shape of grad is " << grad.shape() << '\n';
		// std::cout << "num of elements of dc_lambda is " << dc_lambda.NumElements() << '\n';

		// Create an output tensor
		Tensor *output[nOutputs];
		for (int i = 0; i < nOutputs; ++i){
			OP_REQUIRES_OK(context, context->allocate_output(i, grad0.shape(), &output[i]));
			auto out = output[i]->flat<T>();
			for (int j = 0; j < grad0.NumElements(); ++j){
				out(j) = j + i * 10;
			}
		}
		// auto output_flat = updated_grad->flat<T>();
		// auto pp_flat = prev_weight.flat<T>();
		// auto p_flat = weight.flat<T>();
		// auto g_flat = grad.flat<T>();
		// typename TTypes<T>::Scalar lambda = dc_lambda.scalar<T>();

		// for (int i = 0; i < grad.NumElements(); i++){
		//     output_flat(i) = lambda() * g_flat(i) * (p_flat(i) - pp_flat(i));
		//     pp_flat(i) = p_flat(i);
		// }
	}
};

#define REGISTER_CPU_GROUP(T)                                                     \
	REGISTER_KERNEL_BUILDER(                                                      \
		Name("DelayCompensationGroup").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
		DelayCompensationGroupOp<CPUDevice, T>);
REGISTER_CPU_GROUP(float);
REGISTER_CPU_GROUP(int32);

// } // namespace tensorflow
// } // namespace byted_optimizer