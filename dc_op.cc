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
	.Attr("use_locking: bool = false")
	.SetShapeFn([](InferenceContext *c){
      c->set_output(0, c->input(0));
      return Status::OK(); });

// namespace byted_optimizer {
// namespace tensorflow {

template <typename Device, typename T>
class DelayCompensationOp : public OpKernel
{
private:
	bool use_exclusive_lock_;

public:
	explicit DelayCompensationOp(OpKernelConstruction *context) : OpKernel(context)
	{
		OP_REQUIRES_OK(context, context->GetAttr("use_locking", &use_exclusive_lock_));
	}

	void Compute(OpKernelContext *context) override
	{
		const bool sparse = false;
		const int nInputs = context->num_inputs();

		Tensor prev_weight;
		OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
									context, 1, use_exclusive_lock_, sparse, &prev_weight));
		// std::cout << "input 1 = " << weight1.flat<T>() << '\n';
		Tensor weight = context->input(2);
		Tensor grad = context->input(0);
		Tensor dc_lambda = context->input(3);
		std::cout << "shape of grad is " << grad.shape() << '\n';
		std::cout << "num of elements of dc_lambda is " << dc_lambda.NumElements() << '\n';

		// Create an output tensor
		Tensor *updated_grad = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, grad.shape(), &updated_grad));
		auto output_flat = updated_grad->flat<T>();
		auto pp_flat = prev_weight.flat<T>();
		auto p_flat = weight.flat<T>();
		auto g_flat = grad.flat<T>();
		typename TTypes<T>::Scalar lambda = dc_lambda.scalar<T>();

		for (int i = 0; i < grad.NumElements(); i++)
		{
			output_flat(i) = g_flat(i) + lambda() * g_flat(i) * g_flat(i) * (p_flat(i) - pp_flat(i));
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
	.Attr("use_locking: bool = false")
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
	bool use_exclusive_lock_;

public:
	explicit DelayCompensationGroupOp(OpKernelConstruction *context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("use_locking", &use_exclusive_lock_));
	}

	void Compute(OpKernelContext *context) override
	{
		const bool sparse = false;
		const int nInputs = context->num_inputs();
		const int nOutputs = context->num_outputs();

		// Get input
		Tensor grad[nOutputs], prev_weight[nOutputs], weight[nOutputs];
		for (int i = 0; i < nOutputs; i++){
			grad[i] = context->input(i);
			OP_REQUIRES_OK(context, GetInputTensorFromVariable<Device, T>(
				context, i + nOutputs, use_exclusive_lock_, sparse, &prev_weight[i]));
			weight[i] = context->input(i + nOutputs * 2);
		}
		Tensor dc_lambda = context->input(nInputs - 1);
		auto lambda_scalar = dc_lambda.scalar<T>();

		// Create and set output
		Tensor *output[nOutputs];
		for (int i = 0; i < nOutputs; ++i){
			// allocate output
			OP_REQUIRES_OK(context, context->allocate_output(i, grad[i].shape(), &output[i]));

			auto out = output[i]->flat<T>();
			auto grad_flat = grad[i].flat<T>();
			auto prev_weight_flat = prev_weight[i].flat<T>();
			auto weight_flat = weight[i].flat<T>();
			for (int j = 0; j < out.size(); ++j){
				out(j) = grad_flat(j) + lambda_scalar() * grad_flat(j) * grad_flat(j) * (weight_flat(j) - prev_weight_flat(j));
				prev_weight_flat(j) = weight_flat(j);
			}
		}
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