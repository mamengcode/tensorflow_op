#define EIGEN_USE_THREADS

#include "rmsprop_v2_op.h"

#include "../common/rmsprop_v2.h"
#include "training_op_helpers.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
// using tensorflow::Tensor;

using CPUDevice = Eigen::ThreadPoolDevice;

// using ::byted_optimizer::common::RMSPropV2;
// using ::byted_optimizer::common::RMSPropV2Config;
// using ::byted_optimizer::common::CPU;

namespace byted_optimizer { 
namespace tensorflow {

template <bool is_resource>
static Status ApplyRMSPropV2ShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape<is_resource>(c, 0);  // weights
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape<is_resource>(c, 1), &s));  // vs
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));     // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));     // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));     // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));     // weight decay
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, is_resource>(
          c, 6 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyRMSPropV2")
    .Input("weights: Ref(T)")
    .Input("vs: Ref(T)")
    .Input("lr: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("weight_decay: T")
    .Input("grads: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyRMSPropV2ShapeFn</*is_resource=*/false>);

REGISTER_OP("ResourceApplyRMSPropV2")
    .Input("weights: resource")
    .Input("vs: resource")
    .Input("lr: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("weight_decay: T")
    .Input("grads: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ApplyRMSPropV2ShapeFn</*is_resource=*/true>);

template <typename T>
struct ApplyRMSPropV2<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat weights,
                  typename TTypes<T>::Flat vs,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstScalar weight_decay,
                  typename TTypes<T>::ConstFlat grads) {
    size_t length = weights.size();

    T* weights_ptr = weights.data();
    T* vs_ptr = vs.data();
    const T* g_ptr = grads.data();

    RMSPropV2Config config;
    config.beta2 = *(beta2.data());
    config.epsilon = *(epsilon.data());
    config.weight_decay = *(weight_decay.data());
    float nan_inf_found;

    RMSPropV2::CPUOptimFn<T>(length, &nan_inf_found, g_ptr, vs_ptr, weights_ptr, &config, *(lr.data()));
  }
};

template <typename Device, typename T>
class ApplyRMSPropV2Op : public OpKernel {
 public:
  explicit ApplyRMSPropV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor weights;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &weights));
    Tensor vs;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &vs));
    OP_REQUIRES(
        ctx, weights.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, vs.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(2);
    const Tensor& beta2 = ctx->input(3);
    const Tensor& epsilon = ctx->input(4);
    const Tensor& weight_decay = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(weight_decay.shape()),
                errors::InvalidArgument("weight_decay is not a scalar: ",
                                        weight_decay.shape().DebugString()));

    const Tensor& grads = ctx->input(6);
    OP_REQUIRES(ctx, weights.shape().IsSameSize(vs.shape()),
                errors::InvalidArgument("weights and vs do not have the same shape",
                                        weights.shape().DebugString(), " ",
                                        vs.shape().DebugString()));
    OP_REQUIRES(
        ctx, weights.shape().IsSameSize(grads.shape()),
        errors::InvalidArgument("weights and grads do not have the same shape",
                                weights.shape().DebugString(), " ",
                                grads.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    ApplyRMSPropV2<Device, T>()(
        device, weights.flat<T>(), vs.flat<T>(), 
        lr.scalar<T>(), beta2.scalar<T>(), 
        epsilon.scalar<T>(), weight_decay.scalar<T>(),
        grads.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyRMSPropV2").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyRMSPropV2Op<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyRMSPropV2")                \
                              .HostMemory("weights")                   \
                              .HostMemory("vs")                     \
                              .Device(DEVICE_##D)                  \
                              .TypeConstraint<T>("T"),             \
                          ApplyRMSPropV2Op<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T)                                   \
  template <>                                                 \
  void ApplyRMSPropV2<GPUDevice, T>::operator()(                 \
      const GPUDevice& d, typename TTypes<T>::Flat weights,   \
      typename TTypes<T>::Flat vs,                            \
      typename TTypes<T>::ConstScalar lr,                     \
      typename TTypes<T>::ConstScalar beta2,                  \
      typename TTypes<T>::ConstScalar epsilon,                \
      typename TTypes<T>::ConstScalar weight_decay,           \
      typename TTypes<T>::ConstFlat grads); \
  extern template struct ApplyRMSPropV2<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace tensorflow
}  // namespace byted_optimizer
