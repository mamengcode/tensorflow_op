#ifndef OPTIMIZER_DENSE_UPDATE_FUNCTOR_H_
#define OPTIMIZER_DENSE_UPDATE_FUNCTOR_H_

#define EIGEN_USE_THREADS

#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/tensor_types.h>

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using tensorflow::TTypes;

namespace byted_optimizer {
namespace tensorflow {

enum DenseUpdateType { ADD, SUB, ASSIGN };


template <typename Device, typename T, DenseUpdateType OP>
struct DenseUpdate {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update);
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ADD> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, SUB> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

// template <typename Device>
// Status VariantCopyFn(OpKernelContext* context, const Tensor& from, Tensor* to);

// template <>
// Status VariantCopyFn<CPUDevice>(OpKernelContext* context, const Tensor& from,
//                                 Tensor* to);
// template <>
// Status VariantCopyFn<GPUDevice>(OpKernelContext* context, const Tensor& from,
//                                 Tensor* to);

}  // end namespace tensorflow
}  // end namespace byted_optimizer

#endif  // OPTIMIZER_DENSE_UPDATE_FUNCTOR_H_