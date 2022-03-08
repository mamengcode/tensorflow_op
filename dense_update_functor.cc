#define EIGEN_USE_THREADS

#include "dense_update_functor.h"

#include <tensorflow/core/framework/register_types.h>
#include <tensorflow/core/framework/variant_op_registry.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/platform/mutex.h>
#include <tensorflow/core/platform/types.h>

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace byted_optimizer {
namespace tensorflow {

}  // namespace tensorflow
}  // namespace byted_optimizer