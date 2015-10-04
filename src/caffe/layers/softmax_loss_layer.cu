#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->usingdata2 = this->usingdata2;
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data;
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data;
  if (this->usingdata2) {
    prob_data = prob_.gpu_data2();
    loss_data = bottom[0]->mutable_gpu_diff2();
  } else {
    prob_data = prob_.gpu_data();
    loss_data = bottom[0]->mutable_gpu_diff();
  }
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts;
  if (this->usingdata2) {
    counts = prob_.mutable_gpu_diff2();
  } else {
    counts = prob_.mutable_gpu_diff();
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  if (this->usingdata2) {
    top[0]->mutable_cpu_data2()[0] = loss;
    top[0]->mutable_cpu_diff2()[0] = (Dtype) 1.0; // TAEHOON LEE : I don't know why but it must be done
  } else {
    top[0]->mutable_cpu_data()[0] = loss;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU2(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>( ( (int)label[n * spatial_dim + s] + 1 ) % 10 );

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff;
    const Dtype* prob_data;
    const Dtype* top_data;
     if (this->adversarial || this->usingdata2) {
      bottom_diff = bottom[0]->mutable_gpu_diff2();
    } else {
      bottom_diff = bottom[0]->mutable_gpu_diff();
    }
    if (this->usingdata2) {
      prob_data = prob_.gpu_data2();
      top_data = top[0]->gpu_data2();
    } else {
      prob_data = prob_.gpu_data();
      top_data = top[0]->gpu_data();
    }
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts;
    if (this->usingdata2) {
      counts = prob_.mutable_gpu_diff2();
    } else {
      counts = prob_.mutable_gpu_diff();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    if (this->adversarial) {
      for (int i=0; i<100; ++i)
        bottom[1]->mutable_cpu_data()[i] = caffe_rng_rand() % 10;
      SoftmaxLossBackwardGPU2<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    } else {
      SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
          outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    }
    //LOG(INFO) << this->adversarial << this->usingdata2 << this->manifold;
    //LOG(INFO) << bottom[1]->cpu_data()[0] << " " << bottom[1]->cpu_data()[1] << " " << bottom[1]->cpu_data()[2] << " " << bottom[1]->cpu_data()[3] << " " << bottom[1]->cpu_data()[4] << " "; 
    Dtype loss_weight;
    if (this->usingdata2) {
      loss_weight = top[0]->cpu_diff2()[0];
    } else {
      loss_weight = top[0]->cpu_diff()[0];
    }
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
