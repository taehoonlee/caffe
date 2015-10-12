#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data;
    Dtype* top_data;
    if (this->usingdata2) {
      bottom_data = bottom[i]->gpu_data2();
      top_data = top[i]->mutable_gpu_data2();
    } else {
      bottom_data = bottom[i]->gpu_data();
      top_data = top[i]->mutable_gpu_data();
    }
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff;
    Dtype* bottom_diff;
    if (this->adversarial) {
      top_diff = top[i]->gpu_diff2();
      bottom_diff = bottom[i]->mutable_gpu_diff2();
    } else {
      top_diff = top[i]->gpu_diff();
      bottom_diff = bottom[i]->mutable_gpu_diff();
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1] && !this->adversarial) {
      if (this->manifold) {
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(this->blobs_[1]->mutable_gpu_diff2(), top[i]->gpu_diff2() + n * this->top_dim_);
          this->backward_gpu_bias(this->blobs_[1]->mutable_gpu_diff3(), top[i]->gpu_diff3() + n * this->top_dim_);
        }
      } else {
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(this->blobs_[1]->mutable_gpu_diff(), top_diff + n * this->top_dim_);
        }
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0] && !this->adversarial) {
          if (this->manifold) {
            this->weight_gpu_gemm(bottom[i]->gpu_data() + n * this->bottom_dim_,
                top[i]->gpu_diff2() + n * this->top_dim_, this->blobs_[0]->mutable_gpu_diff2());
            this->weight_gpu_gemm(bottom[i]->gpu_data2() + n * this->bottom_dim_,
                top[i]->gpu_diff3() + n * this->top_dim_, this->blobs_[0]->mutable_gpu_diff3());
          } else {
            this->weight_gpu_gemm(bottom[i]->gpu_data() + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, this->blobs_[0]->mutable_gpu_diff());
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (this->manifold) {
            this->backward_gpu_gemm(top[i]->gpu_diff2() + n * this->top_dim_, weight,
                bottom[i]->mutable_gpu_diff2() + n * this->bottom_dim_);
            this->backward_gpu_gemm(top[i]->gpu_diff3() + n * this->top_dim_, weight,
                bottom[i]->mutable_gpu_diff3() + n * this->bottom_dim_);
          } else {
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + bottom[i]->offset(n));
          }
        } else if (this->adversarial) { // TAEHOON : n is the sample index
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
          Dtype sumsq;
          caffe_gpu_dot(784, bottom_diff + n * this->bottom_dim_, bottom_diff + n * this->bottom_dim_, &sumsq);
          if ( sumsq < 1e-9 )
            sumsq = (Dtype) 0.0;
          else
            sumsq = (Dtype) 1.0 / sqrt(sumsq);
          caffe_gpu_scal(784, sumsq, bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
