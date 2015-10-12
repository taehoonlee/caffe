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
<<<<<<< HEAD
    if (this->bias_term_ && this->param_propagate_down_[1] && !this->adversarial) {
      if (this->manifold) {
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(this->blobs_[1]->mutable_gpu_diff2(), top[i]->gpu_diff2() + top[i]->offset(n));
          this->backward_gpu_bias(this->blobs_[1]->mutable_gpu_diff3(), top[i]->gpu_diff3() + top[i]->offset(n));
        }
      } else {
        for (int n = 0; n < this->num_; ++n) {
          this->backward_gpu_bias(this->blobs_[1]->mutable_gpu_diff(), top_diff + top[i]->offset(n));
        }
=======
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
>>>>>>> BVLC/master
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
<<<<<<< HEAD
        if (this->param_propagate_down_[0] && !this->adversarial) {
          if (this->manifold) {
            this->weight_gpu_gemm(bottom[i]->gpu_data() + bottom[i]->offset(n),
                top[i]->gpu_diff2() + top[i]->offset(n), this->blobs_[0]->mutable_gpu_diff2());
            this->weight_gpu_gemm(bottom[i]->gpu_data2() + bottom[i]->offset(n),
                top[i]->gpu_diff3() + top[i]->offset(n), this->blobs_[0]->mutable_gpu_diff3());
          } else {
            this->weight_gpu_gemm(bottom[i]->gpu_data() + bottom[i]->offset(n),
                top_diff + top[i]->offset(n), this->blobs_[0]->mutable_gpu_diff());
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (this->manifold) {
            this->backward_gpu_gemm(top[i]->gpu_diff2() + top[i]->offset(n), weight,
                bottom[i]->mutable_gpu_diff2() + bottom[i]->offset(n));
            this->backward_gpu_gemm(top[i]->gpu_diff3() + top[i]->offset(n), weight,
                bottom[i]->mutable_gpu_diff3() + bottom[i]->offset(n));
          } else {
            this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
                bottom_diff + bottom[i]->offset(n));
          }
        } else if (this->adversarial) { // TAEHOON : n is the sample index
          this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
          Dtype sumsq;
          caffe_gpu_dot(784, bottom_diff + bottom[i]->offset(n), bottom_diff + bottom[i]->offset(n), &sumsq);
          if ( sumsq < 1e-9 )
            sumsq = (Dtype) 0.0;
          else
            sumsq = (Dtype) 1.0 / sqrt(sumsq);
          caffe_gpu_scal(784, sumsq, bottom_diff + bottom[i]->offset(n));
=======
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
>>>>>>> BVLC/master
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
