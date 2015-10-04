#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data;
  Dtype* top_data;
  if (this->usingdata2) {
    bottom_data = bottom[0]->gpu_data2();
    top_data = top[0]->mutable_gpu_data2();
  } else {
    bottom_data = bottom[0]->gpu_data();
    top_data = top[0]->mutable_gpu_data();
  }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0] && !this->adversarial) {
    const Dtype* top_diff;
    const Dtype* bottom_data;
    Dtype* weights;
    top_diff = top[0]->gpu_diff();
    bottom_data = bottom[0]->gpu_data();
    weights = this->blobs_[0]->mutable_gpu_diff();
    // Gradient with respect to weight
    if (this->manifold) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
            top[0]->gpu_diff2(), bottom[0]->gpu_data(), (Dtype)1., this->blobs_[0]->mutable_gpu_diff2());
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top[0]->gpu_diff3(), bottom[0]->gpu_data2(), (Dtype)1., this->blobs_[0]->mutable_gpu_diff3());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top_diff, bottom_data, (Dtype)1., weights);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1] && !this->adversarial) {
    const Dtype* top_diff;
    Dtype* weights;
    top_diff = top[0]->gpu_diff();
    weights = this->blobs_[1]->mutable_gpu_diff();
    // Gradient with respect to bias
    if (this->manifold) {
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top[0]->gpu_diff2(),
          bias_multiplier_.gpu_data(), (Dtype)1.,
          this->blobs_[1]->mutable_gpu_diff2());
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top[0]->gpu_diff3(),
          bias_multiplier_.gpu_data(), (Dtype)1.,
          this->blobs_[1]->mutable_gpu_diff3());
    } else {
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
          bias_multiplier_.gpu_data(), (Dtype)1.,
          weights);
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff;
    const Dtype* weights;
    Dtype* bottom_diff;
    weights = this->blobs_[0]->gpu_data();
    // Gradient with respect to bottom data
    if (this->manifold) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top[0]->gpu_diff2(), weights, (Dtype)0.,
          bottom[0]->mutable_gpu_diff2());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top[0]->gpu_diff3(), weights, (Dtype)0.,
          bottom[0]->mutable_gpu_diff3());
    } else {
      if (this->adversarial) {
        top_diff = top[0]->gpu_diff2();
        bottom_diff = bottom[0]->mutable_gpu_diff2();
      } else {
        top_diff = top[0]->gpu_diff();
        bottom_diff = bottom[0]->mutable_gpu_diff();
      }
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top_diff, weights, (Dtype)0.,
          bottom_diff);
    }
  } else if (this->adversarial) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top[0]->gpu_diff2(), this->blobs_[0]->gpu_data(), (Dtype)0.,
          bottom[0]->mutable_gpu_diff2());
      Dtype sumsq;
      for (int n=0; n<100; ++n) {
          caffe_gpu_dot(784, bottom[0]->mutable_gpu_diff2() + bottom[0]->offset(n), bottom[0]->mutable_gpu_diff2() + bottom[0]->offset(n), &sumsq);
          sumsq = (Dtype) 1.0 / sqrt(sumsq);
          caffe_gpu_scal(784, sumsq, bottom[0]->mutable_gpu_diff2() + bottom[0]->offset(n));
      }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
