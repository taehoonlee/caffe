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
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
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
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff;
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (this->adversarial) {
      top_diff = top[0]->gpu_diff2();
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff2());
   } else {
      top_diff = top[0]->gpu_diff();
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
          top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff;
    // Gradient with respect to bias
    if (this->adversarial) {
      top_diff = top[0]->gpu_diff2();
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
          bias_multiplier_.gpu_data(), (Dtype)1.,
          this->blobs_[1]->mutable_gpu_diff2());
    } else {
      top_diff = top[0]->gpu_diff();
      caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
          bias_multiplier_.gpu_data(), (Dtype)1.,
          this->blobs_[1]->mutable_gpu_diff());
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff;
    // Gradient with respect to bottom data
    if (this->adversarial) {
      //const Dtype* tmp1 = top[0]->cpu_diff();
      //const Dtype* tmp2 = top[0]->cpu_diff2();
      //LOG_IF(INFO, Caffe::root_solver()) << tmp1[0] << "==" << tmp2[0];


      top_diff = top[0]->gpu_diff2();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
          bottom[0]->mutable_gpu_diff2());


      //tmp1 = bottom[0]->cpu_diff();
      //tmp2 = bottom[0]->cpu_diff2();
      //LOG_IF(INFO, Caffe::root_solver()) << tmp1[0] << "==" << tmp2[0];


    } else {
      top_diff = top[0]->gpu_diff();
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
          top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
          bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
