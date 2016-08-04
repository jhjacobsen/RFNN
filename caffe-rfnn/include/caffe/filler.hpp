// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <ctime>
#include <string>
#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/anigauss.c"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    const int count = blob->count();
    const Dtype value = this->filler_param_.value();
    CHECK(count);
    for (int i = 0; i < count; ++i) {
      data[i] = value;
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
        Dtype(this->filler_param_.max()), blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};



/** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
 *         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
 */
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    DCHECK(blob->count());
    caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
    // We expect the filler to not be called very frequently, so we will
    // just use a simple implementation
    int dim = blob->count() / blob->num();
    CHECK(dim);
    for (int i = 0; i < blob->num(); ++i) {
      Dtype sum = 0;
      for (int j = 0; j < dim; ++j) {
        sum += data[i * dim + j];
      }
      for (int j = 0; j < dim; ++j) {
        data[i * dim + j] /= sum;
      }
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
 *        set inversely proportional to number of incoming nodes, outgoing
 *        nodes, or their average.
 *
 * A Filler based on the paper [Bengio and Glorot 2010]: Understanding
 * the difficulty of training deep feedforward neuralnetworks.
 *
 * It fills the incoming matrix by randomly sampling uniform data from [-scale,
 * scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
 * average, depending on the variance_norm option. You should make sure the
 * input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
 * = fan_out. Note that this is currently not the case for inner product layers.
 *
 * TODO(dox): make notation in above comment consistent with rest & use LaTeX.
 */
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
 public:
  explicit XavierFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype scale = sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
 *        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
 *        nodes, outgoing nodes, or their average.
 *
 * A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
 * accounts for ReLU nonlinearities.
 *
 * Aside: for another perspective on the scaling factor, see the derivation of
 * [Saxe, McClelland, and Ganguli 2013 (v3)].
 *
 * It fills the incoming matrix by randomly sampling Gaussian data with std =
 * sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
 * the variance_norm option. You should make sure the input blob has shape (num,
 * a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
 * is currently not the case for inner product layers.
 */
template <typename Dtype>
class MSRAFiller : public Filler<Dtype> {
 public:
  explicit MSRAFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype n = fan_in;  // default to fan_in
    if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_AVERAGE) {
      n = (fan_in + fan_out) / Dtype(2);
    } else if (this->filler_param_.variance_norm() ==
        FillerParameter_VarianceNorm_FAN_OUT) {
      n = fan_out;
    }
    Dtype std = sqrt(Dtype(2) / n);
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
        blob->mutable_cpu_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

/*!
@brief Fills a Blob with coefficients for bilinear interpolation.

A common use case is with the DeconvolutionLayer acting as upsampling.
You can upsample a feature map with shape of (B, C, H, W) by any integer factor
using the following proto.
\code
layer {
  name: "upsample", type: "Deconvolution"
  bottom: "{{bottom_name}}" top: "{{top_name}}"
  convolution_param {
    kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
    num_output: {{C}} group: {{C}}
    pad: {{ceil((factor - 1) / 2.)}}
    weight_filler: { type: "bilinear" } bias_term: false
  }
  param { lr_mult: 0 decay_mult: 0 }
}
\endcode
Please use this by replacing `{{}}` with your values. By specifying
`num_output: {{C}} group: {{C}}`, it behaves as
channel-wise convolution. The filter shape of this deconvolution layer will be
(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
interpolation kernel for every channel of the filter identically. The resulting
shape of the top feature map will be (B, C, factor * H, factor * W).
Note that the learning rate and the
weight decay are set to 0 in order to keep coefficient values of bilinear
interpolation unchanged during training. If you apply this to an image, this
operation is equivalent to the following call in Python with Scikit.Image.
\code{.py}
out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
\endcode
 */
template <typename Dtype>
class BilinearFiller : public Filler<Dtype> {
 public:
  explicit BilinearFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
  }
};

//////
//Own Gaussian rand sign
//////



template <typename Dtype>
class SignGaussianFiller : public Filler<Dtype> {
 public:
  explicit SignGaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    int sparse = this->filler_param_.sparse();
    CHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      CHECK_GE(blob->num_axes(), 1);
      const int num_outputs = blob->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
      int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
      caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
      for (int i = 0; i < blob->count(); ++i) {
        std::srand(std::time(0)); // use current time as seed for random generator
        int random_sign = std::rand()%2-1; // this will be either +1 or -1
        data[i] *= random_sign*mask[i];
      }
    }
  }

 protected:
  shared_ptr<SyncedMemory> rand_vec_;
};
////////////////////////////////////////////
///////Gaussian Basis Weight Filler////////
//Uses "Fast Anisotropic Gauss Filtering"// 
//by Geusebroek and Smeulders, 2003////////
///////////////////////////////////////////
/*
Code for: "Structured Receptive Fields in CNNs"
By Joern-Henrik Jacobsen, Jan van Gemert, Zhongyu Lou, Arnold W.M. Smeulders
https://arxiv.org/pdf/1605.02971v2.pdf
Author: J.-H. Jacobsen, Jul. 2016
*/
/*!
@brief Gaussian basis weight filler up to order 4
 */

template <typename Dtype>
class BasisFiller : public Filler<Dtype> {
 public:
  explicit BasisFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    int num_basis  = this->filler_param_.num_basis();
    float sigma = this->filler_param_.sigma();
    Dtype* data    = blob->mutable_cpu_data();
    unsigned xpos = (blob->width()-1)/2;
    unsigned ypos = (blob->height()-1)/2;
    Dtype* impulse = new Dtype[blob->width()*blob->height()];
    memset(impulse, 0., sizeof(Dtype) * blob->width() * blob->height());
    impulse[xpos*blob->width()+ypos] = 1.0;
    int index = 0;
    unsigned channels = blob->count() / (blob->width() * blob->height()); 	
    for (int c=0;c<channels;++c){
	Dtype* filter = new Dtype[blob->width()*blob->height()];
	if (c % num_basis==0)
        {
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,0);
	}else if(c % num_basis == 1){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,0);	
	}else if(c % num_basis == 2){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,1);	
	}else if(c % num_basis == 3){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,1);	
	}else if(c % num_basis == 4){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,2,0);	
	}else if(c % num_basis == 5){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,2);	
	}else if(c % num_basis == 6){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,3,0);	
	}else if(c % num_basis == 7){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,3);	
	}else if(c % num_basis == 8){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,2,1);	
	}else if(c % num_basis == 9){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,2);	
	}else if(c % num_basis == 10){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,4,0);	
	}else if(c % num_basis == 11){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,4);	
	}else if(c % num_basis == 12){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,3,1);	
	}else if(c % num_basis == 13){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,3);	
	}else if(c % num_basis == 14){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,2,2);	
	}else if(c % num_basis == 15){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,0);	
	}else if(c % num_basis == 16){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,1);	
	}else if(c % num_basis == 17){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,1);	
	}else if(c % num_basis == 18){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,2,0);	
	}else if(c % num_basis == 19){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,2);	
	}else if(c % num_basis == 20){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,3,0);	
	}else if(c % num_basis == 21){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,3);	
	}else if(c % num_basis == 22){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,2,1);	
	}else if(c % num_basis == 23){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,2);	
	}else if(c % num_basis == 24){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,4,0);	
	}else if(c % num_basis == 25){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,4);	
	}else if(c % num_basis == 26){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,3,1);	
	}else if(c % num_basis == 27){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,3);	
	}else if(c % num_basis == 28){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,2,2);	
	}else if(c % num_basis == 29){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,0);	
	}else if(c % num_basis == 30){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,1);	
	}else if(c % num_basis == 31){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,1);	
	}else if(c % num_basis == 32){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,2,0);	
	}else if(c % num_basis == 33){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,2);	
	}else if(c % num_basis == 34){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,3,0);	
	}else if(c % num_basis == 35){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,3);	
	}else if(c % num_basis == 36){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,2,1);	
	}else if(c % num_basis == 37){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,2);	
	}else if(c % num_basis == 38){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,4,0);	
	}else if(c % num_basis == 39){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,4);	
	}else if(c % num_basis == 40){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,3,1);	
	}else if(c % num_basis == 41){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,3);	
	}else if(c % num_basis == 42){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,2,2);	
	}else if(c % num_basis == 43){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,0);	
	}else if(c % num_basis == 44){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,1);	
	}else if(c % num_basis == 45){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,1);	
	}else if(c % num_basis == 46){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,2,0);	
	}else if(c % num_basis == 47){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,2);	
	}else if(c % num_basis == 48){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,3,0);	
	}else if(c % num_basis == 49){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,3);	
	}else if(c % num_basis == 50){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,2,1);	
	}else if(c % num_basis == 51){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,2);	
	}else if(c % num_basis == 52){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,4,0);	
	}else if(c % num_basis == 53){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,4);	
	}else if(c % num_basis == 54){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,3,1);	
	}else if(c % num_basis == 55){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,3);	
	}else if(c % num_basis == 56){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,2,2);	
	}
	int pos = 0;
	for(int h=0;h<blob->height();++h){
		for(int w=0;w<blob->width();++w){ 
  			data[index] = filter[pos];
			++pos;
			++index;
		} // loop over width
	} // loop over height
 	delete [] filter;
    } // loop over channels/depth
   CHECK_EQ(this->filler_param_.sparse(), -1)<< "Sparsity not supported by this Filler.";
   // Release the impulse
   delete [] impulse; 
  }
/*
  void saveWeights(Blob<Dtype> *blob){
    	std::string filename  = this->filler_param_.filename();
	std::ofstream out(filename.c_str(),std::ios::out);
    	Dtype* data = blob->mutable_cpu_data();
    	int index = 0;
	    unsigned channels = blob->count() / (blob->width() * blob->height()); 	
    	for (int c=0;c<channels;++c){
		out<<"Weights for channel:"<<c<<std::endl;
		for(int h=0;h<blob->height();++h){
			for(int w=0;w<blob->width();++w){ 
  				out<<data[index]<<" ";
				++index;
			} // loop over width
			out<<std::endl; 	
		} // loop over height
		out<<std::endl;
    	} // loop over channels/depth
   	out.close();
  }
*/
};


/*!
@brief Gaussian basis weight filler up to order 3
 */

template <typename Dtype>
class BasisFiller10 : public Filler<Dtype> {
 public:
  explicit BasisFiller10(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    int num_basis  = this->filler_param_.num_basis();
    float sigma = this->filler_param_.sigma();
    Dtype* data    = blob->mutable_cpu_data();
    unsigned xpos = (blob->width()-1)/2;
    unsigned ypos = (blob->height()-1)/2;
    Dtype* impulse = new Dtype[blob->width()*blob->height()];
    memset(impulse, 0., sizeof(Dtype) * blob->width() * blob->height());
    impulse[xpos*blob->width()+ypos] = 1.0;
    int index = 0;
    unsigned channels = blob->count() / (blob->width() * blob->height()); 	
    for (int c=0;c<channels;++c){
	Dtype* filter = new Dtype[blob->width()*blob->height()];
	if (c % num_basis==0)
        {
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,0);
	}else if(c % num_basis == 1){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,0);	
	}else if(c % num_basis == 2){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,1);	
	}else if(c % num_basis == 3){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,1);	
	}else if(c % num_basis == 4){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,2,0);	
	}else if(c % num_basis == 5){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,2);	
	}else if(c % num_basis == 6){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,3,0);	
	}else if(c % num_basis == 7){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,3);	
	}else if(c % num_basis == 8){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,2,1);	
	}else if(c % num_basis == 9){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,2);	
	}else if(c % num_basis == 10){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,0);	
	}else if(c % num_basis == 11){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,1);	
	}else if(c % num_basis == 12){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,1);	
	}else if(c % num_basis == 13){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,2,0);	
	}else if(c % num_basis == 14){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,2);	
	}else if(c % num_basis == 15){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,3,0);	
	}else if(c % num_basis == 16){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,3);	
	}else if(c % num_basis == 17){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,2,1);	
	}else if(c % num_basis == 18){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,2);	
	}else if(c % num_basis == 19){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,0);	
	}else if(c % num_basis == 20){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,1);	
	}else if(c % num_basis == 21){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,1);	
	}else if(c % num_basis == 22){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,2,0);	
	}else if(c % num_basis == 23){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,2);	
	}else if(c % num_basis == 24){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,3,0);	
	}else if(c % num_basis == 25){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,3);	
	}else if(c % num_basis == 26){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,2,1);	
	}else if(c % num_basis == 27){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,2);	
	}else if(c % num_basis == 28){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,0);	
	}else if(c % num_basis == 29){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,1);	
	}else if(c % num_basis == 30){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,1);	
	}else if(c % num_basis == 31){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,2,0);	
	}else if(c % num_basis == 32){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,2);	
	}else if(c % num_basis == 33){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,3,0);	
	}else if(c % num_basis == 34){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,3);	
	}else if(c % num_basis == 35){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,2,1);	
	}else if(c % num_basis == 36){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,2);	
	}else if(c % num_basis == 37){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,1,0);	
	}else if(c % num_basis == 38){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,0,1);	
	}else if(c % num_basis == 39){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,1,1);	
	}else if(c % num_basis == 40){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,2,0);	
	}else if(c % num_basis == 41){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,0,2);	
	}else if(c % num_basis == 42){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,3,0);	
	}else if(c % num_basis == 43){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,0,3);	
	}else if(c % num_basis == 44){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,2,1);	
	}else if(c % num_basis == 45){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*16,sigma*16,0,1,2);	
	}
	int pos = 0;
	for(int h=0;h<blob->height();++h){
		for(int w=0;w<blob->width();++w){ 
  			data[index] = filter[pos];
			++pos;
			++index;
		} // loop over width
	} // loop over height
 	delete [] filter;
    } // loop over channels/depth
    // this->saveWeights(blob);
   CHECK_EQ(this->filler_param_.sparse(), -1)<< "Sparsity not supported by this Filler.";
   // Release the impulse
   delete [] impulse; 
  }
};
/*!
@brief Gaussian basis weight filler up to order 2
 */

template <typename Dtype>
class BasisFiller6 : public Filler<Dtype> {
 public:
  explicit BasisFiller6(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    int num_basis  = this->filler_param_.num_basis();
    float sigma = this->filler_param_.sigma();
    Dtype* data    = blob->mutable_cpu_data();
    unsigned xpos = (blob->width()-1)/2;
    unsigned ypos = (blob->height()-1)/2;
    Dtype* impulse = new Dtype[blob->width()*blob->height()];
    memset(impulse, 0., sizeof(Dtype) * blob->width() * blob->height());
    impulse[xpos*blob->width()+ypos] = 1.0;
    int index = 0;
    unsigned channels = blob->count() / (blob->width() * blob->height()); 	
    for (int c=0;c<channels;++c){
	Dtype* filter = new Dtype[blob->width()*blob->height()];
	if (c % num_basis==0)
        {
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,0);
	}else if(c % num_basis == 1){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,0);	
	}else if(c % num_basis == 2){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,1);	
	}else if(c % num_basis == 3){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,1);	
	}else if(c % num_basis == 4){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,2,0);	
	}else if(c % num_basis == 5){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,2);	
	}else if(c % num_basis == 6){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,0);	
	}else if(c % num_basis == 7){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,1);	
	}else if(c % num_basis == 8){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,1);	
	}else if(c % num_basis == 9){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,2,0);	
	}else if(c % num_basis == 10){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,2);	
	}else if(c % num_basis == 11){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,0);	
	}else if(c % num_basis == 12){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,1);	
	}else if(c % num_basis == 13){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,1);	
	}else if(c % num_basis == 14){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,2,0);	
	}else if(c % num_basis == 15){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,2);	
	}else if(c % num_basis == 16){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,0);	
	}else if(c % num_basis == 17){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,1);	
	}else if(c % num_basis == 18){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,1);	
	}else if(c % num_basis == 19){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,2,0);	
	}else if(c % num_basis == 20){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,2);	
	}
	int pos = 0;
	for(int h=0;h<blob->height();++h){
		for(int w=0;w<blob->width();++w){ 
  			data[index] = filter[pos];
			++pos;
			++index;
		} // loop over width
	} // loop over height
 	delete [] filter;
    } // loop over channels/depth
    // this->saveWeights(blob);
   CHECK_EQ(this->filler_param_.sparse(), -1)<< "Sparsity not supported by this Filler.";
   // Release the impulse
   delete [] impulse; 
  }
};




/*!
@brief Gaussian basis weight filler up to order 1
 */

template <typename Dtype>
class BasisFiller3 : public Filler<Dtype> {
 public:
  explicit BasisFiller3(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    int num_basis  = this->filler_param_.num_basis();
    float sigma = this->filler_param_.sigma();
    Dtype* data    = blob->mutable_cpu_data();
    unsigned xpos = (blob->width()-1)/2;
    unsigned ypos = (blob->height()-1)/2;
    Dtype* impulse = new Dtype[blob->width()*blob->height()];
    memset(impulse, 0., sizeof(Dtype) * blob->width() * blob->height());
    impulse[xpos*blob->width()+ypos] = 1.0;
    int index = 0;
    unsigned channels = blob->count() / (blob->width() * blob->height()); 	
    for (int c=0;c<channels;++c){
	Dtype* filter = new Dtype[blob->width()*blob->height()];
	if (c % num_basis==0)
        {
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,0);
	}else if(c % num_basis == 1){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,1,0);	
	}else if(c % num_basis == 2){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma,sigma,0,0,1);	
	}else if(c % num_basis == 3){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,1,0);	
	}else if(c % num_basis == 4){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*2,sigma*2,0,0,1);	
	}else if(c % num_basis == 5){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,1,0);	
	}else if(c % num_basis == 6){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*4,sigma*4,0,0,1);	
	}else if(c % num_basis == 7){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,1,0);	
	}else if(c % num_basis == 8){
		anigauss<Dtype>(impulse,filter,blob->width(),blob->height(),sigma*8,sigma*8,0,0,1);	
	}
	int pos = 0;
	for(int h=0;h<blob->height();++h){
		for(int w=0;w<blob->width();++w){ 
  			data[index] = filter[pos];
			++pos;
			++index;
		} // loop over width
	} // loop over height
 	delete [] filter;
    } // loop over channels/depth
    // this->saveWeights(blob);
   CHECK_EQ(this->filler_param_.sparse(), -1)<< "Sparsity not supported by this Filler.";
   // Release the impulse
   delete [] impulse; 
  }
};

//////////////////////////////////////////
//////////////////////////////////////////
/////////////////////////////////////////

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "xavier") {
    return new XavierFiller<Dtype>(param);
  } else if (type == "msra") {
    return new MSRAFiller<Dtype>(param);
  } else if (type == "bilinear") {
    return new BilinearFiller<Dtype>(param);
  } else if (type == "gaussian_basis_4th_order") {
    return new BasisFiller<Dtype>(param);
  } else if (type == "gaussian_basis_3rd_order") {
    return new BasisFiller10<Dtype>(param);
  } else if (type == "gaussian_basis_2nd_order") {
    return new BasisFiller6<Dtype>(param);
  } else if (type == "gaussian_basis_1st_order") {
    return new BasisFiller3<Dtype>(param);
  } else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(NULL);
}
}  // namespace caffe
#endif  // CAFFE_FILLER_HPP_
