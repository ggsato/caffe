#include "caffe/layers/val_far_layer.hpp"

#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

using namespace std;
using namespace cv;

namespace caffe {

template <typename Dtype>
void ValFarLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  threshold_ = this->layer_param_.val_far_param().threshold();
}

template <typename Dtype>
void ValFarLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
  top[1]->Reshape(top_shape);
}

template <typename Dtype>
void ValFarLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();

	// the total number of array elements
	int count = bottom[0]->count();
	// the batch size
	int num = bottom[0]->num();
	// the total number of array elements per batch including channels
	int dim = count / num;

	// calculate VAL and FAR
	int positive_pairs = 0;
	int true_positive_pairs = 0;
	int negative_pairs = 0;
	int false_negative_pairs = 0;

	// row
	for(int i = 0; i < num; i ++)
	{
		// column
		for(int j = i + 1; j < num; j ++)
		{
			// features of an input image 1
			const Dtype* fea1 = bottom_data + i * dim;
			// features of an input image 2
			const Dtype* fea2 = bottom_data + j * dim;
			// see paper, ts = 1 - D(X1, X2)
			// note that |f(Xi)|=1 after Norm layer
			Dtype ts = 0;
			for(int k = 0; k < dim; k ++)
			{
			  ts += (fea1[k] * fea2[k]) ;
			}

			float distance = 1 - ts;
			
			// the label is the same, this means i and j are a positive pair
			if(label[j] == label[i])
			{
				positive_pairs += 1;
				if(distance <= threshold_)
				{
					true_positive_pairs += 1;
				}
			}
			else
			{
				negative_pairs += 1;
				if(distance <= threshold_)
				{
					false_negative_pairs += 1;
				}
			}

		}
	}

	// calculate VAR and FAR
	top[0]->mutable_cpu_data()[0] = true_positive_pairs / positive_pairs;
	top[1]->mutable_cpu_data()[0] = false_negative_pairs / negative_pairs;
	
}

INSTANTIATE_CLASS(ValFarLayer);
REGISTER_LAYER_CLASS(ValFar);

}  // namespace caffe
