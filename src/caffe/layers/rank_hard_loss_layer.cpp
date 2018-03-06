#include "caffe/layers/rank_hard_loss_layer.hpp"

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

int myrandom (int i) { return caffe_rng_rand()%i;}


template <typename Dtype>
void RankHardLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  diff_.ReshapeLike(*bottom[0]);
  // both of dis_ and mask_ have the same shape
  // (batch_size, batch_size, 1, 1)
  // this is actually a matrix of input images
  dis_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
  mask_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
}


template <typename Dtype>
void RankHardLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom)
{

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	int pair_size = rank_param.pair_size();
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	// hard_num is the number of hard negatives
	int hard_num = neg_num * hard_ratio;
	// rand_num is the number of random negatives
	int rand_num = neg_num * rand_ratio;

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();

	// the total number of array elements
	int count = bottom[0]->count();
	// the batch size
	int num = bottom[0]->num();
	// the total number of array elements per batch
	int dim = bottom[0]->count() / bottom[0]->num();

	// reshaped data of dis_ and mask_
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	// initialize both matrix by zero
	for(int i = 0; i < num * num; i ++)
	{
		dis_data[i] = 0;
		mask_data[i] = 0;
	}

	// calculate distance between each element and store in dis_data matrix //

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
			// see paper, maybe, ts = D(X1, X2)
			Dtype ts = 0;
			for(int k = 0; k < dim; k ++)
			{
			  ts += (fea1[k] * fea2[k]) ;
			}
			// upper right half
			dis_data[i * num + j] = -ts;
			// lower left half
			dis_data[j * num + i] = -ts;
		}
	}

	// select negative samples //

	// negative pairs
	vector<pair<float, int> >negpairs;
	vector<int> sid1;
	vector<int> sid2;

	// for each positive pair
	for(int i = 0; i < num; i += pair_size)
	{
		// clear previous items in vectors
		negpairs.clear();
		sid1.clear();
		sid2.clear();

		// for each pair
		for(int j = 0; j < num; j ++)
		{
			// the label is the same, this means i and j are positive
			if(label[j] == label[i])
				continue;

			// see paper, L(Xi, Xi+, Xi-) = max{0, D(Xi, Xi+) - D(Xi, Xi-) + M}
			// i+ = i + 1
			// i- = j
			Dtype tloss = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));

			// skip if the loss is zero(no gradient)
			if(tloss == 0) continue;

			// select this one as a negative sample
			negpairs.push_back(make_pair(dis_data[i * num + j], j));
		}

		// set mask on negative samples
		if(negpairs.size() <= neg_num)
		{
			for(int j = 0; j < negpairs.size(); j ++)
			{
				int id = negpairs[j].second;
				mask_data[i * num + id] = 1;
			}
			continue;
		}

		sort(negpairs.begin(), negpairs.end());

		for(int j = 0; j < neg_num; j ++)
		{
			sid1.push_back(negpairs[j].second);
		}
		for(int j = neg_num; j < negpairs.size(); j ++)
		{
			sid2.push_back(negpairs[j].second);
		}

		std::random_shuffle(sid1.begin(), sid1.end(), myrandom);
		
		for(int j = 0; j < min(hard_num, (int)(sid1.size()) ); j ++)
		{
			mask_data[i * num + sid1[j]] = 1;
		}
		for(int j = hard_num; j < sid1.size(); j ++)
		{
			sid2.push_back(sid1[j]);
		}
		
		std::random_shuffle(sid2.begin(), sid2.end(), myrandom);
		
		for(int j = 0; j < min( rand_num, (int)(sid2.size()) ); j ++)
		{
			mask_data[i * num + sid2[j]] = 1;
		}

	}


}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();      // 4
	int pair_size = rank_param.pair_size();  // 5
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	// calculate distances, and set mask on negative samples
	set_mask(bottom);

	// calculate loss
	Dtype loss = 0;
	// the number of losses, the last 2 means 2 losses(tloss1 and tloss2)
	int cnt = neg_num * num / pair_size * 2;
	// for each positive pair
	for(int i = 0; i < num; i += pair_size)
	{
		// for each sample
		for(int j = 0; j < num; j ++)
		{
			// skip if this is positive
			if(mask_data[i * num + j] == 0) continue;

			// calculate losses
			// see paper, L
			Dtype tloss1 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			// L with the 2nd negative
			Dtype tloss2 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));
			loss += tloss1 + tloss2;
		}
	}

	// loss average
	loss = loss / cnt;
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();

	RankParameter rank_param = this->layer_param_.rank_param();
	int neg_num = rank_param.neg_num();
	int pair_size = rank_param.pair_size();
	float hard_ratio = rank_param.hard_ratio();
	float rand_ratio = rank_param.rand_ratio();
	float margin = rank_param.margin();

	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	// reset differentials
	for(int i = 0; i < count; i ++ )
		bottom_diff[i] = 0;

	// the number of losses
	int cnt = neg_num * num / pair_size * 2;

	// for each positive pair
	for(int i = 0; i < num; i += pair_size)
	{
		// original feature
		const Dtype* fori = bottom_data + i * dim;
		// positive feature
	    const Dtype* fpos = bottom_data + (i + 1) * dim;

	    // original differential
	    Dtype* fori_diff = bottom_diff + i * dim;
	    // positive differential
		Dtype* fpos_diff = bottom_diff + (i + 1) * dim;

		// for each sample
		for(int j = 0; j < num; j ++)
		{
			// skip if i and j are positive
			if(mask_data[i * num + j] == 0) continue;

			// calculate 2 losses
			Dtype tloss1 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[i * num + j] + Dtype(margin));
			Dtype tloss2 = max(Dtype(0), dis_data[i * num + i + 1] - dis_data[(i + 1) * num + j] + Dtype(margin));

			// negative feature
			const Dtype* fneg = bottom_data + j * dim;
			// negative differential
			Dtype* fneg_diff = bottom_diff + j * dim;

			// if each loss is larger than zero, calculate differential respectively
			if(tloss1 > 0)
			{
				for(int k = 0; k < dim; k ++)
			    {
					fori_diff[k] += (fneg[k] - fpos[k]); // / (pairNum * 1.0 - 2.0);
					fpos_diff[k] += -fori[k]; // / (pairNum * 1.0 - 2.0);
					fneg_diff[k] +=  fori[k];
			    }
			}
			if(tloss2 > 0)
			{
				for(int k = 0; k < dim; k ++)
				{
					fori_diff[k] += -fpos[k]; // / (pairNum * 1.0 - 2.0);
				    fpos_diff[k] += fneg[k]-fori[k]; // / (pairNum * 1.0 - 2.0);
				    fneg_diff[k] += fpos[k];
				}
			}

		}
	}

	// set differentials
	for (int i = 0; i < count; i ++)
	{
		bottom_diff[i] = bottom_diff[i] / cnt;
	}

}

#ifdef CPU_ONLY
STUB_GPU(RankHardLossLayer);
#endif

INSTANTIATE_CLASS(RankHardLossLayer);
REGISTER_LAYER_CLASS(RankHardLoss);

}  // namespace caffe
