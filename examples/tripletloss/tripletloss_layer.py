import caffe
import numpy as np

class TripletLossLayer(caffe.Layer):
    """
    Compute the Triplet Loss based on the Google's FaceNet paper.
    """

    def setup(self, bottom, top):
        # check if input pair is a triplet
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute triplet loss. The bottom length was {}".format(len(bottom)))
            
        params = eval(self.param_str)
        try:
            self.margin = float(params['margin'])
        except:
            self.margin = 1.0
            
        try:
            self.debug = params['debug']
        except:
            self.debug = 0

    def reshape(self, bottom, top):
        # check input shapes match
        if bottom[0].count != bottom[1].count or bottom[1].count != bottom[2].count:
            raise Exception("Inputs must have the same dimension.")
        # differences are shape of inputs
        self.diff_pos = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff_neg = np.zeros_like(bottom[0].data, dtype=np.float32)
        # normalized
        self.norm_anc = bottom[0].data
        self.log('norm_anc = {}'.format(self.norm_anc))
        self.norm_pos = bottom[1].data
        self.log('norm_pos = {}'.format(self.norm_pos))
        self.norm_neg = bottom[2].data
        self.log('norm_neg = {}'.format(self.norm_neg))
        # loss
        self.batch_size = bottom[0].data.shape[0]
        self.log('batch_size = {}'.format(self.batch_size))
        self.loss = np.zeros(self.batch_size, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        """ computes a loss
        Loss = SUM[i->N](Di_pos - Di_neg + margin), 0 <= i <= N(the batch size)
        Dpos = sqrt(L2(IMGi_anc - IMGi_pos))
        Dneg = sqrt(L2(IMGi_anc - IMGi_neg))
        """
        
        self.diff_pos[...] = self.norm_anc - self.norm_pos
        self.log('diff_pos = {}'.format(self.diff_pos))
        self.diff_neg[...] = self.norm_anc - self.norm_neg
        self.log('diff_neg = {}'.format(self.diff_neg))
        dist_pos = np.sum(self.diff_pos**2, axis=1)
        self.log('dist_pos = {}'.format(dist_pos))
        dist_neg = np.sum(self.diff_neg**2, axis=1)
        self.log('dist_neg = {}'.format(dist_neg))
        # calculate a loss for each item
        for i in range(self.batch_size):
            loss = dist_pos[i] - dist_neg[i] + self.margin
            self.log('loss[{}] = {}'.format(i, loss))
            self.loss[i] = max(0, loss)
        total_loss = np.sum(self.loss)
        self.log('total loss = {}, mini_batch_size={}'.format(total_loss, self.batch_size))
        top[0].data[...] = total_loss / self.batch_size

    def backward(self, top, propagate_down, bottom):
        """ computes a gradient w.r.t. each IMG
        dL/dDanc = SUM[i->N]{2(IMGi_neg - IMGi_pos)} if Lossi > 0 else 0
        dL/dDpos = SUM[i->N](-2(IMGi_anc - IMGi_pos)) if Lossi > 0 else 0
        dL/dDneg = SUM[i->N](2(IMGi_anc - IMGi_neg)) if Lossi > 0 else 0
        """
        # gradient w.r.t. Danc
        diff_org = self.norm_neg - self.norm_pos
        for i in range(self.batch_size):
            if self.loss[i] == 0:
                diff_org[i] = 0
        bottom[0].diff[...] = 2 * diff_org
        self.log('anc diff = {}'.format(bottom[0].diff))
        
        # gradient w.r.t. Dpos
        for i in range(self.batch_size):
            self.diff_pos[i] = 0
        bottom[1].diff[...] = -2 * self.diff_pos
        self.log('pos diff = {}'.format(bottom[1].diff))
        
        # gradient w.r.t. Dneg
        for i in range(self.batch_size):
            self.diff_neg[i] = 0
        bottom[2].diff[...] = 2 * self.diff_neg
        self.log('neg diff = {}'.format(bottom[2].diff))
        
    def log(self, message):
        if self.debug == 0:
            return
        
        print(message)
        
class PairwiseDistanceLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute a pair wise distance. The bottom length was {}".format(len(bottom)))
            
        params = eval(self.param_str)
        try:
            self.debug = params['debug']
        except:
            self.debug = 0
        
    def reshape(self, bottom, top):
        # check input shapes match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
            
        self.batch_size = bottom[0].data.shape[0]
        top[0].reshape(1)
            
    def forward(self, bottom, top):
        """ computes a distance
        Dpos = sqrt(L2(IMGi_acr - IMGi_pos))
        """
        dist = np.sum((bottom[0].data - bottom[1].data)**2, axis=1)
        self.log('L2 squared dist = {}'.format(dist))
        top[0].data[...] = np.sum(dist) / self.batch_size
        
    def log(self, message):
        if self.debug == 0:
            return
        
        print(message)
        