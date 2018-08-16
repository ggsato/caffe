import caffe
import numpy as np

class MeanTripletLossLayer(caffe.Layer):
    """
    Compute the Triplet Loss based on the Google's FaceNet paper.
    """

    def setup(self, bottom, top):
        print('# setup start.')
        
        params = eval(self.param_str)
        try:
            self.margin = float(params['margin'])
        except:
            self.margin = 1.0
            
        print('margin = {}'.format(self.margin))
        print('# setup end.')

    def reshape(self, bottom, top):
        print('# reshape start.')
        
        # self.hardest_pos: [batch size]
        self.hardest_pos = np.zeros_like(bottom[0].data, dtype=np.float32)
        print('self.hardest_pos = {}'.format(self.hardest_pos))
        
        # self.hardest_neg: [batch size]
        self.hardest_neg = np.zeros_like(bottom[0].data, dtype=np.float32)
        print('self.hardest_neg = {}'.format(self.hardest_neg))
        
        # self.losses: [batch size]
        self.losses = np.zeros_like(bottom[0].data, dtype=np.float32)
        print('self.losses = {}'.format(self.losses))
        
        # pairwise distances output with shape [batch_size, batch_size]
        top[0].reshape(1)
        print('top[0] data shape = {}'.format(top[0].data.shape))
        
        print('# reshape end.')
        
    def forward(self, bottom, top):
        """ computes a loss
        Note that the Loss is not averaged by the number of triplet sets.
        Loss = SUM[i->N](Di_pos - Di_neg + margin), 0 <= i <= N(the batch size)
        Dpos = sqrt(L2(IMGi_acr - IMGi_pos))
        Dneg = sqrt(L2(IMGi_acr - IMGi_neg))
        """
        print('# forward start.')
        
        self.hardest_pos = bottom[0].data
        print('self.hardest_pos = {}'.format(self.hardest_pos))
        
        self.hardest_neg = bottom[1].data
        print('self.hardest_neg = {}'.format(self.hardest_neg))
        
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        self.losses = np.maximum(self.hardest_pos - self.hardest_neg + self.margin, 0.0)
        print('self.losses = {}'.format(self.losses))

        # Get final mean triplet loss
        top[0].data[...] = np.mean(self.losses)
        print('loss = {}'.format(top[0].data))

        print('# forward end.')

    def backward(self, top, propagate_down, bottom): #now under constraction.
        """ computes a gradient w.r.t. each IMG
        dL/dDorg = SUM[i->N]{2(IMGi_neg - IMGi_pos)} if Lossi > 0 else 0
        dL/dDpos = SUM[i->N](-2(IMGi_anc - IMGi_pos)) if Lossi > 0 else 0
        dL/dDneg = SUM[i->N](2(IMGi_anc - IMGi_neg)) if Lossi > 0 else 0
        """
        pass

#        # gradient w.r.t. Dorg
#        diff_org = self.norm_neg - self.norm_pos
#        for i in range(self.batch_size):
#            if self.loss[i] == 0:
#                diff_org[i] = 0
#        bottom[0].diff[...] = 2 * diff_org
#        print('org diff = {}'.format(bottom[0].diff))
#        
#        # gradient w.r.t. Dpos
#        for i in range(self.batch_size):
#            self.diff_pos[i] = 0
#        bottom[1].diff[...] = -2 * self.diff_pos
#        print('pos diff = {}'.format(bottom[1].diff))
#        
#        # gradient w.r.t. Dneg
#        for i in range(self.batch_size):
#            self.diff_neg[i] = 0
#        bottom[2].diff[...] = 2 * self.diff_neg
#        print('neg diff = {}'.format(bottom[2].diff))