import caffe
import numpy as np

class PairwiseDistancesLayer(caffe.Layer):
    """
    Compute the Triplet Loss based on the Google's FaceNet paper.
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        print('# reshape start.')
        
        # self.batch_size: number of batch_size. [1]
        self.batch_size = bottom[0].data.shape[0]
        print('batch_size = {}'.format(self.batch_size))
        
        # self.diff: differences are shape of channel [channel]
        self.diff = np.zeros_like(bottom[0].data.shape[1], dtype=np.float32)
        print('diff = {}'.format(self.diff))
        
        # self.dist: distance is scalar [1]
        self.dist = np.zeros(1, dtype=np.float32)
        print('dist = {}'.format(self.dist))
        
        # normalize (# I still keep using this function.)
        self.norm = self.normalize(bottom[0].data)
        print('norm = \n{}'.format(self.norm))
        
        # pairwise distances output with shape [batch_size, batch_size]
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[0])
        print('top[0] data shape = {}'.format(top[0].data.shape))
        
        print('# reshape end.')
        
    def normalize(self, array):
        # ||f(x)||_2=1
        l2 = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
        # avoid to devide by zero
        l2[l2==0] = 1
        return array / l2

    def forward(self, bottom, top):
        """ computes a loss
        Note that the Loss is not averaged by the number of triplet sets.
        Loss = SUM[i->N](Di_pos - Di_neg + margin), 0 <= i <= N(the batch size)
        Dpos = sqrt(L2(IMGi_acr - IMGi_pos))
        Dneg = sqrt(L2(IMGi_acr - IMGi_neg))
        """
        
        print('# forward start.')
        
        for i in range(0, self.batch_size):
            for j in range(0, i):                
                print('i = {}'.format(i))
                print('j = {}'.format(j))
                print('self.norm = \n{}'.format(self.norm[i]))
                
                self.diff = self.norm[i] - self.norm[j]
                print('self.diff = \n{}'.format(self.diff))
                self.dist = np.sum(self.diff**2, axis=0)
                print('self.dist = {}'.format(self.dist))

                top[0].data[i,j] = self.dist
                top[0].data[j,i] = self.dist
                
        print('top[0] data = \n{}'.format(top[0].data))
        print('# forward end.')

    def backward(self, top, propagate_down, bottom): #now under constract.
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