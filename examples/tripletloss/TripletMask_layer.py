import caffe
import numpy as np


class TripletMaskLayer(caffe.Layer):
    """
    Compute matrix shaped a-p mask and a-n mask.
    this layer is forward only.
    """

    def setup(self, bottom, top):
        print('# setup start.')
        print('# setup end.')

    def reshape(self, bottom, top):
        print('# reshape start.')
        
        # Check that i and j are distinct
        self.indices_equal = np.eye(bottom[0].data.shape[0]).astype(np.bool)
        self.indices_not_equal = np.logical_not(self.indices_equal)
        self.labels_equal = np.zeros((bottom[0].data.shape[0], bottom[0].data.shape[0])).astype(np.bool)
        
        
        print('bottom data shape = {}'.format(bottom[0].data.shape))
        print('indices_equal = \n{}'.format(self.indices_equal))
        print('indices_not_equal = \n{}'.format(self.indices_not_equal))
        print('labels_equal = \n{}'.format(self.labels_equal))
        
        # anchor-positive mask output with shape [batch_size, batch_size]
        top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[0])
        print('top[0] data shape = {}'.format(top[0].data.shape))
        
        # anchor-negative mask output with shape [batch_size, batch_size]
        top[1].reshape(bottom[0].data.shape[0], bottom[0].data.shape[0])
        print('top[1] data shape = {}'.format(top[1].data.shape))
      
        print('# reshape end.')

        

    def forward(self, bottom, top):
        """1. make a 2D mask where mask[a, p] is True if a and p are distinct and have same label.
           2. make a 2D mask where mask[a, n] is True if a and n have distinct labels.
        Args:
            labels: np.int32 `ndarray` with shape [batch_size]
        Returns:
            2 masks: np.bool `ndarray` with shape [batch_size, batch_size]
            top[0]: anchor-positive mask.
            top[1]: anchor-negative mask.
        """
        print('# forward start.')
        
        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        self.labels_equal = np.equal(np.expand_dims(bottom[0].data, 0), np.expand_dims(bottom[0].data, 1))
        
        print('labels_equal = \n{}'.format(self.labels_equal))
        
        # Combine the two masks
        top[0].data[...] = np.logical_and(self.indices_not_equal, self.labels_equal)
        top[1].data[...] = np.logical_not(self.labels_equal)
        
        print('a-p mask = \n{}'.format(top[0].data))
        print('a-n mask = \n{}'.format(top[0].data))
        
        print('# forward end.')

    def backward(self, top, propagate_down, bottom):
        # this layer is forward only.
        pass