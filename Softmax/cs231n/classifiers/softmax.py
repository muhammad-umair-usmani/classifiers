import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - X: A numpy array of shape (N, D) containing a minibatch of data.
      - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
      """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #print X[0].shape #D,(3073,)
    #print X[0].shape[0]
    no_of_train=X.shape[0]
    no_of_classes=W.shape[1]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    for i in range(no_of_train):
        fi=X[i].dot(W)
        #normalize f_i
        fi-=np.max(fi)

        exp_fi=np.exp(fi)
        sum_of_fj=np.sum(exp_fi)
        prob_fi=exp_fi/sum_of_fj
        
        loss += -np.log(prob_fi[y[i]])
        
        for k in range(no_of_classes):
            dW[:,k]+=(prob_fi[k]-(k==y[i]))*X[i]
        
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss/=no_of_train
    loss+=0.5 * reg * np.sum(W * W)
    dW/=no_of_train
    dW+=reg*W
    return loss,dW

def softmax_loss_vectorized(W, X, y, reg):
    
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    no_of_train=X.shape[0]
    no_of_classes=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    fi=X.dot(W)
    #print fi.shape
    #normaliza fi
    fi-=np.max(fi,axis=1,keepdims=True)
    exp_fi=np.exp(fi)
    sum_of_fj=np.sum(exp_fi,axis=1,keepdims=True)
    prob=(exp_fi)/sum_of_fj
    
    loss=np.sum(-np.log(prob[np.arange(no_of_train),y]))
    
    ind=np.zeros_like(prob)
    ind[np.arange(no_of_train), y]=1
    dW=X.T.dot(prob - ind)

    
    
    loss/=no_of_train
    loss+=0.5*reg*np.sum(W*W)
    dW/=no_of_train
    dW+=reg*W
    return loss, dW
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

