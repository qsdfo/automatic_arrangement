import theano.tensor as T


def accuracy_measure(true_frame, pred_frame):
    # true_frame must be a binary vector
    vis_on = T.eq(true_frame, 1).nonzero()[0]
    vis_off = T.eq(true_frame, 0).nonzero()[0]

    true_positive = T.sum(pred_frame[vis_on], axis=1)
    false_negative = T.sum(1 - pred_frame[vis_on], axis=1)
    false_positive = T.sum(pred_frame[vis_off], axis=1)

    quotient = true_positive + false_negative + false_positive

    accuracy_measure = T.switch(quotient == 0, 0, true_positive / quotient)

    return accuracy_measure
