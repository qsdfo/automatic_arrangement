import theano.tensor as T


def precision_measure(true_frame, pred_frame):
    # true_frame must be a binary vector
    vis_on = T.eq(true_frame, 1).nonzero()[0]
    true_positive = T.sum(pred_frame[vis_on], axis=1)

    vis_off = T.eq(true_frame, 0).nonzero()[0]
    false_positive = T.sum(pred_frame[vis_off], axis=1)

    quotient = true_positive + false_positive

    prediction_measure = T.switch(quotient == 0, 0, true_positive / quotient)

    return prediction_measure
