import theano.tensor as T


def accuracy_measure(true_frame, pred_frame):
    # true_frame must be a binary vector
    true_positive = T.sum(pred_frame * true_frame, axis=1)
    false_negative = T.sum(1 - (pred_frame * true_frame), axis=1)
    false_positive = T.sum(pred_frame * (1 - true_frame), axis=1)

    quotient = true_positive + false_negative + false_positive

    accuracy_measure = T.switch(T.eq(quotient, 0), 0, true_positive / quotient)

    return accuracy_measure
