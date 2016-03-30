from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import theano.gradient as G
from Utils.init import shared_normal, shared_zeros
import theano


def maker(n):
    def action(x):
        return x ** n
    return action


def build_network(input_size, hidden_size):
    srng = RandomStreams(seed=12345)

    X = T.fmatrix('X')
    W_input_to_hidden1 = shared_normal(input_size, hidden_size)
    b_hidden1 = shared_zeros(hidden_size)

    hidden1 = T.dot(X, W_input_to_hidden1) + b_hidden1
    hidden1 = hidden1 * (hidden1 > 0)
    hidden1 = hidden1 * srng.binomial(size=(hidden_size,), p=0.5)

    cost = T.sum(hidden1)
    parameters = [W_input_to_hidden1, b_hidden1]

    grad = G.grad(cost,parameters)

    import pdb; pdb.set_trace()
    theano.printing.pydotprint(cost, outfile="hid1.png")


    return X, hidden1, parameters

if __name__ == '__main__':
	a, b, c, d = build_network(200,300)
