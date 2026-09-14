import numpy as np

N = 2         # Batch size (number of samples)
D_in = 3      # Input features per sample
H = 4         # Neurons in the hidden layer
D_out = 2     # Output classes

X = np.random.randn(N, D_in)
w1 = np.random.randn(D_in, H) * 0.01
b1 = np.zeros((1, H))
w2 = np.random.randn(H, D_out) * 0.01
b2 = np.zeros((1, D_out))
y = np.array([1, 0]) #Real labels

def forward(X, w1, b1, w2, b2):
    Z1 = X @ w1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ w2 + b2

    #print("Shapes are: ",Z1.shape,Z2.shape,sep="\n")

    shifted_Z2 = Z2 - np.max(Z2, axis=1, keepdims=True)
    Prob = np.exp(shifted_Z2)/np.sum(np.exp(shifted_Z2), axis=1, keepdims=True)

    #print("Probabilities:\n", Prob)
    #print("Row sums:", Prob.sum(axis=1))

    cache = {"X": X, "Z1": Z1, "A1": A1, "Prob": Prob} #For carrying values over

    return Prob, cache 

def getLoss(Prob,y):
    correctProb = Prob[np.arange(len(y)), y]
    loss = -(np.mean(np.log(correctProb + 1e-12)))

    #print("True class probabilities:", correctProb)
    #print("Computed Cross-Entropy Loss:", loss)

    return loss

Prob, cache = forward(X, w1, b1, w2, b2)
loss = getLoss(Prob, y)

print("Loss:", loss)
print("Cache:", list(cache.items()))