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

def backward(cache, w2, y):
    N = len(y)

    dZ2 = cache["Prob"].copy()
    dZ2[np.arange(N), y] -= 1.0
    dZ2 /= N

    dw2 = cache["A1"].T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ w2.T
    dZ1 = dA1 * (cache["Z1"] > 0)

    dw1 = cache["X"].T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dw1, db1, dw2, db2


Prob, cache = forward(X, w1, b1, w2, b2)
loss = getLoss(Prob, y)
dw1, db1, dw2, db2 = backward(cache, w2, y)

print("Loss:", loss)
print("dw1 shape:", dw1.shape, "(matches w1:", dw1.shape == w1.shape, ")")
print("db1 shape:", db1.shape, "(matches b1:", db1.shape == b1.shape, ")")
print("dw2 shape:", dw2.shape, "(matches w2:", dw2.shape == w2.shape, ")")
print("db2 shape:", db2.shape, "(matches b2:", db2.shape == b2.shape, ")")