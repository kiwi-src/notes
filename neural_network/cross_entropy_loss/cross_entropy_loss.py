import math
import random

sum = 0.0
def cross_entropy_loss(prob_y1, label):
    prob_y0 = 1 - prob_y1

    # Label == 1
    cross_entropy_class_y1 = label * -math.log(prob_y1)  # p(Y=1|X)

    # Label == 0
    cross_entropy_class_y0 = (1 - label) * -math.log(prob_y0)  # p(Y=0|X)

    loss = cross_entropy_class_y0 + cross_entropy_class_y1
    return loss


def mean_cross_entropy_loss(probs, labels):
    loss = 0
    for prob, label in zip(probs, labels):
        loss += cross_entropy_loss(prob_y1=prob, label=label)
    mean_loss = loss/len(labels)
    return mean_loss


def compute_probs(inputs, labels):
    data = [[x, y] for x, y in zip(inputs, labels)]
    probs = {}
    for x in set(inputs):
        num_occurance_x = sum([1 for e in data if e[0] == x])
        num_occurance_y1_given_x = sum(
            [1 for e in data if e[0] == x and e[1] == 1])
        if num_occurance_y1_given_x == 0:
            probs[x] = 0
        else:
            probs[x] = num_occurance_y1_given_x/num_occurance_x
    return probs


def model(inputs, probs_y1_given_x):
    probs = []
    for x in inputs:
        if x == 0:
            probs.append(probs_y1_given_x[0])
        else:
            probs.append(probs_y1_given_x[1])
    return probs


def lower_bound():
    print('LOWER BOUND')
    best_loss = float('inf')
    best_probs = None
    inputs = [0, 0, 1, 1]
    labels = [0, 1, 0, 1]

    for _ in range(100000):
        probs_y1_given_x = [random.random() for _ in range(2)]
        probs = model(inputs, probs_y1_given_x)
        loss = mean_cross_entropy_loss(probs, labels)

        if loss < best_loss:
            best_loss = loss
            best_probs = probs

    print(f'Best loss = {best_loss}')
    print(f'Best probs = {best_probs}')


def upper_bound():
    print('UPPER BOUND')
    best_loss = float('inf')
    labels = [1, 0, 1, 1]
    for i in range(10000):
        p = i/10000
        if p < 0.00001 or p > 0.99999:
            continue
        probs = [p for _ in range(len(labels))]
        loss = mean_cross_entropy_loss(probs=probs, labels=labels)
        if loss < best_loss:
            best_loss = loss
            best_probs = probs
    print(f'Best Loss = {best_loss}')
    print(f'Best Probs = {best_probs}')


def print_probs():
    print('PRINT PROBS')
    inputs = [1, 1, 1, 0]
    labels = [1, 1, 1, 0]
    probs = compute_probs(inputs, labels)
    print('| X | Y | P(Y=1|X) |')
    print('| - | - | - |')
    for x, y in zip(inputs, labels):
        print(f'| {x} | {y} | {probs[x]:.4f} |')


if __name__ == '__main__':
    #lower_bound()
    #upper_bound()
    #print_probs()

    mean_loss = mean_cross_entropy_loss([3/4, 3/4, 3/4, 3/4], [1, 0, 1, 1])
    print(f'Mean Loss = {mean_loss:.6f}')
