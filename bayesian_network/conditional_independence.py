import collections

def compute_probs(data, labels):
    frequency_x1_x2 = collections.defaultdict(int)
    frequency_x1 = collections.defaultdict(int)
    frequency_x2 = collections.defaultdict(int)
    frequency_y = collections.defaultdict(int)

    for features, y in zip(data, labels):
        x1 = features[0]
        x2 = features[1]
        frequency_x1[(x1, y)] += 1
        frequency_x2[(x2, y)] += 1
        frequency_x1_x2[(x1, x2, y)] += 1
        frequency_y[y] += 1

    for features, y in zip(data, labels):
        x1 = features[0]
        x2 = features[1]
        prob_x1_given_y = frequency_x1[(x1, y)] / frequency_y[y]
        prob_x2_given_y = frequency_x2[(x2, y)] / frequency_y[y]
        prob_x1_x2_given_y = frequency_x1_x2[(x1, x2, y)] / frequency_y[y]
        print(f'P(X1={x1}|Y={y})*P(X2={x2}|Y={y})={prob_x1_given_y * prob_x2_given_y}, P(X1,X2|Y)={prob_x1_x2_given_y}')


if __name__ == '__main__':
    print('Conditional independent')
    data = [[0,0], [0,1], [1,0], [1,1]]
    labels = [0, 0, 1, 1]
    compute_probs(data, labels)

    print('Not Conditional independent')
    data = [[1,0], [1,0], [1,0], [0,1]]
    labels = [0, 0, 1, 1]
    compute_probs(data, labels)
