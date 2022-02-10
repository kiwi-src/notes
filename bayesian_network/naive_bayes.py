import collections

def compute_probs(data, labels):
    frequency_x1_x2_y = collections.defaultdict(int)
    frequency_x1_y = collections.defaultdict(int)
    frequency_x2_y = collections.defaultdict(int)
    frequency_y = collections.defaultdict(int)
    frequency_x1 = collections.defaultdict(int)
    frequency_x2 = collections.defaultdict(int)

    for features, y in zip(data, labels):
        x1 = features[0]
        x2 = features[1]
        frequency_x1_y[(x1, y)] += 1
        frequency_x2_y[(x2, y)] += 1
        frequency_x1_x2_y[(x1, x2, y)] += 1
        frequency_y[y] += 1
        frequency_x1[x1] += 1
        frequency_x2[x2] += 1

    for features, y in zip(data, labels):
        x1 = features[0]
        x2 = features[1]

        # P(X1|Y)
        prob_x1_given_y = frequency_x1_y[(x1, y)] / frequency_y[y]
        
        # P(X2|Y)
        prob_x2_given_y = frequency_x2_y[(x2, y)] / frequency_y[y]
    
        prob_y = frequency_y[y] / len(data)
        z = frequency_x1_y[(x1, 0)] / frequency_y[0] * \
            frequency_x2_y[(x2, 0)] / frequency_y[0] * \
            frequency_y[0] / len(data) + frequency_x1_y[(x1, 1)] / frequency_y[1] * \
            frequency_x2_y[(x2, 1)] / frequency_y[1] * \
            frequency_y[1] / len(data) 
       
        # P(Y|X1,X2) = P(X1|Y) * P(X2|Y) * P(Y) / P(X1,X2)
        prob_y_given_x1_x2 = prob_x1_given_y * prob_x2_given_y * prob_y / z    
        prob_y_given_x1 = frequency_x1_y[(x1, y)] / frequency_x1[x1]
        prob_y_given_x2 = frequency_x2_y[(x2, y)] / frequency_x2[x2]
        prob_x1 = frequency_x1[x1] / len(data)
        prob_x2 = frequency_x2[x2] / len(data)
        print(f'Option 1: P(Y|X1={x1},X2={x2})={prob_y_given_x1_x2:.4f}')

        # P(Y|X1,X2) = P(Y|X1) * P(X1) * P(Y|X2) * P(X2) / (P(X1,X2) * P(Y))
        prob_y_given_x1_x2 = prob_y_given_x1 * prob_x1 * prob_y_given_x2 * prob_x2 / (z * prob_y)
        print(f'Option 2: P(Y|X1={x1},X2={x2})={prob_y_given_x1_x2:.4f}')


if __name__ == '__main__':
    print('Conditional independent')
    data = [[0,0], [0,1], [1,0], [1,0]]
    labels = [0, 0, 1, 0]
    compute_probs(data, labels)