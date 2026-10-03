import math

def mean(x):
    return sum(x) / len(x)

def var(x):
    μ = mean(x)
    return mean((xi - μ) ** 2 for xi in x)

def std(x):
    return math.sqrt(var(x))

# Example usage:
if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print("Mean:", mean(data))
    print("Variance:", var(data))
    print("Standard Deviation:", std(data))
