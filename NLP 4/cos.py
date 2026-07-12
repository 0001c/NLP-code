import math

def cos_sim(a, b):
    # 点积
    dot = sum(x * y for x, y in zip(a, b))
    # 向量模长
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(y**2 for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 零向量避免除0
    return dot / (norm_a * norm_b)

if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    print(cos_sim(vec1, vec2))
