import sys

input = sys.stdin.readline
N = int(input())

# 기본 피보나치
def fibo(n):
    if n == 0 : return 0
    elif n == 1 : return 1
    
    return fibo(n-1) + fibo(n-2)
print(fibo(N))

# 한번 방문한 곳은 재귀에 빠지지 않게 한다.
fibo_c = [-999999] * (N+1)
def fibo_memozation(n):
    if fibo_c[n] != -999999 : return fibo_c[n]
    if n == 0 : 
        return 0
    elif n == 1 : 
        return 1
    fibo_c[n] = fibo(n-1) + fibo(n-2)
    return fibo_memozation(n-1) + fibo_memozation(n-2)


print(fibo(N))
