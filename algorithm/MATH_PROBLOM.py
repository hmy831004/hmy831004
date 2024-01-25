
# 약수, N진수, GCD, LCM, 소수 
import sys
input = sys.stdin.readline



# 최대 공약수, 최소 공배수 일반 접근
#N,M = map(int,input().split())
# min_n = min(N,M)
# for n in range(min_n,0,-1):
#     if N % n == 0 and M % n == 0 : 
#         print(n)
#         print(n * (N//n)*(M//n))
#         break
    
# 유클리드 호재법 으로 접근한 GCD(Greatest Common Divisor) , LCD(Least Common Multiple)
def GCD(x,y):
    while y:
        x = y
        y = y % x 
    return x

def LCM(x,y):
    result = (x*y)//GCD(x,y)
    return result

# N번째 큰수 - https://www.acmicpc.net/problem/2693
# T = int(input())
# A = [sorted(list(map(int,input().split()))) for _ in range(T)]
# for i in range(T):
#     print(A[i][-3])

# 소수 구하기 - https://www.acmicpc.net/problem/1978
# N = int(input())
# nums = list(map(int,input().split()))
# count = 0 
# for num in nums:
#     if num == 1 : continue
#     flag = True
#     for i in range(2,num):
#         if num % i == 0 : 
#             flag = False
#             break
#     if flag : count+=1
# print(count)

# 쉽게 푸는 문제 : https://www.acmicpc.net/problem/1292
# N = 45
# A,B =  map(int,input().split())
# nums = [[i]*i for i in range(1,N+1)]
# total = []
# for num in nums:
#     total.extend(num)
# print(sum(total[(A-1):B]))

# 소수 : https://www.acmicpc.net/problem/2581
# 범위에 소수를 모두 구하는 문제
# M = int(input())
# N = int(input())

# def iss(num):
#     count = 0
#     if num == 1 : return False
#     for i in range(2,(num//2)+1) :
#         if num % i ==0 : 
#             count +=1
#             break
#     return True if count == 0 else False
# iss_n = [] 
# for n in range(M,N+1):
#     if iss(n):
#         iss_n.append(n)

# print("-1") if len(iss_n) == 0  else print(f"{sum(iss_n)}\n{min(iss_n)}")

# 소수 : https://www.acmicpc.net/problem/2581
# 에라토스테네스의 체를 이용해 N까지의 모든 소수를 구하고 거기서 부터 소수를 구하기.
M = int(input())
N = int(input())
a = [False,False] + [True]*(N-1)
primes=[]
for i in range(2,N+1):
  if a[i]:
    primes.append(i)
    for j in range(2*i, N+1, i):
        a[j] = False
idx = -1
for i,prime in enumerate(primes):
   if prime >= M:
      idx = i
      break
if idx == -1 :
   print(idx)
else:
    primes = primes[idx:]
    print(f'{sum(primes)}\n{min(primes)}')





