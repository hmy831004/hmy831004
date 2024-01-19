# https://www.acmicpc.net/problem/2559
'''
투포인터를 처음부터 생각하지 말고 기본 for문으로 풀수 있는지를 시간 복잡도로 계산해보고 시간 복잡도가 초가 되면
연속되는 배열의 값을 처리 할 수 있는지에 대해서 투포인터를 떠올려서 계산.
문제 : 2<= N <= 1e5  , 1 <= K <= N 
초기 아이디어 -> 각 for문이 시작하면 K개의 숫자들 더한다. 
for문에서 O(N), for문 안에서 연산 K ->O(K) , 시간 복잡도 = O(NK) = 1e10 으로써 2억(2e8)이 넘어감
시간 복잡도가 넘어 가기 때문에 다른 아이디어가 있는지 생각해보고 투포인터를 떠올림

1. 아이디어
- 투포인터를 활용
- for문으로, 처음에 k개값을 저장
- 다음인덱스 더해주고, 이전 인덱스 빼줌
- 이때마다 최대값을 갱신
2. 시간복잡도
- O(N) = 1e5 >  가능

3. 자료구조
 - 각 숫다르 N개 저장 배열 :int[]
    - 숫자들 최대 100 > INT 가능 , int
 - K개의 값을 저장하는 변수 : int
    -  최대 : K * 100 = 1e5 * 100 = 1e7 > INT 가능
 - 최대값 ,: int
'''


import sys
input = sys.stdin.readline

N,K = map(int,input().split())
nums = list(map(int,input().split()))
each = 0
# K개를 더해주기
for i in range(K):
    each += nums[i]
maxv = each
# 다음 인덱스 더해주고, 이전인덱스 빼주기
for i in range(K,N):
    each +=nums[i] # each에 새로 포인팅하고 있는 부분을 더해줌
    each -= nums[i-K] # each를 더했던 변수의 처음 포인터 부분을 빼줌
    maxv = max(maxv,each) 

print(maxv)