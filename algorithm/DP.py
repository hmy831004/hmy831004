# https://www.acmicpc.net/problem/11726
'''
DP : Dynamic Programming - 이전의 값을 재활용 하는 알고리즘
DP의 핵심은 점화식을 구하는 것 : 예제: An = An-1 + An-2 
점화식을 찾기 위해선 N 몇개를 해보면 규칙을 찾을 수 있음

아이디어
 - 점화식 : An = An-1 + An-2
 - N값 구하기 위해, for문 3부터 N까지의 값을 구해주기
 - 이전값과 이전이전값 더해서, 10007로 나눈 나머지 값 저장

시간복잡도
 - for N : O(N)
자료구조
 - DP값 저장하는 (경우의수) 배열: int[], 최대값 : 10007보다 작음 : INT 사용가능

'''

import sys 
input = sys.stdin.readline

n = int(input())
rs = [0,1,2]

for i in range(3,n+1):
    rs.append((rs[i-1] + rs[i-2])%10007) # 이 점화식을 세울 수 있는가가 핵심, 하나씩 찾아봐라
print(rs[n])