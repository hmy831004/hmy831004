# https://www.acmicpc.net/problem/11047

'''
GREEDY 문제의 아이디어
-  큰 금액의 동전부터 차감 
- 그리디로 해도 되는가? 라는 물음에 대해서는 반례를 통해 예시를 들어봐라: 동전의 개수가 무한대라서 없는것으로 보임
 
1. 이번 문제 아이디어 
 - 동전을 저장한뒤, 반대로 뒤집음 (큰 동전부터 차감 할 것이기 때문에)
 - 동전 for > 
    - 동전 사용개수 추가
    - 동전 사용한 만큼 K값 갱신

시간복잡도
 -  for: N > O(N)
자료구조
 - 동전금액 : int[] , 최대값 : 1e6 > INT 가능
 - 현재남은금액: int, 최대값 : 1e8 > INT가능 , K
 - 동전개수 : int, 최대값 : 1e8 > INT 가능 , cnt
'''

import sys
input = sys.stdin.readline

N,K = map(int,input().split())
nums = [int(input()) for _ in range(N)]
nums.sort(reverse=True)

cnt = 0 
for n in nums:
    if n <= K:
        cnt = cnt + K // n
    K = K % n
    if K == 0 :
        break

print(cnt)