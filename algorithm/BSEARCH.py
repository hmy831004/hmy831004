# https://www.acmicpc.net/problem/1920
'''
N = (1 ≤ N ≤ 100,000), M = (1 ≤ M ≤ 100,000)
처음 아이디어 : 
 - M개의 수마다 각각 어디에 있는지 찾기
 - for : M개의 수
 - for : N개의 수 안에 있는지 확인 
처음 시간 복잡도:
 - for : M개의수 > O(M)
 - for : N개의 수안에 있는지 확인 > O(N)
 - O(MN) = 1e10 > 시간초과(2e8)

 그럼 어떻게 풀지?? 
 투포인터?? 연속하다는 특징이 없어서 사용 불가 
 정렬해서 이진탐색이 가능한가? 
  - N개의 수 먼저 정렬
  - M개의 수 하나씩 이진탐색으로 확인

  N개의 수 정렬 : O(N *logN) 정렬은 항상 이것
  M개의 수 이진탐색 : O(M *logN)
  O((N+M)logN) = 2e5*20 = 4e6 
  왜 20이 나왔나 log2N = log2(10^5)인데 계산의 편의상 log2(10^6)이라고둠.
  2^10 ~= 10^3 -> 10의6승을 만들기 위해 각 식에 제곱 = (2^10)^2 ~= (10^3)^2
  log2(10^6) ~= log2(2^20) = 20

1. 아이디어
 - N개의 숫자를 정렬
 - M개를 for 돌면서, 이직탐색
 - 이진탐색 안에서 마지막에 데이터를 찾으면, 1출력, 아니면 0 출력
2. 시간복잡도
 - N개 입력값 정렬 = O(NlogN)
 - M개를 N개중에서 탐색 = O(M* logN)
 - 총합 : O((N+M)logN) > 가능
3. 자료구조
 - 탐색 대상수 : int[]
    - 모든 수 범위 : -2^31 ~ 2^31 > INT 가능
 - 탐색 하려는 수: int[]
    - 모든 수 범위 : -2^31 ~ 2^31 > INT 가능

'''

import sys
input = sys.stdin.readline

N = int(input())
nums = list(map(int,input().split()))
M = int(input())
target_list = list(map(int,input().split()))
nums.sort() # 이진탐색 가능 

def search(st,en,target):
    if st== en:
        if nums[st] == target:
            print(1)
        else:
            print(0)
        return
    mid = (st+en)//2 # 작은 인덱스를 선택, 나머지를 버려서
    if nums[mid] < target:
        search(mid+1,en,target)
    else:
        search(st,mid,target)

# 타겟마다 이진탐색을 해서 찾는 숫자가 있는지 없는지
for each_target in target_list:
    search(0,N-1,each_target)