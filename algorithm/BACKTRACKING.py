# https://www.acmicpc.net/problem/15649
"""
1. 아이디어
 - 백트레킹 재귀함수 안에서, for 돌면서 숫자 선택(이때 방문여부 확인)
 - 재귀함수에서 M개를 선택할 경우 print
2. 시간복잡도
 - N! , 10까지 가능함(중복 없을때) , 중복 있을때는 8 까지 가능
3. 자료구조
 - 결과값 저장 int[]
 - 방문여부 체크 bool[]

뺵트레킹에서 4,3 의 중복되지 않는 순열이라고 하면
뒤에 '3'의 의미는 recursive의 depth라고 생각하면되고, 앞의 4는 [1,2,3,4] 까지의 숫자 라고 생각하면됨

"""


import sys
input = sys.stdin.readline

N,M = map(int,input().split())
rs = []
chk = [False] * (N+1) # 처음 인덱스 안쓰고 나머지 인덱스를 rs의 값에 따라 바로 사용하기 위해 N+1로함

def recur(num):
    if num == M:
        print(' '.join(map(str,rs)))
        return 
    for i in range(1,N+1):
        if chk[i] == False:
            chk[i] = True
            rs.append(i)
            recur(num+1)
            chk[i] = False
            rs.pop()
recur(0)