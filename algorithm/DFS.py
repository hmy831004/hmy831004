# https://www.acmicpc.net/problem/2667
# DFS 알고리즘 : 
# 아이디어 : 2중 for , 값 1  && 방문x => DFS
# 시간복잡도 : O(V+E)
# V = N ^ 2 , E = 4 * N ^2 , V +E = 5N^2 ~= N^2 = 625
# 자료구조 : 그래프 저장 : int[][] , 방문여부: bool[][], 결과값 : int[]


import sys
input = sys.stdin.readline

N = int(input())
# readline 으로 읽을 시에 마지막에 '\n' 개행문자 붙기 때문에 제거
maps = [list(map(int, input().strip())) for _ in range(N)]
chk = [[False]* N for _ in range(N)]
result = []
each = 0
# 4방향, 오른쪽, 이래쪽, 왼쪽, 위쪽
dy = [0,1,0,-1]
dx = [1,0,-1,0]
def DFS(y,x):
    global each 
    each +=1
    for k in range(4):
        ny = y + dy[k]
        nx = x + dx[k]
        if 0 <= ny < N and 0<= nx <N:
            if maps[ny][nx] == 1 and chk[ny][nx] == False:
                chk[ny][nx] = True
                DFS(ny,nx)

for j in range(N):
    for i in range(N):
        if maps[j][i] == 1 and chk[j][i] == False:
            chk[j][i] = True
            each = 0
            DFS(j,i)
            result.append(each)

result.sort()
print(len(result))
for i in result:
    print(i)