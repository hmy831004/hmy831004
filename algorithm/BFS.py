
# https://www.acmicpc.net/problem/1926 

# 아이디어 - 1이 이어지면 숫자를 키운다.(BFS ) , 이중 포문 돌면서 하나씩 검사(반복이 일어나지 않도록 검사), 
# 시간복잡도 - BFS 시간복잡도 = O(V+E),  V = m * n ,   E = V * 4 , O(V+E) = 5V -> 5* m * n , 5 * 500 * 500 = 100만 < 1초당 2억개 연산가능이기 때문에 통과
# 자료구조 - 그래프 전체 지도(이차원배열, int[][]) , 방문여부(이차원패열 bool[][])



import sys
from collections import deque
input = sys.stdin.readline


n,m = map(int,input().split())
maps = [list(map(int,input().split())) for _ in range(n)]
chk = [[False] * m  for _ in range(n)]

dy = [0,1,0,-1]
dx = [1,0,-1,0]
def BFS(y,x):
    rs = 1 

    q = deque()                           
    q.append((y,x))
    while q :
        ey,ex = q.popleft()
        for k in range(4):
            ny = ey + dy[k]
            nx = ex + dx[k]
            if (0 <=ny < n) & (0 <= nx < m): 
                if (maps[ny][nx] == 1) & (chk[ny][nx] == False):
                    chk[ny][nx] = True
                    rs +=1 
                    q.append((ny,nx))
    return rs 

max_v = 0
cnt = 0 
for j in range(n):
    for i in range(m):
        if (maps[j][i] == 1) & (chk[j][i] == False) :
            cnt +=1 
            chk[j][i] = True
            max_v = max(max_v,BFS(j,i))
        
print(cnt)
print(max_v)