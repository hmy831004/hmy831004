# https://www.acmicpc.net/problem/1197

'''
MST : Minimum Spanning Tree, 그래프를 연결하는 최소 비용 구하는 문제, 보통 양방향 그래프로 나옴(단방향으로 안나옴)
 - Kruskal : 전체 간선 중 작은 것 부터 연결
 - Prim : 현재 연결된 트리에 이어진 간성중 가장 작은것을 추가
 - Prim을 사용하기 위해서 heap 자료 구조를 사용 해야함, '현재 연결된'트리에 이어진 간선중 가장 작은것을 추가(Heap에 넣으면 자동으로 계산)
heap 
 - 최대값,최소값을 빠르게 계산하기 위한 자료구조
 - 이진 트리 구조 
 - 처음에 저장할때부터 최대값 or 최소값 인지 확인함
 - Heap 삽입 삭제는 LogE 시간복잡도를 가짐 


아이디어
 - MST 기본문제, 외우기
 - 간선을 인접리스트에 집어넣기
 - 힙에 시작점 넣기
 - 힙이 빌때까지 다음의 작업을 반복
    - 힙의 최소값 꺼내서, 해당 노드 방문 안했다면
        - 방문표시, 해당 비용 추가, 연결된 간선들 힙에 넣어주기
    

시간복잡도(MST)
 - MST : O(ElogE)
 - Edge 리스트에 저장: O(E)
 - Heap안 모든 Edge에 연결된 간선확인 : O(E+E)
 - 모든 간선 힙에 삽입: O(ElogE)
 - O(E+2E+ElogE) = O(3E+ElogE) = O(E(3+logE)) = O(ElogE)

자료구조
 - 간선 저장 되는 인접리스트 : (무게, 노드번호)
 - heap : (무게, 노드번호)
 - 방문 여부 : bool[]
 - MST 결과값 : int
'''

import sys
import heapq
input = sys.stdin.readline

V,E = map(int,input().split())
edge = [[] for _ in range(V+1)]
for i in range(E):
    a,b,c = map(int,input().split())
    edge[a].append([c,b])
    edge[b].append([a,b])


heap = [[0,1]]
chk = [False] * (V+1)
rs = 0
while heap:
    w, next_node= heapq.heappop(heap)
    if chk[next_node] == False:
        chk[next_node]=True
        rs += w
        for next_edge in edge[next_node]:
            if chk[next_edge[1]] == False:
                heapq.heappush(heap,next_edge)
print(rs)

'''
MST 문제인지 알기 위한 팁
모든 노드가 연결되도록 한다거나, 이미 연결된 노드를 최소의 비용으로 줄인다 라는 말이 나오면 MST
'''
