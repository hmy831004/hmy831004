# https://www.acmicpc.net/problem/1753
'''
다익스트라 알고리즘 : 최단 경로 탐색 알고리즘(GPS에서사용), 
    한 노드에서 다른 모든 노드까지 가는데 최소비용, 
작동 원리 : 
 - 간선 : 인접리스트, 거리 배열: 초기값 무한대로 설정 해서 최소 거리가 나오면 갱신, 
 - 힙에 시작점 추가
 - 힙에서 현재 노드 빼면서, 간선 통할 때 더 거리 짧아진다면 거리 갱싱 및 힙에 추가, 거리가 안짧아진다면 heap에 추가 안함

아이디어 :
 - 한점에서 다른 모든 점으로의 최단경로 > 다익스트라
 - 모든 점 거리 초기값 무한대로 설정
 - 간선, 인접리스트 저장
 - 시작점 거리 0 설정 및 힙에 추가 !!
 - 힙에서 하나씩 빼면서 수행할 것
    - 최신 값인지 확인
    - 간선을 타고 간 비용이 더 작으면 갱신
    - 새로운 거리 힙에 추가. !! (새로운 경로를 발견하지 않으면 힙에 추가안함)

시간 복잡도 : 
 - 다익스트라 시간복잡도 : O(ElogV)
    - E(간선수): 3e5, logV = 20 -> log10^6 ~= 20
    - E : 3e5
    - V : 2e4, lgV ~= 20
    - ElgV = 6e6 > 가능
 - O(ElogV) = 6e6 > 가능

변수 :
 - 다익스트라 사용 힙 : [비용(int),다음노드(int)]
  - 비용 최대값 : 10 * 2e4 = 2e5 -> INT 사용가능
  - 다음 노드 : 2e4 -> INT 가능
 - 거리배열 : int[]
  - 거리 최대값 : 10 * 2e4 = 2e5 -> INT 가능
 - 간선 저장, 인접리스트(비용,노드번호)

'''

import sys 
import heapq
input = sys.stdin.readline
INF = sys.maxsize

V,E = map(int,input().split())
K = int(input())

edge = [[] for _ in range(V+1)]
dist = [INF] * (V+1)
for i in range(E):
    u,v,w = map(int,input().split())
    edge[u].append([w,v])

# 시작점 초기화
dist[K] = 0
heap = [[0,K]]
while heap:
    # each weight, each vertex
    ew,ev = heapq.heappop(heap)
    # 최신 값인지 확인, 같아야지 최신값이고 그 아래 코드를 진행.
    if dist[ev] != ew: continue
    for nw, nv in edge[ev]:
        # 현재 저장된 nv노드로 가는 비용이, ew(nv 노드로 가기 전 까지의 노드의 거리비용) + nw(nv 노드로 가는 비용) 보다 더 크다면 최소 거리를 업데이트
        if dist[nv] > ew + nw: 
            dist[nv] = ew + nw
            heapq.heappush(heap,[dist[nv],nv])


for i in range(1,V+1):
    if dist[i] == INF: print("INF")
    else : print(dist[i])