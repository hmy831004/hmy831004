# 문자열 문제들
# https://www.acmicpc.net/workbook/view/9432
# https://velog.io/@codusl100/%EB%B0%B1%EC%A4%80-%EB%AC%B8%EC%9E%90%EC%97%B4-%EB%AC%B8%EC%A0%9C-%EC%A0%95%EB%A6%AC
# https://star7sss.tistory.com/263
# 회문(팰린드롬), 문자열 뒤집기, 조건에 맞게 재정렬, 특정 단어 추출, 애너램(문자를 재배열해 다른뜻을 가진 단어로 바꿈), 가장 긴 팰린드롬 찾기

# 팰린드롬 만들기 : https://www.acmicpc.net/problem/1254
"""
아이디어
 - 문자가 팰린드롬인지 확인한다.
 - 
시간복잡도
변수 
"""
import sys
import re
from collections import Counter,defaultdict

input = sys.stdin.readline

def ispalindrome(str):
    str = str.lower()
    str = re.sub("[^a-z0-9]","",str)
    str= list(str)
    
    while len(str) > 1:
        if str.pop(0) == str.pop(): continue
        else :
            return '회문이 아닙니다.'
    return "회문입니다."
def ispalindrome2(str):
    str = str.lower()
    str = re.sub("[^a-z0-9]","",str)
    str_back = str[::-1]
    if str == str_back:
        return '회문입니다.'
    return '회문이 아닙니다.'
def reverseString(str):
    """
    투 포인터를 이용해서 문자열 정렬
    """
    str = list(str)
    left_idx, right_idx = 0, len(str)-1
    while left_idx < right_idx:
        str[left_idx], str[right_idx] = str[right_idx], str[left_idx]
        left_idx  +=1
        right_idx -=1
    return str

def condition_sort(str_list):
    """
        조건에 맞게 문자열 재정렬, sort의 key 함수를 사용해서 어떤 키로 정렬 할 것인지 결정.
    """
    str_list =  ['1 C','1 A', '1 B', '6 A', '2 D', '4 B']
    str_list = [x.split() for x in str_list]
    str_list = sorted(str_list,key=lambda x : x[1])
    return str_list
def word_count_widh_bandword(str,ban_word):
    """
        ban_word가 아닌 문자중에 가장 많이 등장한 문자, 혹은 등장 횟수 구하기
    """
    ban_word = "hit"
    paragraph = "Bob hit a ball, the hit BALL flew far after it was hit"
    str = str.lower()
    str = re.sub("[^\w]"," ",str) # 구두점 제거
    str = [x for x in str.split() if x != ban_word]
    counts = Counter(str)
    print(counts.most_common()[0][1])

def anagrams(str_list):
    """
        아이디어 : 모든 텍스트를 정렬한 것을 defaultdict의 key로 넣고 valu는 정렬되기 전의 값을 넣으면 정렬된 key에 각 value가 그룹지어 져서 저장 되게 된다.
    """
    # data = ["eat","tea","tan","ate","nat","bat"]
    sort_data = defaultdict(list)
    for word in str_list:
        sort_data[''.join(sorted(word))].append(word)
    print(sort_data.values())

def max_palindrome(str):
    """
    아이디어 : str을 모두 순회 하면서 가장 긴 팰린드롬을 찾는다. 
    """
    str = list(str)
    max_n = 0
    for i in range(len(str)):
        for j in range(len(str),i,-1):
            tmp = str[i:j]
            if tmp == tmp[::-1]:
                max_n = max(max_n,len(tmp))
    
    return max_n

        
# input_str = input().strip()
# print(ispalindrome(input_str))
# print(reverseString(input_str))    

anagrams(["eat","tea","tan","ate","nat","bat"])
